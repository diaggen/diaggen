# pylint: disable=no-value-for-parameter

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import igl
import quadrants as qd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.solver_health import (
    FEMPrincipalStrainWitness,
    FEMSubstepSafetyExtrema,
    ImplicitFEMPositiveJFeasibleStep,
    ImplicitFEMTrueResidualProbe,
    ImplicitFEMTrueResidualSample,
)
from genesis.engine.states.solvers import FEMSolverState
from genesis.utils.misc import qd_to_torch, tensor_to_array
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat

from .base_solver import Solver
from .fem_coarse_space import (
    build_material_connected_partition_of_unity,
    build_weighted_rigid_motion_candidates,
)

if TYPE_CHECKING:
    from genesis.engine.entities import FEMEntity

DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_FLOOR = 0.20
DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_SCHEDULE_LENGTH = 8
TRUE_RESIDUAL_PROBE_SCHEDULE = (0, 50, 100, 250, 500)


def _linear_corotated_safety_extrema(
    *,
    rest_positions: np.ndarray,
    tetrahedra: np.ndarray,
    current_positions: np.ndarray,
    active: np.ndarray,
    lame_mu: np.ndarray,
    lame_lambda: np.ndarray,
    global_substep_index: int | None = None,
    env_index: int = 0,
    fem_entity_index: int | None = None,
    fem_entity_name: str | None = None,
    vertex_global_offset: int = 0,
    tet_global_offset: int = 0,
    floor_height_m: float | None = None,
) -> FEMSubstepSafetyExtrema | None:
    """Reduce one volumetric linear-corotated FEM state.

    The optional metadata enables one bounded public argmax witness for the
    qualification-only B=1 path.  Omitting it preserves the scalar-only
    source-compatible helper behavior used by unrelated consumers.
    """
    rest = np.asarray(rest_positions, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    current = np.asarray(current_positions, dtype=np.float64)
    active_mask = np.asarray(active, dtype=np.bool_)
    mu = np.asarray(lame_mu, dtype=np.float64)
    lam = np.asarray(lame_lambda, dtype=np.float64)
    if (
        rest.ndim != 2
        or rest.shape[1] != 3
        or current.ndim != 3
        or current.shape[1:] != rest.shape
        or tets.ndim != 2
        or tets.shape[1] != 4
        or active_mask.shape != (current.shape[0], tets.shape[0])
        or mu.shape != (tets.shape[0],)
        or lam.shape != (tets.shape[0],)
        or not np.isfinite(rest).all()
        or not np.isfinite(current).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(lam).all()
        or np.any(mu <= 0.0)
        or np.any(lam < 0.0)
        or np.any(tets < 0)
        or np.any(tets >= rest.shape[0])
    ):
        raise ValueError("implicit FEM safety extrema received malformed volumetric state")
    if not np.any(active_mask):
        return None

    rest_edges = np.stack(
        (rest[tets[:, 0]] - rest[tets[:, 3]], rest[tets[:, 1]] - rest[tets[:, 3]], rest[tets[:, 2]] - rest[tets[:, 3]]),
        axis=2,
    )
    rest_det = np.linalg.det(rest_edges)
    if not np.isfinite(rest_det).all() or np.any(rest_det == 0.0):
        raise ValueError("implicit FEM safety extrema require nondegenerate rest tetrahedra")
    rest_inverse = np.linalg.inv(rest_edges)
    rest_volume = np.abs(rest_det) / 6.0

    current_edges = np.stack(
        (
            current[:, tets[:, 0]] - current[:, tets[:, 3]],
            current[:, tets[:, 1]] - current[:, tets[:, 3]],
            current[:, tets[:, 2]] - current[:, tets[:, 3]],
        ),
        axis=3,
    )
    deformation = current_edges @ rest_inverse
    jacobian = np.linalg.det(deformation)
    singular_values = np.linalg.svd(deformation, compute_uv=False)
    principal_strain = np.max(np.abs(singular_values - 1.0), axis=2)
    # Match ``Elastic._pre_compute_linear_corotated`` exactly. In particular,
    # an improper polar factor is not corrected here: J <= 0 is separately
    # retained as the inversion hard-safety evidence.
    u, _, vh = np.linalg.svd(deformation)
    rotation = u @ vh
    f_hat = np.swapaxes(rotation, 2, 3) @ deformation
    epsilon = 0.5 * (f_hat + np.swapaxes(f_hat, 2, 3)) - np.eye(3, dtype=np.float64)
    trace = np.trace(epsilon, axis1=2, axis2=3)
    density = mu[None] * np.sum(epsilon * epsilon, axis=(2, 3)) + 0.5 * lam[None] * trace * trace
    energy = density * rest_volume[None]
    if not np.isfinite(jacobian).all() or not np.isfinite(principal_strain).all() or not np.isfinite(energy).all():
        raise ValueError("implicit FEM safety extrema computation produced a non-finite value")

    selected = active_mask
    selected_jacobian = jacobian[selected]
    selected_strain = principal_strain[selected]
    selected_energy = energy[selected]
    witness = None
    if (
        current.shape[0] == 1
        and global_substep_index is not None
        and fem_entity_index is not None
        and fem_entity_name is not None
        and floor_height_m is not None
    ):
        if type(global_substep_index) is not int or global_substep_index < 0:
            raise ValueError("global_substep_index must be a nonnegative int")
        if type(env_index) is not int or env_index != 0:
            raise ValueError("qualification FEM witness requires env_index == 0")
        if type(fem_entity_index) is not int or fem_entity_index < 0 or not isinstance(fem_entity_name, str) or not fem_entity_name:
            raise ValueError("FEM witness entity identity is malformed")
        if type(vertex_global_offset) is not int or vertex_global_offset < 0 or type(tet_global_offset) is not int or tet_global_offset < 0:
            raise ValueError("FEM witness global offsets must be nonnegative ints")
        if type(floor_height_m) is not float or not np.isfinite(floor_height_m):
            raise ValueError("floor_height_m must be a finite float")
        active_indices = np.argwhere(active_mask[0])[:, 0]
        max_value = float(np.max(principal_strain[0, active_indices]))
        tied = active_indices[principal_strain[0, active_indices] == max_value]
        tet_local_index = int(np.min(tied))
        tet_local_vertices = tuple(int(value) for value in tets[tet_local_index])
        vertex_positions = tuple(
            tuple(float(component) for component in current[0, vertex_index]) for vertex_index in tet_local_vertices
        )
        centroid = tuple(float(sum(vertex[axis] for vertex in vertex_positions) / 4.0) for axis in range(3))
        floor_candidate = any(vertex[2] > floor_height_m for vertex in vertex_positions) and any(
            vertex[2] <= floor_height_m for vertex in vertex_positions
        )
        witness = FEMPrincipalStrainWitness(
            global_substep_index=global_substep_index,
            env_index=env_index,
            fem_entity_index=fem_entity_index,
            fem_entity_name=fem_entity_name,
            tet_local_index=tet_local_index,
            tet_global_index=tet_global_offset + tet_local_index,
            tet_entity_local_vertex_indices=tet_local_vertices,
            tet_global_vertex_indices=tuple(vertex_global_offset + value for value in tet_local_vertices),
            current_vertex_positions_m=vertex_positions,
            current_centroid_m=centroid,
            principal_stretch_strain=max_value,
            floor_height_m=floor_height_m,
            floor_tet_candidate_geometrically_possible=floor_candidate,
        )
    return FEMSubstepSafetyExtrema(
        min_j=float(np.min(selected_jacobian)),
        max_principal_stretch_strain=float(np.max(selected_strain)),
        max_tet_elastic_energy_j=float(np.max(selected_energy)),
        total_elastic_energy_j=float(np.sum(selected_energy, dtype=np.float64)),
        no_inversion=bool(np.min(selected_jacobian) > 0.0),
        principal_strain_witness=witness,
    )


def _pcg_effective_residual_squared_threshold(
    *, initial_residual_squared: float, absolute_residual_squared_floor: float, pcg_rtol: float
) -> float:
    """Return the documented PCG stopping threshold for one zero-start solve."""
    values = (initial_residual_squared, absolute_residual_squared_floor, pcg_rtol)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("PCG stopping inputs must be finite and nonnegative")
    return max(absolute_residual_squared_floor, initial_residual_squared * pcg_rtol * pcg_rtol)


@qd.data_oriented
class FEMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._enable_floor = options.enable_floor
        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        self._linear_solver = options.linear_solver
        self._sparse_direct_backend = os.environ.get("GENESIS_FEM_SPARSE_DIRECT_BACKEND", "scipy")
        if self._sparse_direct_backend not in ("scipy", "cudss"):
            raise ValueError(
                "GENESIS_FEM_SPARSE_DIRECT_BACKEND must be either 'scipy' or 'cudss'"
            )
        self._cudss_anchor_age = int(os.environ.get("GENESIS_FEM_CUDSS_ANCHOR_AGE", "0"))
        self._cudss_pcg_max_iterations = int(
            os.environ.get("GENESIS_FEM_CUDSS_PCG_MAX_ITERATIONS", "6")
        )
        self._cudss_pcg_rtol = float(os.environ.get("GENESIS_FEM_CUDSS_PCG_RTOL", "1e-8"))
        self._cudss_gpu_assembly = os.environ.get("GENESIS_FEM_CUDSS_GPU_ASSEMBLY") == "1"
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._pcg_rtol = options.pcg_rtol
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._enable_rigid_mode_deflation = options.enable_rigid_mode_deflation
        self._enable_material_coarse_preconditioner = options.enable_material_coarse_preconditioner
        self._true_residual_probe_global_substep = options.true_residual_probe_global_substep
        self._enable_vertex_constraints = options.enable_vertex_constraints
        self._enable_qualification_safety_extrema = options.enable_qualification_safety_extrema
        self._enable_development_implicit_fem_positive_j_feasible_step = (
            options.enable_development_implicit_fem_positive_j_feasible_step
        )
        self._enable_development_implicit_fem_positive_j_alpha_one_only = (
            options.enable_development_implicit_fem_positive_j_alpha_one_only
        )
        self._enable_development_direct_replay_min_j_query = (
            options.enable_development_direct_replay_min_j_query
        )

        # CPU sparse-direct state is populated once per implicit physical
        # substep.  SAP consumes the velocity-unit matrix and momentum data
        # without reconstructing a second FEM operator.
        self._direct_velocity_matrices = ()
        self._direct_velocity_matrix = None
        self._direct_velocity_factors = ()
        self._direct_velocity_gpu_factors = {}
        self._direct_velocity_cudss_factors = []
        self._direct_velocity_cudss_ages = []
        self._direct_cudss_factorizations = 0
        self._direct_cudss_pcg_solves = 0
        self._direct_cudss_pcg_iterations = 0
        self._direct_cudss_pcg_fallbacks = 0
        self._direct_cudss_last_relative_residual = 0.0
        self._direct_cudss_max_relative_residual = 0.0
        self._direct_gpu_template = None
        self._direct_gpu_lower_contribution_mask = None
        self._direct_gpu_lower_contribution_indices = None
        self._direct_gpu_diagonal_indices = None
        self._direct_gpu_static_tensors = None
        self._direct_velocity_rhs = None
        self._direct_free_velocity = None
        self._direct_free_residual = None
        self._direct_free_residual_norm = None
        self._direct_free_positions = None
        self._direct_geometry_candidates = None
        self._direct_assembly_time_s = 0.0
        self._direct_factor_time_s = 0.0
        self._direct_solve_time_s = 0.0
        self._direct_transfer_time_s = 0.0
        self._material_connected_pou_cache = {}

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_stress = list()
        self._mats_compute_energy_gradient_hessian = list()
        self._mats_compute_energy = list()

        # boundary
        self.setup_boundary()

        # lazy initialization
        self._constraints_initialized = False

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_batch_fields(self):
        self.batch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_iterations = qd.field(dtype=gs.qd_int, shape=(self._B,), needs_grad=False)
        self.batch_linesearch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_budget_exhausted = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_breakdown = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_linesearch_budget_exhausted = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        if self._true_residual_probe_global_substep is not None:
            probe_shape = (len(TRUE_RESIDUAL_PROBE_SCHEDULE), self._B)
            self.true_residual_probe_actual_iterations = qd.field(
                dtype=gs.qd_int, shape=probe_shape, needs_grad=False
            )
            self.true_residual_probe_pcg_active = qd.field(
                dtype=gs.qd_bool, shape=probe_shape, needs_grad=False
            )
            self.true_residual_probe_true_rTr = qd.field(
                dtype=gs.qd_float, shape=probe_shape, needs_grad=False
            )
            self.true_residual_probe_recursive_rTr = qd.field(
                dtype=gs.qd_float, shape=probe_shape, needs_grad=False
            )

        pcg_state = qd.types.struct(
            rTr=gs.qd_float,
            rTr_initial=gs.qd_float,
            termination_threshold=gs.qd_float,
            rTz=gs.qd_float,
            rTr_new=gs.qd_float,
            rTz_new=gs.qd_float,
            pTAp=gs.qd_float,
            alpha=gs.qd_float,
            beta=gs.qd_float,
        )
        self.pcg_state = pcg_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

        linesearch_state = qd.types.struct(
            prev_energy=gs.qd_float,
            energy=gs.qd_float,
            step_size=gs.qd_float,
            m=gs.qd_float,
        )
        self.linesearch_state = linesearch_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

        if (
            self._enable_development_implicit_fem_positive_j_feasible_step
            and not self._enable_development_implicit_fem_positive_j_alpha_one_only
        ):
            self._development_implicit_fem_positive_j_base_min_j = qd.field(
                dtype=gs.qd_float, shape=(self._B,), needs_grad=False
            )
            self._development_implicit_fem_positive_j_trial_min_j = qd.field(
                dtype=gs.qd_float, shape=(self._B,), needs_grad=False
            )
            self._development_implicit_fem_positive_j_accepted_alpha = qd.field(
                dtype=gs.qd_float, shape=(self._B,), needs_grad=False
            )
            self._development_implicit_fem_positive_j_witness_tet_id = qd.field(
                dtype=gs.qd_int, shape=(self._B,), needs_grad=False
            )
            self._development_implicit_fem_positive_j_schedule_min_j = qd.field(
                dtype=gs.qd_float,
                shape=(DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_SCHEDULE_LENGTH, self._B),
                needs_grad=False,
            )
        if self._enable_development_direct_replay_min_j_query:
            self._development_direct_replay_current_min_relative_j = qd.field(
                dtype=gs.qd_float, shape=(self._B,), needs_grad=False
            )

    def init_element_fields(self):
        # element state in vertices
        element_state_v = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
        )

        # element state in elements
        element_state_el = qd.types.struct(
            actu=gs.qd_float,  # actuation
        )

        # element state without gradient
        element_state_el_ng = qd.types.struct(
            active=gs.qd_bool,
        )

        # element info (properties that remain static through time)
        element_info = qd.types.struct(
            el2v=gs.qd_ivec4,  # vertex index of an element
            mu=gs.qd_float,  # lame parameters (1)
            lam=gs.qd_float,  # lame parameters (2)
            mass_scaled=gs.qd_float,  # scaled element mass. The real mass is mass_scaled / self._vol_scale
            mat_idx=gs.qd_int,  # material model index
            B=gs.qd_mat3,  # inverse of the deformation gradient at rest state
            V=gs.qd_float,  # rest volume of the element
            V_scaled=gs.qd_float,  # scaled rest volume of the element
            friction_mu=gs.qd_float,  # friction coefficient for contact
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )

        # element state for energy
        element_state_el_energy = qd.types.struct(
            energy=gs.qd_float,  # energy density for the element
            gradient=gs.qd_mat3,  # gradient density for the element, del energy / del F
        )

        element_state_v_energy = qd.types.struct(
            inertia=gs.qd_vec3,  # inertia for the vertex
            force=gs.qd_vec3,
        )

        element_v_info = qd.types.struct(
            mass=gs.qd_float,  # mass of the vertex
            mass_inv=gs.qd_float,  # inverse mass of the vertex
            mass_over_dt2=gs.qd_float,  # scaled mass of the vertex over dt^2
            friction_mu=gs.qd_float,  # friction coefficient for contact
        )

        pcg_state_v = qd.types.struct(
            diag3x3=gs.qd_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.qd_mat3,  # preconditioner
            x=gs.qd_vec3,  # solution vector
            r=gs.qd_vec3,  # residual vector
            z=gs.qd_vec3,  # preconditioned residual vector
            p=gs.qd_vec3,  # search direction vector
            Ap=gs.qd_vec3,  # matrix-vector product
        )

        linesearch_state_v = qd.types.struct(
            x_prev=gs.qd_vec3,  # solution vector
        )

        # construct field
        self.elements_v = element_state_v.field(
            shape=(self.sim.substeps_local + 1, self.n_vertices, self._B),
            needs_grad=True,
            layout=qd.Layout.SOA,
        )
        self.elements_el = element_state_el.field(
            shape=(self.sim.substeps_local + 1, self.n_elements, self._B),
            needs_grad=True,
            layout=qd.Layout.SOA,
        )
        self.elements_el_ng = element_state_el_ng.field(
            shape=(self.sim.substeps_local + 1, self.n_elements, self._B),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )
        self.elements_i = element_info.field(
            shape=(self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_el_energy = element_state_el_energy.field(
            shape=(self._B, self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_el_hessian = qd.field(shape=(self._B, 3, 3, self.n_elements), dtype=gs.qd_mat3)

        self.elements_v_energy = element_state_v_energy.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_v_info = element_v_info.field(
            shape=(self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.pcg_state_v = pcg_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.linesearch_state_v = linesearch_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

    def init_surface_fields(self):
        n_vertices_max = self.n_vertices
        n_surfaces_max = self.n_surfaces

        # surface info (for coupling)
        surface_state = qd.types.struct(
            tri2v=gs.qd_ivec3,  # vertex index of a triangle
            tri2el=gs.qd_int,  # element index of a triangle
            active=gs.qd_bool,
        )

        # for rendering (this is more of a surface)
        surface_state_render_v = qd.types.struct(
            vertices=gs.qd_vec3,
        )

        surface_state_render_f = qd.types.struct(
            indices=gs.qd_int,
        )

        # construct field
        self.surface = surface_state.field(
            shape=(n_surfaces_max),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.surface_render_v = surface_state_render_v.field(
            shape=(n_vertices_max, self._B),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )
        self.surface_render_f = surface_state_render_f.field(
            shape=(n_surfaces_max * 3),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        # UV coordinates for rendering (per-vertex UVs, initialized to zeros)
        self.surface_render_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(n_vertices_max, 1),), needs_grad=False)

    def init_rigid_mode_fields(self):
        """Allocate the fixed rigid basis and its enabled-only coarse fields."""
        self._rigid_mode_component_count = len(self._entities)

        component_by_vertex = np.empty(self.n_vertices, dtype=np.int32)
        basis = np.zeros((self.n_vertices, 6, 3), dtype=gs.np_float)
        rest_positions = np.zeros((self.n_vertices, 3), dtype=gs.np_float)
        for entity in self._entities:
            rest_positions[entity.v_start : entity.v_start + entity.n_vertices] = tensor_to_array(
                entity.init_positions, dtype=gs.np_float
            )

        masses = np.asarray(self.elements_v_info.mass.to_numpy(), dtype=gs.np_float)
        for component, entity in enumerate(self._entities):
            vertex_slice = slice(entity.v_start, entity.v_start + entity.n_vertices)
            component_by_vertex[vertex_slice] = component
            entity_positions = rest_positions[vertex_slice]
            entity_masses = masses[vertex_slice]
            total_mass = entity_masses.sum()
            center = np.sum(entity_positions * entity_masses[:, None], axis=0) / total_mass
            relative = entity_positions - center
            basis[vertex_slice, 0, 0] = 1.0
            basis[vertex_slice, 1, 1] = 1.0
            basis[vertex_slice, 2, 2] = 1.0
            basis[vertex_slice, 3] = np.cross(np.asarray((1.0, 0.0, 0.0)), relative)
            basis[vertex_slice, 4] = np.cross(np.asarray((0.0, 1.0, 0.0)), relative)
            basis[vertex_slice, 5] = np.cross(np.asarray((0.0, 0.0, 1.0)), relative)

        self.rigid_mode_component_by_vertex = qd.field(
            dtype=gs.qd_int, shape=(self.n_vertices,), needs_grad=False
        )
        self.rigid_mode_component_by_vertex.from_numpy(component_by_vertex)
        self.rigid_mode_basis = qd.field(dtype=gs.qd_vec3, shape=(self.n_vertices, 6), needs_grad=False)
        self.rigid_mode_basis.from_numpy(basis)

        rigid_mode_vec6 = qd.types.vector(6, gs.qd_float)
        rigid_mode_mat6 = qd.types.matrix(6, 6, gs.qd_float)
        self.rigid_mode_A_basis = qd.field(
            dtype=gs.qd_vec3, shape=(self._B, self.n_vertices, 6), needs_grad=False
        )
        self.rigid_mode_coarse_matrix = qd.field(
            dtype=rigid_mode_mat6, shape=(self._B, self._rigid_mode_component_count), needs_grad=False
        )
        self.rigid_mode_coarse_inverse = qd.field(
            dtype=rigid_mode_mat6, shape=(self._B, self._rigid_mode_component_count), needs_grad=False
        )
        self.rigid_mode_coarse_rhs = qd.field(
            dtype=rigid_mode_vec6, shape=(self._B, self._rigid_mode_component_count), needs_grad=False
        )
        self.rigid_mode_coarse_coeff = qd.field(
            dtype=rigid_mode_vec6, shape=(self._B, self._rigid_mode_component_count), needs_grad=False
        )

    def init_material_coarse_fields(self):
        """Allocate current-position motion bases for each per-entity material region."""
        mu = self.elements_i.mu.to_numpy()
        lam = self.elements_i.lam.to_numpy()
        tetrahedra = self.elements_i.el2v.to_numpy()

        supports = []
        self.material_coarse_groups = []
        for entity in self._entities:
            element_slice = slice(entity.el_start, entity.el_start + entity.n_elements)
            materials, inverse = np.unique(
                np.column_stack((mu[element_slice], lam[element_slice])), axis=0, return_inverse=True
            )
            entity_tetrahedra = tetrahedra[element_slice]
            for material_idx, (group_mu, group_lam) in enumerate(materials):
                vertices = np.unique(entity_tetrahedra[inverse == material_idx])
                support = np.zeros(self.n_vertices, dtype=np.bool_)
                support[vertices] = True
                supports.append(support)
                self.material_coarse_groups.append(
                    {
                        "fem_entity_idx": int(entity.idx),
                        "mu": float(group_mu),
                        "lambda": float(group_lam),
                        "vertex_count": int(len(vertices)),
                    }
                )

        self._material_coarse_group_count = len(supports)
        self.material_coarse_support = qd.field(
            dtype=gs.qd_bool, shape=(self._material_coarse_group_count, self.n_vertices), needs_grad=False
        )
        self.material_coarse_support.from_numpy(np.asarray(supports, dtype=np.bool_))
        self.material_coarse_vertex_count = qd.field(
            dtype=gs.qd_int, shape=(self._material_coarse_group_count,), needs_grad=False
        )
        self.material_coarse_vertex_count.from_numpy(
            np.asarray([support.sum() for support in supports], dtype=gs.np_int)
        )
        self.material_coarse_centroid = qd.field(
            dtype=gs.qd_vec3, shape=(self._B, self._material_coarse_group_count), needs_grad=False
        )
        self.material_coarse_basis = qd.field(
            dtype=gs.qd_vec3,
            shape=(self._B, self._material_coarse_group_count, self.n_vertices, 6),
            needs_grad=False,
        )
        self.material_coarse_norm_squared = qd.field(
            dtype=qd.types.vector(6, gs.qd_float),
            shape=(self._B, self._material_coarse_group_count),
            needs_grad=False,
        )
        material_vec6 = qd.types.vector(6, gs.qd_float)
        material_mat6 = qd.types.matrix(6, 6, gs.qd_float)
        self.material_coarse_matrix = qd.field(
            dtype=material_mat6, shape=(self._B, self._material_coarse_group_count), needs_grad=False
        )
        self.material_coarse_inverse = qd.field(
            dtype=material_mat6, shape=(self._B, self._material_coarse_group_count), needs_grad=False
        )
        self.material_coarse_rhs = qd.field(
            dtype=material_vec6, shape=(self._B, self._material_coarse_group_count), needs_grad=False
        )
        self.material_coarse_coeff = qd.field(
            dtype=material_vec6, shape=(self._B, self._material_coarse_group_count), needs_grad=False
        )

    def _init_surface_info(self):
        self.vertices_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_vertices,))
        self.elements_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_elements,))
        self.compute_surface_vertices()
        vertices_on_surface_np = self.vertices_on_surface.to_numpy()
        (surface_vertices_np,) = vertices_on_surface_np.nonzero()
        self.surface_vertices = qd.field(
            dtype=qd.i32,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vertices.from_numpy(surface_vertices_np.astype(np.int32, copy=False))

        # ``surface.tri2el`` is the explicit face-to-owner map populated by
        # FEMEntity.  The old implementation marked a tet as a surface tet
        # when *any* of its vertices happened to lie on the boundary.  That
        # admits interior-only elements and is not the same contact universe
        # as the explicit exterior triangles.  Keep the diagnostic boolean
        # kernel available for legacy analysis, but make the public surface
        # element list the unique, sorted mechanical owners of those faces.
        surface_triangle_owners = np.asarray(self.surface.tri2el.to_numpy(), dtype=np.int64)
        if surface_triangle_owners.shape != (self.n_surfaces,):
            raise RuntimeError("FEM surface owner table has an unexpected shape")
        surface_elements_np = np.unique(surface_triangle_owners).astype(np.int32, copy=False)
        if surface_elements_np.size and (
            np.any(surface_elements_np < 0) or np.any(surface_elements_np >= self.n_elements)
        ):
            raise RuntimeError("FEM surface owner table contains an invalid mechanical element")
        self.elements_on_surface.fill(False)
        self._mark_surface_elements_from_explicit_owners(surface_elements_np)
        self.surface_elements = qd.field(
            dtype=qd.i32,
            shape=(len(surface_elements_np),),
            needs_grad=False,
        )
        self.surface_elements.from_numpy(surface_elements_np.astype(np.int32, copy=False))

        surface_triangles_np = self.surface.tri2v.to_numpy()
        pos_np = self.elements_v.pos.to_numpy()[0, :, 0, :][surface_vertices_np]
        surface_vertices_mapping = np.full(self.n_vertices, -1, dtype=np.int32)
        surface_vertices_mapping[surface_vertices_np] = np.arange(len(surface_vertices_np))
        mass = igl.massmatrix(pos_np, surface_vertices_mapping[surface_triangles_np])
        surface_vert_mass_np = mass.diagonal().astype(gs.np_float, copy=False)
        self.surface_vert_mass = qd.field(
            dtype=gs.qd_float,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vert_mass.from_numpy(surface_vert_mass_np)

    @qd.kernel
    def compute_surface_vertices(self):
        for i_v in range(self.n_vertices):
            self.vertices_on_surface[i_v] = False

        for i_s in range(self.n_surfaces):
            tri2v = self.surface[i_s].tri2v
            for i in qd.static(range(3)):
                self.vertices_on_surface[tri2v[i]] = True

    @qd.kernel
    def compute_surface_elements(self):
        """Diagnostic legacy vertex-adjacent surface-element classification."""
        for i_e in range(self.n_elements):
            i_v = self.elements_i[i_e].el2v
            self.elements_on_surface[i_e] = (
                self.vertices_on_surface[i_v[0]]
                or self.vertices_on_surface[i_v[1]]
                or self.vertices_on_surface[i_v[2]]
                or self.vertices_on_surface[i_v[3]]
            )

    @qd.kernel
    def _mark_surface_elements_from_explicit_owners(self, owners: qd.types.ndarray()):
        """Mark the exact mechanical owner set for a host-built owner array."""
        for i in range(owners.shape[0]):
            self.elements_on_surface[owners[i]] = True

    def init_ckpt(self):
        self._ckpt = dict()

    def init_constraints(self):
        self._constraints_initialized = True

        vertex_constraint_info = qd.types.struct(
            is_constrained=gs.qd_bool,  # boolean flag indicating if vertex is constrained
            target_pos=gs.qd_vec3,  # target position for the constraint
            is_soft_constraint=gs.qd_bool,  # use spring for soft constraints
            stiffness=gs.qd_float,  # spring stiffness
            link_idx=gs.qd_int,  # index of the rigid link (-1 if not linked)
            link_offset_pos=gs.qd_vec3,  # offset position of link
            link_init_quat=gs.qd_vec4,  # offset rotation of link
        )

        # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
        self.vertex_constraints = vertex_constraint_info.field(
            shape=(self.n_vertices, self._B), needs_grad=False, layout=qd.Layout.AOS
        )

        self.vertex_constraints.is_constrained.fill(False)
        self.vertex_constraints.link_idx.fill(-1)

    def reset_grad(self):
        self.elements_v.grad.fill(0)
        self.elements_el.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self.tet_wrong_order = qd.field(dtype=gs.qd_bool, shape=(), needs_grad=False)

        # batch fields
        self.init_batch_fields()

        # rendering
        self.envs_offset = qd.Vector.field(3, dtype=qd.f32, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(np.float32))

        # elements and bodies
        self._n_elements_max = self.n_elements
        self._n_vertices_max = self.n_vertices
        if self.n_elements_max > 0:
            self.init_element_fields()
            self.init_surface_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            if self._use_implicit_solver and self._enable_rigid_mode_deflation:
                self.init_rigid_mode_fields()
            if self._use_implicit_solver and self._enable_material_coarse_preconditioner:
                self.init_material_coarse_fields()

        for mat in self._mats:
            mat.build(self)

        if self.n_elements_max > 0:
            self._init_surface_info()
            if self.tet_wrong_order[None]:
                raise RuntimeError(
                    "The order of vertices in the tetrahedral elements is not correct. "
                    "Please check the input mesh or the FEM solver implementation."
                )

        if self.n_vertices_max > 0 and self._enable_vertex_constraints and not self._constraints_initialized:
            self.init_constraints()

        if self.n_elements_max > 0 and self._use_implicit_solver and self._linear_solver == "sparse_direct":
            tetrahedra = np.asarray(self.elements_i.el2v.to_numpy(), dtype=np.int64)
            self._direct_tetrahedra = tetrahedra
            element_dofs = (tetrahedra[:, :, None] * 3 + np.arange(3, dtype=np.int64)).reshape(-1, 12)
            element_shape = (self.n_elements, 12, 12)
            self._direct_element_rows = np.broadcast_to(element_dofs[:, :, None], element_shape).ravel()
            self._direct_element_cols = np.broadcast_to(element_dofs[:, None, :], element_shape).ravel()
            self._direct_element_mapping = np.empty((self.n_elements, 4, 3), dtype=np.float64)
            self._direct_element_mapping[:, :3] = np.asarray(self.elements_i.B.to_numpy(), dtype=np.float64)
            self._direct_element_mapping[:, 3] = -self._direct_element_mapping[:, :3].sum(axis=1)
            self._direct_element_volume = np.asarray(self.elements_i.V.to_numpy(), dtype=np.float64)
            if self._sparse_direct_backend == "cudss" and self._cudss_gpu_assembly:
                matrix_size = self.n_vertices * 3
                diagonal = np.arange(matrix_size, dtype=np.int64)
                pattern_rows = np.concatenate((self._direct_element_rows, diagonal))
                pattern_cols = np.concatenate((self._direct_element_cols, diagonal))
                pattern = sp.coo_matrix(
                    (np.ones(pattern_rows.size), (pattern_rows, pattern_cols)),
                    shape=(matrix_size, matrix_size),
                ).tocsr()
                pattern.sum_duplicates()
                pattern.sort_indices()
                full_rows = np.repeat(
                    np.arange(matrix_size, dtype=np.int64), np.diff(pattern.indptr)
                )
                lower_source = np.flatnonzero(full_rows >= pattern.indices)
                lower_keys = matrix_size * full_rows[lower_source] + pattern.indices[lower_source]
                contribution_mask = self._direct_element_rows >= self._direct_element_cols
                contribution_keys = (
                    matrix_size * self._direct_element_rows[contribution_mask]
                    + self._direct_element_cols[contribution_mask]
                )
                contribution_indices = np.searchsorted(lower_keys, contribution_keys)
                diagonal_indices = np.searchsorted(lower_keys, matrix_size * diagonal + diagonal)
                pattern.data.fill(0.0)
                full_diagonal_positions = np.searchsorted(
                    matrix_size * full_rows + pattern.indices,
                    matrix_size * diagonal + diagonal,
                )
                pattern.data[full_diagonal_positions] = 1.0
                self._direct_gpu_template = pattern
                self._direct_gpu_lower_contribution_mask = contribution_mask
                self._direct_gpu_lower_contribution_indices = contribution_indices
                self._direct_gpu_diagonal_indices = diagonal_indices

        # FIXME: _gravity must be a raw qd.field() — see comment in mpm_solver.py
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    @property
    def is_active(self):
        return self.n_elements_max > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> "FEMEntity":
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material.idx = mat.idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material.idx = len(self._mats_idx)
            self._mats_idx.append(material.idx)
            self._mats_update_stress.append(material.update_stress)
            self._mats_compute_energy_gradient_hessian.append(material.compute_energy_gradient_hessian)
            self._mats_compute_energy.append(material.compute_energy)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            name=name,
        )

        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def init_pos_and_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            self.elements_v[f + 1, i_v, i_b].vel = self.elements_v[f, i_v, i_b].vel

    @qd.kernel
    def compute_vel(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
            pos_v0 = self.elements_v[f, i_v0, i_b].pos
            pos_v1 = self.elements_v[f, i_v1, i_b].pos
            pos_v2 = self.elements_v[f, i_v2, i_b].pos
            pos_v3 = self.elements_v[f, i_v3, i_b].pos
            D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

            V_scaled = self.elements_i[i_e].V_scaled
            B = self.elements_i[i_e].B
            F = D @ B
            J = F.determinant()

            stress = qd.Matrix.zero(gs.qd_float, 3, 3)
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    stress = self._mats_update_stress[mat_idx](
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                    )

            verts = self.elements_i[i_e].el2v
            mass_scaled = self.elements_i[i_e].mass_scaled
            H_scaled = -V_scaled * stress @ B.transpose()
            for k in qd.static(range(3)):
                force_scaled = qd.Vector([H_scaled[j, k] for j in range(3)])

                # store so forces can be read out
                self.elements_v_energy[i_b, verts[k]].force = force_scaled

                dv = self.substep_dt * force_scaled / mass_scaled
                self.elements_v[f + 1, verts[k], i_b].vel += dv
                self.elements_v[f + 1, verts[3], i_b].vel -= dv

    @qd.kernel
    def apply_uniform_force(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # NOTE: damping should only be applied to velocity from internal force and thus come first here
            #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
            #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
            #       however, this inevitably damp the gravity.
            self.elements_v[f + 1, i_v, i_b].vel *= qd.exp(-self.substep_dt * self.damping)
            # Add gravity (avoiding damping on gravity)
            self.elements_v[f + 1, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @qd.kernel
    def compute_pos(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.substep_dt * self.elements_v[f + 1, i_v, i_b].vel + self.elements_v[f, i_v, i_b].pos
            )

    @qd.kernel
    def precompute_material_data(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            J, F = self._compute_ele_J_F(f, i_e, i_b)  # use last time step's pos to compute
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self._mats[mat_idx].pre_compute(J=J, F=F, i_e=i_e, i_b=i_b)

    @qd.kernel
    def init_pos_and_inertia(self, f: qd.i32):
        dt2 = self.substep_dt**2
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            if qd.static(self._enable_vertex_constraints):
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and not vc.is_soft_constraint:
                    self.elements_v[f + 1, i_v, i_b].pos = vc.target_pos
                    self.elements_v_energy[i_b, i_v].inertia = vc.target_pos
                else:
                    self.elements_v_energy[i_b, i_v].inertia = (
                        self.elements_v[f, i_v, i_b].pos
                        + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                        + self._gravity[i_b] * dt2
                    )
                    self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            else:
                self.elements_v_energy[i_b, i_v].inertia = (
                    self.elements_v[f, i_v, i_b].pos
                    + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                    + self._gravity[i_b] * dt2
                )
                self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos

        # Element activity is state, not topology.  The implicit solver writes
        # the completed position/velocity frame here, so preserve the matching
        # per-element activity mask for public completed-frame readback too.
        # Without this propagation, a just-completed frame reports every
        # element inactive even though the next physical substep still solves
        # the same active volume.
        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            self.elements_el_ng[f + 1, i_e, i_b].active = self.elements_el_ng[f, i_e, i_b].active

    @qd.func
    def _compute_ele_J_F(self, f: qd.i32, i_e: qd.i32, i_b: qd.i32):
        """
        Compute the determinant (J) and deformation gradient (F) for an element.
        """
        i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
        pos_v0 = self.elements_v[f, i_v0, i_b].pos
        pos_v1 = self.elements_v[f, i_v1, i_b].pos
        pos_v2 = self.elements_v[f, i_v2, i_b].pos
        pos_v3 = self.elements_v[f, i_v3, i_b].pos
        D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

        B = self.elements_i[i_e].B
        F = D @ B
        J = F.determinant()

        return J, F

    @qd.kernel
    def _init_development_direct_replay_current_min_relative_j(self):
        for i_b in range(self._B):
            self._development_direct_replay_current_min_relative_j[i_b] = qd.math.inf

    @qd.kernel
    def _reduce_development_direct_replay_current_min_relative_j(
        self, f: qd.i32, element_start: qd.i32, element_count: qd.i32
    ):
        for i_b, i_e_local in qd.ndrange(self._B, element_count):
            i_e = element_start + i_e_local
            if self.elements_el_ng[f, i_e, i_b].active:
                J, _ = self._compute_ele_J_F(f, i_e, i_b)
                qd.atomic_min(
                    self._development_direct_replay_current_min_relative_j[i_b], J
                )

    def get_development_direct_replay_current_min_relative_j(
        self, entity: FEMEntity, *, env_index: int = 0
    ) -> float:
        """Return one scalar committed relative-J value for direct replay only."""
        if not self._enable_development_direct_replay_min_j_query:
            raise RuntimeError("development direct replay min-J query is not enabled")
        if entity._solver is not self:
            raise ValueError("development direct replay min-J entity belongs to another FEM solver")
        if type(env_index) is not int or not 0 <= env_index < self._B:
            raise ValueError("development direct replay min-J environment is invalid")
        self._init_development_direct_replay_current_min_relative_j()
        self._reduce_development_direct_replay_current_min_relative_j(
            int(self.sim.cur_substep_local), int(entity.el_start), int(entity.n_elements)
        )
        values = np.asarray(
            self._development_direct_replay_current_min_relative_j.to_numpy(),
            dtype=np.float64,
        )
        if values.shape != (self._B,) or not np.isfinite(values).all():
            raise RuntimeError("development direct replay committed relative-J is nonfinite")
        return float(values[env_index])

    @qd.kernel
    def compute_ele_hessian_gradient(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    if self._mats[mat_idx]._hessian_ready:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                        )
                    else:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient_hessian(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                            hessian_field=self.elements_el_hessian,
                        )

    @qd.func
    def _func_compute_element_mapping_matrix(self, i_vs, B, i_b):
        """
        Compute the element mapping matrix S for an element.
        """
        S = qd.Matrix.zero(gs.qd_float, 4, 3)
        S[:3, :] = B
        S[3, :] = -B[0, :] - B[1, :] - B[2, :]

        if qd.static(self._enable_vertex_constraints):
            for i in qd.static(range(4)):
                vc = self.vertex_constraints[i_vs[i], i_b]
                if vc.is_constrained and not vc.is_soft_constraint:
                    S[i, :] = qd.Vector.zero(gs.qd_float, 3)
        return S

    @qd.func
    def _func_compute_ele_energy(self, f: qd.i32):
        """
        Compute the energy for each element in the batch. Should only be used in linesearch.
        """
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self.elements_el_energy[i_b, i_e].energy = self._mats[mat_idx].compute_energy(
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                        i_e=i_e,
                        i_b=i_b,
                    )

            # add linearized damping energy
            if self._damping_beta > gs.EPS:
                damping_beta_over_dt = self._damping_beta / self._substep_dt
                i_vs = self.elements_i[i_e].el2v
                B = self.elements_i[i_e].B
                S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )

                self.elements_el_energy[i_b, i_e].energy += 0.5 * damping_beta_over_dt * St_x_diff.dot(H_St_x_diff)

    @qd.kernel
    def accumulate_vertex_force_preconditioner(self, f: qd.i32):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        # inertia
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v_energy[i_b, i_v].force = -self.elements_v_info[i_v].mass_over_dt2 * (
                (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia)
                + (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos) * damping_alpha_dt
            )
            self.pcg_state_v[i_b, i_v].diag3x3 = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                self.pcg_state_v[i_b, i_v].diag3x3[i, i] = (
                    self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor
                )

        # elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            gradient = self.elements_el_energy[i_b, i_e].gradient
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)
            force = -V * gradient @ S.transpose()

            # atomic
            for i in qd.static(range(4)):
                self.elements_v_energy[i_b, i_vs[i]].force += force[:, i]

            if self._damping_beta > gs.EPS:
                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )
                S_H_St_x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i, j in qd.static(qd.ndrange(4, 3)):
                    S_H_St_x_diff[i * 3 : i * 3 + 3] += S[i, j] * H_St_x_diff[j * 3 : j * 3 + 3]
                for i in qd.static(range(4)):
                    self.elements_v_energy[i_b, i_vs[i]].force += (
                        -damping_beta_over_dt * V * S_H_St_x_diff[i * 3 : i * 3 + 3]
                    )

            # diagonal 3-by-3 block of hessian
            for k, i, j in qd.ndrange(4, 3, 3):
                self.pcg_state_v[i_b, i_vs[k]].diag3x3 += (
                    V * damping_beta_factor * S[k, i] * S[k, j] * self.elements_el_hessian[i_b, i, j, i_e]
                )

        # implicit soft target penalty: E = 0.5 * k * ||x - target||^2
        if qd.static(self._enable_vertex_constraints):
            for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
                if not self.batch_active[i_b]:
                    continue
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and vc.is_soft_constraint:
                    pos_error = self.elements_v[f + 1, i_v, i_b].pos - vc.target_pos
                    self.elements_v_energy[i_b, i_v].force += -vc.stiffness * pos_error
                    for i in qd.static(range(3)):
                        self.pcg_state_v[i_b, i_v].diag3x3[i, i] += vc.stiffness

        # inverse
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            # Use 3-by-3 diagonal block inverse for preconditioner
            self.pcg_state_v[i_b, i_v].prec = self.pcg_state_v[i_b, i_v].diag3x3.inverse()

            # Other options for preconditioner:
            # Uncomment one of the following lines to test different preconditioners
            # Use identity for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix.identity(gs.qd_float, 3)

            # Use diagonal for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix([[1 / self.pcg_state_v[i_b, i_v].diag3x3[0, 0], 0, 0],
            #                                            [0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[1, 1], 0],
            #                                            [0, 0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[2, 2]]])

    @qd.func
    def _func_rigid_mode_basis(self, i_b: qd.i32, i_v: qd.i32, i_mode: qd.i32):
        basis = self.rigid_mode_basis[i_v, i_mode]
        if qd.static(self._enable_vertex_constraints):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                basis = qd.Vector.zero(gs.qd_float, 3)
        return basis

    @qd.kernel
    def _init_rigid_mode_A_basis_and_coarse_matrix(self):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        for i_b, i_v, i_mode in qd.ndrange(self._B, self.n_vertices, 6):
            if not self.batch_active[i_b]:
                continue
            basis = self._func_rigid_mode_basis(i_b, i_v, i_mode)
            self.rigid_mode_A_basis[i_b, i_v, i_mode] = (
                self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor * basis
            )
            if qd.static(self._enable_vertex_constraints):
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and vc.is_soft_constraint:
                    self.rigid_mode_A_basis[i_b, i_v, i_mode] += vc.stiffness * basis

        for i_b, i_c in qd.ndrange(self._B, self._rigid_mode_component_count):
            self.rigid_mode_coarse_matrix[i_b, i_c] = qd.Matrix.zero(gs.qd_float, 6, 6)

    @qd.kernel
    def _accumulate_rigid_mode_elastic_A_basis(self):
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

            for i_mode in qd.static(range(6)):
                p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    p9[i * 3 : i * 3 + 3] = (
                        p9[i * 3 : i * 3 + 3]
                        + S[j, i] * self._func_rigid_mode_basis(i_b, i_vs[j], i_mode)
                    )

                new_p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    new_p9[i * 3 : i * 3 + 3] = (
                        new_p9[i * 3 : i * 3 + 3]
                        + self.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
                    )

                for i in qd.static(range(4)):
                    contribution = (
                        S[i, 0] * new_p9[0:3]
                        + S[i, 1] * new_p9[3:6]
                        + S[i, 2] * new_p9[6:9]
                    ) * V * damping_beta_factor
                    for j in qd.static(range(3)):
                        qd.atomic_add(self.rigid_mode_A_basis[i_b, i_vs[i], i_mode][j], contribution[j])

    @qd.kernel
    def _reduce_rigid_mode_coarse_matrix(self):
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            i_c = self.rigid_mode_component_by_vertex[i_v]
            for i_row in qd.static(range(6)):
                basis = self._func_rigid_mode_basis(i_b, i_v, i_row)
                for i_col in qd.static(range(6)):
                    qd.atomic_add(
                        self.rigid_mode_coarse_matrix[i_b, i_c][i_row, i_col],
                        basis.dot(self.rigid_mode_A_basis[i_b, i_v, i_col]),
                    )

    @qd.kernel
    def _invert_rigid_mode_coarse_matrix(self):
        for i_b, i_c in qd.ndrange(self._B, self._rigid_mode_component_count):
            self.rigid_mode_coarse_inverse[i_b, i_c] = self.rigid_mode_coarse_matrix[i_b, i_c].inverse()

    def _prepare_rigid_mode_coarse_operator(self):
        self._init_rigid_mode_A_basis_and_coarse_matrix()
        self._accumulate_rigid_mode_elastic_A_basis()
        self._reduce_rigid_mode_coarse_matrix()
        self._invert_rigid_mode_coarse_matrix()

    @qd.func
    def _func_material_coarse_basis(self, i_b: qd.i32, i_g: qd.i32, i_v: qd.i32, i_mode: qd.i32):
        basis = self.material_coarse_basis[i_b, i_g, i_v, i_mode]
        if qd.static(self._enable_vertex_constraints):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                basis = qd.Vector.zero(gs.qd_float, 3)
        return basis

    @qd.kernel
    def _refresh_material_coarse_basis(self, f: qd.i32):
        self.material_coarse_centroid.fill(0.0)
        self.material_coarse_norm_squared.fill(0.0)
        for i_b, i_g, i_v in qd.ndrange(self._B, self._material_coarse_group_count, self.n_vertices):
            if self.material_coarse_support[i_g, i_v]:
                position = self.elements_v[f, i_v, i_b].pos
                for axis in qd.static(range(3)):
                    qd.atomic_add(
                        self.material_coarse_centroid[i_b, i_g][axis],
                        position[axis] / self.material_coarse_vertex_count[i_g],
                    )

        for i_b, i_g, i_v, i_mode in qd.ndrange(
            self._B, self._material_coarse_group_count, self.n_vertices, 6
        ):
            value = qd.Vector.zero(gs.qd_float, 3)
            if self.material_coarse_support[i_g, i_v]:
                axis = qd.Vector.zero(gs.qd_float, 3)
                if i_mode < 3:
                    axis[i_mode] = 1.0
                    value = axis
                else:
                    axis[i_mode - 3] = 1.0
                    value = axis.cross(
                        self.elements_v[f, i_v, i_b].pos - self.material_coarse_centroid[i_b, i_g]
                    )
                qd.atomic_add(self.material_coarse_norm_squared[i_b, i_g][i_mode], value.norm_sqr())
            self.material_coarse_basis[i_b, i_g, i_v, i_mode] = value

        for i_b, i_g, i_v, i_mode in qd.ndrange(
            self._B, self._material_coarse_group_count, self.n_vertices, 6
        ):
            if self.material_coarse_support[i_g, i_v]:
                self.material_coarse_basis[i_b, i_g, i_v, i_mode] /= qd.sqrt(
                    self.material_coarse_norm_squared[i_b, i_g][i_mode]
                )

    @qd.kernel
    def _reset_material_coarse_matrix(self):
        for i_b, i_g in qd.ndrange(self._B, self._material_coarse_group_count):
            self.material_coarse_matrix[i_b, i_g] = qd.Matrix.zero(gs.qd_float, 6, 6)

    @qd.kernel
    def _load_material_coarse_column(self, i_g: qd.i32, i_mode: qd.i32):
        for i_b in range(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            self.pcg_state_v[i_b, i_v].p = self._func_material_coarse_basis(i_b, i_g, i_v, i_mode)

    @qd.kernel
    def _material_coarse_product_and_reduce(self, i_g: qd.i32, i_col: qd.i32):
        self.compute_Ap(False)
        for i_b, i_v, i_row in qd.ndrange(self._B, self.n_vertices, 6):
            if self.batch_pcg_active[i_b]:
                qd.atomic_add(
                    self.material_coarse_matrix[i_b, i_g][i_row, i_col],
                    self._func_material_coarse_basis(i_b, i_g, i_v, i_row).dot(
                        self.pcg_state_v[i_b, i_v].Ap
                    ),
                )

    @qd.kernel
    def _invert_material_coarse_matrix(self):
        for i_b, i_g in qd.ndrange(self._B, self._material_coarse_group_count):
            if self.batch_active[i_b]:
                self.material_coarse_inverse[i_b, i_g] = self.material_coarse_matrix[i_b, i_g].inverse()

    def _prepare_material_coarse_operator(self):
        self._reset_material_coarse_matrix()
        for i_g in range(self._material_coarse_group_count):
            for i_mode in range(6):
                self._load_material_coarse_column(i_g, i_mode)
                self._material_coarse_product_and_reduce(i_g, i_mode)
        self._invert_material_coarse_matrix()

    @qd.func
    def compute_Ap(self, use_solution: qd.template()):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not (self.batch_active[i_b] if qd.static(use_solution) else self.batch_pcg_active[i_b]):
                continue
            vector = self.pcg_state_v[i_b, i_v].x if qd.static(use_solution) else self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].Ap = (
                self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor * vector
            )
            if qd.static(self._enable_vertex_constraints):
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and vc.is_soft_constraint:
                    self.pcg_state_v[i_b, i_v].Ap += vc.stiffness * vector

        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not (self.batch_active[i_b] if qd.static(use_solution) else self.batch_pcg_active[i_b]):
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

            p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 4)):
                vector = (
                    self.pcg_state_v[i_b, i_vs[j]].x
                    if qd.static(use_solution)
                    else self.pcg_state_v[i_b, i_vs[j]].p
                )
                p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + S[j, i] * vector

            new_p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    new_p9[i * 3 : i * 3 + 3] + self.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
                )

            # atomic
            for i in qd.static(range(4)):
                self.pcg_state_v[i_b, i_vs[i]].Ap += (
                    (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * V * damping_beta_factor
                )

    @qd.kernel
    def init_pcg_solve(self):
        for i_b in range(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTr_initial = 0.0
            self.pcg_state[i_b].termination_threshold = self._pcg_threshold
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = self.elements_v_energy[i_b, i_v].force
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z
            qd.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr_initial = self.pcg_state[i_b].rTr
            self.pcg_state[i_b].termination_threshold = qd.max(
                self._pcg_threshold,
                self.pcg_state[i_b].rTr_initial * self._pcg_rtol * self._pcg_rtol,
            )
            valid = (
                self.pcg_state[i_b].rTr >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTr)
                and not qd.math.isinf(self.pcg_state[i_b].rTr)
                and self.pcg_state[i_b].termination_threshold >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].termination_threshold)
                and not qd.math.isinf(self.pcg_state[i_b].termination_threshold)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self.pcg_state[i_b].termination_threshold

    @qd.kernel
    def one_pcg_iter(self):
        self.compute_Ap(False)

        # compute pTAp
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            qd.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

        # compute alpha and update x, r, z, rTr, rTz
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            valid = (
                self.pcg_state[i_b].rTz > 0.0
                and self.pcg_state[i_b].pTAp > 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTz)
                and not qd.math.isinf(self.pcg_state[i_b].rTz)
                and not qd.math.isnan(self.pcg_state[i_b].pTAp)
                and not qd.math.isinf(self.pcg_state[i_b].pTAp)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
                self.pcg_state[i_b].rTr_new = 0.0
                self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            qd.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        # Preserve the final residual even when this iteration converges. The
        # public health snapshot must never report the previous iterate.
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            valid = (
                self.pcg_state[i_b].rTr_new >= 0.0
                and self.pcg_state[i_b].rTz_new >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTr_new)
                and not qd.math.isinf(self.pcg_state[i_b].rTr_new)
                and not qd.math.isnan(self.pcg_state[i_b].rTz_new)
                and not qd.math.isinf(self.pcg_state[i_b].rTz_new)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
                self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
                self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new
            if qd.math.isnan(self.pcg_state[i_b].beta) or qd.math.isinf(self.pcg_state[i_b].beta):
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self.pcg_state[i_b].termination_threshold

        # update p
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    @qd.kernel
    def _init_pcg_solve_rigid_mode(self):
        for i_b in range(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTr_initial = 0.0
            self.pcg_state[i_b].termination_threshold = self._pcg_threshold
            self.pcg_state[i_b].rTz = 0.0
        if qd.static(self._enable_rigid_mode_deflation):
            for i_b, i_c in qd.ndrange(self._B, self._rigid_mode_component_count):
                self.rigid_mode_coarse_rhs[i_b, i_c] = qd.Vector.zero(gs.qd_float, 6)
        if qd.static(self._enable_material_coarse_preconditioner):
            for i_b, i_g in qd.ndrange(self._B, self._material_coarse_group_count):
                self.material_coarse_rhs[i_b, i_g] = qd.Vector.zero(gs.qd_float, 6)
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = self.elements_v_energy[i_b, i_v].force
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            if qd.static(self._enable_rigid_mode_deflation):
                i_c = self.rigid_mode_component_by_vertex[i_v]
                for i_mode in qd.static(range(6)):
                    qd.atomic_add(
                        self.rigid_mode_coarse_rhs[i_b, i_c][i_mode],
                        self._func_rigid_mode_basis(i_b, i_v, i_mode).dot(self.pcg_state_v[i_b, i_v].r),
                    )
            if qd.static(self._enable_material_coarse_preconditioner):
                for i_g, i_mode in qd.ndrange(self._material_coarse_group_count, 6):
                    qd.atomic_add(
                        self.material_coarse_rhs[i_b, i_g][i_mode],
                        self._func_material_coarse_basis(i_b, i_g, i_v, i_mode).dot(
                            self.pcg_state_v[i_b, i_v].r
                        ),
                    )

    @qd.kernel
    def _solve_rigid_mode_coarse_rhs(self):
        if qd.static(self._enable_rigid_mode_deflation):
            for i_b, i_c in qd.ndrange(self._B, self._rigid_mode_component_count):
                self.rigid_mode_coarse_coeff[i_b, i_c] = (
                    self.rigid_mode_coarse_inverse[i_b, i_c] @ self.rigid_mode_coarse_rhs[i_b, i_c]
                )
        if qd.static(self._enable_material_coarse_preconditioner):
            for i_b, i_g in qd.ndrange(self._B, self._material_coarse_group_count):
                self.material_coarse_coeff[i_b, i_g] = (
                    self.material_coarse_inverse[i_b, i_g] @ self.material_coarse_rhs[i_b, i_g]
                )

    @qd.kernel
    def _finish_init_pcg_solve_rigid_mode(self):
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            correction = qd.Vector.zero(gs.qd_float, 3)
            if qd.static(self._enable_rigid_mode_deflation):
                i_c = self.rigid_mode_component_by_vertex[i_v]
                for i_mode in qd.static(range(6)):
                    correction += self._func_rigid_mode_basis(i_b, i_v, i_mode) * self.rigid_mode_coarse_coeff[
                        i_b, i_c
                    ][i_mode]
            if qd.static(self._enable_material_coarse_preconditioner):
                for i_g, i_mode in qd.ndrange(self._material_coarse_group_count, 6):
                    correction += self._func_material_coarse_basis(
                        i_b, i_g, i_v, i_mode
                    ) * self.material_coarse_coeff[i_b, i_g][i_mode]
            self.pcg_state_v[i_b, i_v].z += correction
            qd.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr_initial = self.pcg_state[i_b].rTr
            self.pcg_state[i_b].termination_threshold = qd.max(
                self._pcg_threshold,
                self.pcg_state[i_b].rTr_initial * self._pcg_rtol * self._pcg_rtol,
            )
            valid = (
                self.pcg_state[i_b].rTr >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTr)
                and not qd.math.isinf(self.pcg_state[i_b].rTr)
                and self.pcg_state[i_b].termination_threshold >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].termination_threshold)
                and not qd.math.isinf(self.pcg_state[i_b].termination_threshold)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self.pcg_state[i_b].termination_threshold
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z

    @qd.kernel
    def _rigid_mode_compute_Ap_and_pTAp(self):
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_iterations[i_b] += 1
            self.pcg_state[i_b].pTAp = 0.0
        self.compute_Ap(False)
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            qd.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

    @qd.kernel
    def _rigid_mode_update_x_r_and_coarse_rhs(self):
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            valid = (
                self.pcg_state[i_b].rTz > 0.0
                and self.pcg_state[i_b].pTAp > 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTz)
                and not qd.math.isinf(self.pcg_state[i_b].rTz)
                and not qd.math.isnan(self.pcg_state[i_b].pTAp)
                and not qd.math.isinf(self.pcg_state[i_b].pTAp)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
        if qd.static(self._enable_rigid_mode_deflation):
            for i_b, i_c in qd.ndrange(self._B, self._rigid_mode_component_count):
                self.rigid_mode_coarse_rhs[i_b, i_c] = qd.Vector.zero(gs.qd_float, 6)
        if qd.static(self._enable_material_coarse_preconditioner):
            for i_b, i_g in qd.ndrange(self._B, self._material_coarse_group_count):
                self.material_coarse_rhs[i_b, i_g] = qd.Vector.zero(gs.qd_float, 6)
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            if qd.static(self._enable_rigid_mode_deflation):
                i_c = self.rigid_mode_component_by_vertex[i_v]
                for i_mode in qd.static(range(6)):
                    qd.atomic_add(
                        self.rigid_mode_coarse_rhs[i_b, i_c][i_mode],
                        self._func_rigid_mode_basis(i_b, i_v, i_mode).dot(self.pcg_state_v[i_b, i_v].r),
                    )
            if qd.static(self._enable_material_coarse_preconditioner):
                for i_g, i_mode in qd.ndrange(self._material_coarse_group_count, 6):
                    qd.atomic_add(
                        self.material_coarse_rhs[i_b, i_g][i_mode],
                        self._func_material_coarse_basis(i_b, i_g, i_v, i_mode).dot(
                            self.pcg_state_v[i_b, i_v].r
                        ),
                    )

    @qd.kernel
    def _finish_rigid_mode_pcg_iter_and_update_p(self):
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            correction = qd.Vector.zero(gs.qd_float, 3)
            if qd.static(self._enable_rigid_mode_deflation):
                i_c = self.rigid_mode_component_by_vertex[i_v]
                for i_mode in qd.static(range(6)):
                    correction += self._func_rigid_mode_basis(i_b, i_v, i_mode) * self.rigid_mode_coarse_coeff[
                        i_b, i_c
                    ][i_mode]
            if qd.static(self._enable_material_coarse_preconditioner):
                for i_g, i_mode in qd.ndrange(self._material_coarse_group_count, 6):
                    correction += self._func_material_coarse_basis(
                        i_b, i_g, i_v, i_mode
                    ) * self.material_coarse_coeff[i_b, i_g][i_mode]
            self.pcg_state_v[i_b, i_v].z += correction
            qd.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            valid = (
                self.pcg_state[i_b].rTr_new >= 0.0
                and self.pcg_state[i_b].rTz_new >= 0.0
                and not qd.math.isnan(self.pcg_state[i_b].rTr_new)
                and not qd.math.isinf(self.pcg_state[i_b].rTr_new)
                and not qd.math.isnan(self.pcg_state[i_b].rTz_new)
                and not qd.math.isinf(self.pcg_state[i_b].rTz_new)
            )
            if not valid:
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
                self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
                self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new
            if qd.math.isnan(self.pcg_state[i_b].beta) or qd.math.isinf(self.pcg_state[i_b].beta):
                self.batch_pcg_breakdown[i_b] = True
                self.batch_pcg_active[i_b] = False
            else:
                self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self.pcg_state[i_b].termination_threshold

        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    @qd.kernel
    def _count_active_pcg_iterations(self):
        for i_b in range(self._B):
            if self.batch_pcg_active[i_b]:
                self.batch_pcg_iterations[i_b] += 1

    @qd.kernel
    def _capture_true_residual_probe(self, sample_index: qd.i32):
        self.compute_Ap(True)
        for i_b in range(self._B):
            self.true_residual_probe_actual_iterations[sample_index, i_b] = self.batch_pcg_iterations[i_b]
            self.true_residual_probe_pcg_active[sample_index, i_b] = self.batch_pcg_active[i_b]
            self.true_residual_probe_true_rTr[sample_index, i_b] = 0.0
            self.true_residual_probe_recursive_rTr[sample_index, i_b] = self.pcg_state[i_b].rTr
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            residual = self.elements_v_energy[i_b, i_v].force - self.pcg_state_v[i_b, i_v].Ap
            qd.atomic_add(self.true_residual_probe_true_rTr[sample_index, i_b], residual.dot(residual))

    def _true_residual_probe_enabled_now(self):
        return (
            self._true_residual_probe_global_substep is not None
            and self.sim.cur_substep_global == self._true_residual_probe_global_substep
        )

    def get_true_residual_probe(self):
        """Return the bounded scalar receipt for the configured completed substep."""
        if not self._true_residual_probe_enabled_now():
            return None
        actual = np.asarray(self.true_residual_probe_actual_iterations.to_numpy(), dtype=np.int64)
        active = np.asarray(self.true_residual_probe_pcg_active.to_numpy(), dtype=np.bool_)
        true_rtr = np.asarray(self.true_residual_probe_true_rTr.to_numpy(), dtype=np.float64)
        recursive_rtr = np.asarray(self.true_residual_probe_recursive_rTr.to_numpy(), dtype=np.float64)
        samples = tuple(
            ImplicitFEMTrueResidualSample(
                completed_iteration=completed_iteration,
                actual_iterations_by_batch=tuple(int(value) for value in actual[sample_index]),
                pcg_active_by_batch=tuple(bool(value) for value in active[sample_index]),
                true_residual_squared_by_batch=tuple(float(value) for value in true_rtr[sample_index]),
                recursive_residual_squared_by_batch=tuple(float(value) for value in recursive_rtr[sample_index]),
            )
            for sample_index, completed_iteration in enumerate(TRUE_RESIDUAL_PROBE_SCHEDULE)
        )
        return ImplicitFEMTrueResidualProbe(
            global_substep_index=int(self._true_residual_probe_global_substep),
            samples=samples,
        )

    def pcg_solve(self):
        self.batch_pcg_iterations.fill(0)
        capture_probe = self._true_residual_probe_enabled_now()
        if self._enable_rigid_mode_deflation or self._enable_material_coarse_preconditioner:
            self._init_pcg_solve_rigid_mode()
            self._solve_rigid_mode_coarse_rhs()
            self._finish_init_pcg_solve_rigid_mode()
            if capture_probe:
                self._capture_true_residual_probe(0)
            for i in range(self._n_pcg_iterations):
                self._rigid_mode_compute_Ap_and_pTAp()
                self._rigid_mode_update_x_r_and_coarse_rhs()
                self._solve_rigid_mode_coarse_rhs()
                self._finish_rigid_mode_pcg_iter_and_update_p()
                if capture_probe and i + 1 in TRUE_RESIDUAL_PROBE_SCHEDULE:
                    self._capture_true_residual_probe(TRUE_RESIDUAL_PROBE_SCHEDULE.index(i + 1))
        else:
            self.init_pcg_solve()
            if capture_probe:
                self._capture_true_residual_probe(0)
            for i in range(self._n_pcg_iterations):
                self._count_active_pcg_iterations()
                self.one_pcg_iter()
                if capture_probe and i + 1 in TRUE_RESIDUAL_PROBE_SCHEDULE:
                    self._capture_true_residual_probe(TRUE_RESIDUAL_PROBE_SCHEDULE.index(i + 1))

    @qd.kernel
    def init_linesearch(self, f: qd.i32):
        for i_b in range(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy = 0.0
            self.linesearch_state[i_b].step_size = 1.0
            self.linesearch_state[i_b].m = 0.0

        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].prev_energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            if qd.static(self._enable_vertex_constraints):
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and vc.is_soft_constraint:
                    soft_diff = self.elements_v[f + 1, i_v, i_b].pos - vc.target_pos
                    self.linesearch_state[i_b].prev_energy += 0.5 * vc.stiffness * soft_diff.dot(soft_diff)
            self.linesearch_state_v[i_b, i_v].x_prev = self.elements_v[f + 1, i_v, i_b].pos
            self.linesearch_state[i_b].m -= self.pcg_state_v[i_b, i_v].x.dot(self.elements_v_energy[i_b, i_v].force)
        # Elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

    @qd.kernel
    def one_linesearch_iter(self, f: qd.i32):
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy = 0.0

        # update pos and compute Inertia energy
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.linesearch_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
            )
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            # damping
            if self._damping_alpha > 0.0:
                damping_alpha_dt = self._damping_alpha * self._substep_dt
                diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
                self.linesearch_state[i_b].energy += (
                    0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff) * damping_alpha_dt
                )
            if qd.static(self._enable_vertex_constraints):
                vc = self.vertex_constraints[i_v, i_b]
                if vc.is_constrained and vc.is_soft_constraint:
                    soft_diff = self.elements_v[f + 1, i_v, i_b].pos - vc.target_pos
                    self.linesearch_state[i_b].energy += 0.5 * vc.stiffness * soft_diff.dot(soft_diff)

        # compute elastic energy
        self._func_compute_ele_energy(f)
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

        # check condition
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = (
                self.linesearch_state[i_b].energy
                > self.linesearch_state[i_b].prev_energy
                + self._linesearch_c * self.linesearch_state[i_b].step_size * self.linesearch_state[i_b].m
            )
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size *= self._linesearch_tau

    @qd.kernel
    def skip_linesearch(self, f: qd.i32):
        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f + 1, i_v, i_b].pos + self.pcg_state_v[i_b, i_v].x

    def linesearch(self, f: qd.i32):
        """
        Note
        ------
        https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
        """
        if self._n_linesearch_iterations <= 0:
            self.skip_linesearch(f)
            return
        self.init_linesearch(f)
        for i in range(self._n_linesearch_iterations):
            self.one_linesearch_iter(f)

    @qd.func
    def _development_implicit_fem_positive_j_at_alpha(self, f: qd.i32, i_b: qd.i32, i_e: qd.i32, alpha):
        i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
        pos_v0 = self.elements_v[f + 1, i_v0, i_b].pos + alpha * self.pcg_state_v[i_b, i_v0].x
        pos_v1 = self.elements_v[f + 1, i_v1, i_b].pos + alpha * self.pcg_state_v[i_b, i_v1].x
        pos_v2 = self.elements_v[f + 1, i_v2, i_b].pos + alpha * self.pcg_state_v[i_b, i_v2].x
        pos_v3 = self.elements_v[f + 1, i_v3, i_b].pos + alpha * self.pcg_state_v[i_b, i_v3].x
        F = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3]) @ self.elements_i[i_e].B
        return F.determinant()

    @qd.kernel
    def _init_development_implicit_fem_positive_j_reduction(self):
        for i_b in range(self._B):
            self._development_implicit_fem_positive_j_base_min_j[i_b] = qd.math.inf
            self._development_implicit_fem_positive_j_trial_min_j[i_b] = qd.math.inf
            self._development_implicit_fem_positive_j_accepted_alpha[i_b] = 0.0
            self._development_implicit_fem_positive_j_witness_tet_id[i_b] = -1
            for i_schedule in qd.static(range(DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_SCHEDULE_LENGTH)):
                self._development_implicit_fem_positive_j_schedule_min_j[i_schedule, i_b] = qd.math.inf

    @qd.kernel
    def _reduce_development_implicit_fem_positive_j_endpoints(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.elements_el_ng[f + 1, i_e, i_b].active:
                continue
            qd.atomic_min(
                self._development_implicit_fem_positive_j_base_min_j[i_b],
                self._development_implicit_fem_positive_j_at_alpha(f, i_b, i_e, 0.0),
            )
            qd.atomic_min(
                self._development_implicit_fem_positive_j_trial_min_j[i_b],
                self._development_implicit_fem_positive_j_at_alpha(f, i_b, i_e, 1.0),
            )
            for i_schedule in qd.static(range(DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_SCHEDULE_LENGTH)):
                alpha = qd.static(1.0 / (2**i_schedule))
                qd.atomic_min(
                    self._development_implicit_fem_positive_j_schedule_min_j[i_schedule, i_b],
                    self._development_implicit_fem_positive_j_at_alpha(f, i_b, i_e, alpha),
                )

    @qd.kernel
    def _select_development_implicit_fem_positive_j_alpha_and_witness(self, f: qd.i32):
        for i_b in range(self._B):
            base_min = self._development_implicit_fem_positive_j_base_min_j[i_b]
            if base_min == qd.math.inf:
                self._development_implicit_fem_positive_j_base_min_j[i_b] = 1.0
                self._development_implicit_fem_positive_j_trial_min_j[i_b] = 1.0
                self._development_implicit_fem_positive_j_accepted_alpha[i_b] = 1.0
                self._development_implicit_fem_positive_j_witness_tet_id[i_b] = -1
                continue

            base_infeasible = base_min < DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_FLOOR
            alpha = 0.0
            if not base_infeasible:
                for i_schedule in qd.static(range(DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_SCHEDULE_LENGTH)):
                    scheduled_alpha = qd.static(1.0 / (2**i_schedule))
                    if (
                        alpha == 0.0
                        and self._development_implicit_fem_positive_j_schedule_min_j[i_schedule, i_b]
                        >= DEVELOPMENT_IMPLICIT_FEM_POSITIVE_J_FLOOR
                    ):
                        alpha = scheduled_alpha

            witness_alpha = 0.0 if base_infeasible else 1.0
            witness_j = qd.math.inf
            witness_tet = -1
            for i_e in range(self.n_elements):
                if not self.elements_el_ng[f + 1, i_e, i_b].active:
                    continue
                candidate_j = self._development_implicit_fem_positive_j_at_alpha(f, i_b, i_e, witness_alpha)
                if candidate_j < witness_j:
                    witness_j = candidate_j
                    witness_tet = i_e
            self._development_implicit_fem_positive_j_accepted_alpha[i_b] = alpha
            self._development_implicit_fem_positive_j_witness_tet_id[i_b] = witness_tet

    @qd.kernel
    def _commit_development_implicit_fem_positive_j_position(self, f: qd.i32):
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            alpha = self._development_implicit_fem_positive_j_accepted_alpha[i_b]
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.elements_v[f + 1, i_v, i_b].pos + alpha * self.pcg_state_v[i_b, i_v].x
            )

    def _apply_development_implicit_fem_positive_j_feasible_step(self, f: qd.i32):
        """Commit the largest fixed endpoint-feasible PCG position update per environment."""
        self._init_development_implicit_fem_positive_j_reduction()
        self._reduce_development_implicit_fem_positive_j_endpoints(f)
        self._select_development_implicit_fem_positive_j_alpha_and_witness(f)
        self._commit_development_implicit_fem_positive_j_position(f)

    def _sparse_direct_solve(self, f: qd.i32):
        """Solve the current free-FEM Newton system and retain its velocity form for SAP."""
        if self._sparse_direct_backend == "cudss" and self._cudss_gpu_assembly:
            return self._sparse_direct_solve_cudss_gpu(f)

        transfer_start = time.perf_counter()
        hessians = np.asarray(self.elements_el_hessian.to_numpy(), dtype=np.float64)
        force = np.asarray(self.elements_v_energy.force.to_numpy(), dtype=np.float64)
        mass_over_dt2 = np.asarray(self.elements_v_info.mass_over_dt2.to_numpy(), dtype=np.float64)
        positions = np.asarray(self.elements_v.pos.to_numpy(), dtype=np.float64)
        if self._enable_vertex_constraints:
            is_constrained = np.asarray(self.vertex_constraints.is_constrained.to_numpy(), dtype=np.bool_).T
            is_soft = np.asarray(self.vertex_constraints.is_soft_constraint.to_numpy(), dtype=np.bool_).T
            constraint_stiffness = np.asarray(self.vertex_constraints.stiffness.to_numpy(), dtype=np.float64).T
        else:
            is_constrained = np.zeros((self._B, self.n_vertices), dtype=np.bool_)
            is_soft = np.zeros_like(is_constrained)
            constraint_stiffness = np.zeros((self._B, self.n_vertices), dtype=np.float64)
        self._direct_transfer_time_s += time.perf_counter() - transfer_start

        h = float(self.substep_dt)
        damping_alpha_factor = 1.0 + self._damping_alpha * h
        damping_beta_factor = 1.0 + self._damping_beta / h
        matrix_size = self.n_vertices * 3
        diagonal_dofs = np.arange(matrix_size, dtype=np.int64)
        matrices = []
        factors = []
        velocity_rhs = np.empty((self._B, matrix_size), dtype=np.float64)
        velocity_updates = np.empty_like(velocity_rhs)

        assembly_start = time.perf_counter()
        for i_b in range(self._B):
            mapping = self._direct_element_mapping.copy()
            hard_constrained = is_constrained[i_b] & ~is_soft[i_b]
            mapping[hard_constrained[self._direct_tetrahedra]] = 0.0
            element_hessian = np.moveaxis(hessians[i_b], 2, 0)
            element_matrix = np.einsum(
                "eki,eijab,elj->ekalb",
                mapping,
                element_hessian,
                mapping,
                optimize=True,
            )
            element_matrix *= self._direct_element_volume[:, None, None, None, None] * damping_beta_factor
            position_matrix = sp.coo_matrix(
                (element_matrix.reshape(-1), (self._direct_element_rows, self._direct_element_cols)),
                shape=(matrix_size, matrix_size),
            ).tocsc()
            vertex_diagonal = mass_over_dt2 * damping_alpha_factor
            vertex_diagonal += np.where(
                is_constrained[i_b] & is_soft[i_b], constraint_stiffness[i_b], 0.0
            )
            position_matrix += sp.csc_matrix(
                (np.repeat(vertex_diagonal, 3), (diagonal_dofs, diagonal_dofs)),
                shape=(matrix_size, matrix_size),
            )
            velocity_matrix = position_matrix * (h * h)
            velocity_matrix.sum_duplicates()
            velocity_matrix.sort_indices()

            base_velocity = (positions[f + 1, :, i_b] - positions[f, :, i_b]).reshape(-1) / h
            velocity_rhs[i_b] = velocity_matrix @ base_velocity + h * force[i_b].reshape(-1)
            matrices.append(velocity_matrix)
        self._direct_assembly_time_s += time.perf_counter() - assembly_start

        factor_start = time.perf_counter()
        if self._sparse_direct_backend == "cudss":
            from genesis.utils.cudss_sparse import CudssSpdFactor

            if not self._direct_velocity_cudss_factors:
                self._direct_velocity_cudss_factors = [
                    CudssSpdFactor(velocity_matrix) for velocity_matrix in matrices
                ]
                self._direct_velocity_cudss_ages = [0 for _ in matrices]
                self._direct_cudss_factorizations += len(matrices)
            else:
                for i_b, (factor, velocity_matrix) in enumerate(
                    zip(self._direct_velocity_cudss_factors, matrices, strict=True)
                ):
                    next_age = self._direct_velocity_cudss_ages[i_b] + 1
                    refresh = self._cudss_anchor_age <= 0 or next_age >= self._cudss_anchor_age
                    if refresh:
                        if factor.matches_pattern(velocity_matrix):
                            factor.refactor(velocity_matrix)
                        else:
                            factor.close()
                            self._direct_velocity_cudss_factors[i_b] = CudssSpdFactor(
                                velocity_matrix
                            )
                        self._direct_velocity_cudss_ages[i_b] = 0
                        self._direct_cudss_factorizations += 1
                    else:
                        self._direct_velocity_cudss_ages[i_b] = next_age
            factors.extend(self._direct_velocity_cudss_factors)
        else:
            for velocity_matrix in matrices:
                factors.append(spla.splu(velocity_matrix))
        self._direct_factor_time_s += time.perf_counter() - factor_start

        solve_start = time.perf_counter()
        for i_b, factor in enumerate(factors):
            force_rhs = h * force[i_b].reshape(-1)
            if self._sparse_direct_backend == "cudss" and self._direct_velocity_cudss_ages[i_b] > 0:
                update, iterations, relative_residual, converged = factor.solve_pcg(
                    matrices[i_b],
                    force_rhs,
                    max_iterations=self._cudss_pcg_max_iterations,
                    relative_tolerance=self._cudss_pcg_rtol,
                )
                self._direct_cudss_pcg_solves += 1
                self._direct_cudss_pcg_iterations += iterations
                self._direct_cudss_last_relative_residual = relative_residual
                self._direct_cudss_max_relative_residual = max(
                    self._direct_cudss_max_relative_residual, relative_residual
                )
                if not converged:
                    fallback_start = time.perf_counter()
                    if factor.matches_pattern(matrices[i_b]):
                        factor.refactor(matrices[i_b])
                    else:
                        factor.close()
                        factor = CudssSpdFactor(matrices[i_b])
                        self._direct_velocity_cudss_factors[i_b] = factor
                        factors[i_b] = factor
                    self._direct_factor_time_s += time.perf_counter() - fallback_start
                    self._direct_velocity_cudss_ages[i_b] = 0
                    self._direct_cudss_factorizations += 1
                    self._direct_cudss_pcg_fallbacks += 1
                    update = factor.solve(force_rhs)
                velocity_updates[i_b] = update
            else:
                velocity_updates[i_b] = factor.solve(force_rhs)
        self._direct_solve_time_s += time.perf_counter() - solve_start

        if (
            self._sparse_direct_backend == "cudss"
            and os.environ.get("GENESIS_FEM_CUDSS_TELEMETRY") == "1"
            and (int(self.sim.cur_substep_global) + 1) % 20 == 0
        ):
            print(
                "[cuDSS FEM] "
                f"substeps={int(self.sim.cur_substep_global) + 1} "
                f"factors={self._direct_cudss_factorizations} "
                f"pcg_solves={self._direct_cudss_pcg_solves} "
                f"pcg_iterations={self._direct_cudss_pcg_iterations} "
                f"fallbacks={self._direct_cudss_pcg_fallbacks} "
                f"last_relative_residual={self._direct_cudss_last_relative_residual:.3e} "
                f"max_relative_residual={self._direct_cudss_max_relative_residual:.3e} "
                f"assembly_s={self._direct_assembly_time_s:.6f} "
                f"factor_s={self._direct_factor_time_s:.6f} "
                f"solve_s={self._direct_solve_time_s:.6f} "
                f"transfer_s={self._direct_transfer_time_s:.6f}",
                flush=True,
            )

        transfer_start = time.perf_counter()
        self.pcg_state_v.x.from_numpy((h * velocity_updates).reshape(self._B, self.n_vertices, 3))
        self._direct_transfer_time_s += time.perf_counter() - transfer_start

        position_residual = force.reshape(self._B, -1) - np.stack(
            [(matrix @ velocity_updates[i_b]) / h for i_b, matrix in enumerate(matrices)]
        )
        initial_rtr = np.einsum("bi,bi->b", force.reshape(self._B, -1), force.reshape(self._B, -1))
        final_rtr = np.einsum("bi,bi->b", position_residual, position_residual)
        self.pcg_state.rTr_initial.from_numpy(initial_rtr)
        self.pcg_state.rTr.from_numpy(final_rtr)
        self.pcg_state.rTz.from_numpy(final_rtr)
        self.pcg_state.termination_threshold.from_numpy(
            np.maximum(self._pcg_threshold, initial_rtr * self._pcg_rtol * self._pcg_rtol)
        )
        self.batch_pcg_active.fill(False)
        self.batch_pcg_iterations.fill(0)

        self._direct_velocity_matrices = tuple(matrices)
        self._direct_velocity_matrix = matrices[0] if self._B == 1 else None
        self._direct_velocity_factors = tuple(factors)
        self._direct_velocity_gpu_factors.clear()
        self._direct_velocity_rhs = velocity_rhs

    def _sparse_direct_solve_cudss_gpu(self, f: qd.i32):
        """Assemble the exact fixed-graph FEM operator on CUDA and factor it with cuDSS."""
        from genesis.utils.cudss_sparse import CudssSpdFactor, CudssSpdMatrix

        hessian = qd_to_torch(self.elements_el_hessian, copy=False)
        force = qd_to_torch(self.elements_v_energy.force, copy=False)
        mass_over_dt2 = qd_to_torch(self.elements_v_info.mass_over_dt2, copy=False)
        positions = qd_to_torch(self.elements_v.pos, copy=False)
        device = hessian.device
        if self._direct_gpu_static_tensors is None:
            self._direct_gpu_static_tensors = {
                "mapping": torch.as_tensor(
                    self._direct_element_mapping, dtype=torch.float64, device=device
                ),
                "tetrahedra": torch.as_tensor(
                    self._direct_tetrahedra, dtype=torch.long, device=device
                ),
                "volume": torch.as_tensor(
                    self._direct_element_volume, dtype=torch.float64, device=device
                ),
                "contribution_mask": torch.as_tensor(
                    self._direct_gpu_lower_contribution_mask, dtype=torch.bool, device=device
                ),
                "contribution_indices": torch.as_tensor(
                    self._direct_gpu_lower_contribution_indices, dtype=torch.long, device=device
                ),
                "diagonal_indices": torch.as_tensor(
                    self._direct_gpu_diagonal_indices, dtype=torch.long, device=device
                ),
            }
        static = self._direct_gpu_static_tensors
        if self._enable_vertex_constraints:
            is_constrained = qd_to_torch(
                self.vertex_constraints.is_constrained, copy=True
            )
            is_soft = qd_to_torch(
                self.vertex_constraints.is_soft_constraint, copy=True
            )
            constraint_stiffness = qd_to_torch(
                self.vertex_constraints.stiffness, copy=True
            )

        h = float(self.substep_dt)
        damping_alpha_factor = 1.0 + self._damping_alpha * h
        damping_beta_factor = 1.0 + self._damping_beta / h
        matrix_size = self.n_vertices * 3
        factors = []
        matrices = []
        velocity_rhs_device = []
        velocity_updates_device = []

        assembly_start = time.perf_counter()
        for i_b in range(self._B):
            mapping = static["mapping"]
            if self._enable_vertex_constraints:
                hard_constrained = is_constrained[:, i_b] & ~is_soft[:, i_b]
                mapping = mapping * (~hard_constrained[static["tetrahedra"]])[:, :, None]
            element_hessian = hessian[i_b].movedim(2, 0)
            element_matrix = torch.einsum(
                "eki,eijab,elj->ekalb",
                mapping,
                element_hessian,
                mapping,
            )
            element_matrix *= (
                static["volume"] * damping_beta_factor
            )[:, None, None, None, None]

            if not self._direct_velocity_cudss_factors:
                factor = CudssSpdFactor(self._direct_gpu_template)
                self._direct_velocity_cudss_factors.append(factor)
                self._direct_velocity_cudss_ages.append(0)
            else:
                factor = self._direct_velocity_cudss_factors[i_b]
            lower_values = torch.zeros(
                factor._values.size, dtype=torch.float64, device=device
            )
            lower_values.index_add_(
                0,
                static["contribution_indices"],
                element_matrix.reshape(-1)[static["contribution_mask"]],
            )
            vertex_diagonal = mass_over_dt2 * damping_alpha_factor
            if self._enable_vertex_constraints:
                vertex_diagonal = vertex_diagonal + torch.where(
                    is_constrained[:, i_b] & is_soft[:, i_b],
                    constraint_stiffness[:, i_b],
                    0.0,
                )
            lower_values[static["diagonal_indices"]] += vertex_diagonal.repeat_interleave(3)
            lower_values *= h * h
            factor.refactor_lower_values(lower_values)

            base_velocity = (
                positions[f + 1, :, i_b] - positions[f, :, i_b]
            ).reshape(-1) / h
            force_rhs = h * force[i_b].reshape(-1)
            velocity_rhs_device.append(factor.matvec_gpu(base_velocity) + force_rhs)
            velocity_updates_device.append(factor.solve_gpu(force_rhs[:, None])[:, 0])
            factors.append(factor)
            matrices.append(CudssSpdMatrix(factor))
        torch.cuda.synchronize(device)
        self._direct_assembly_time_s += time.perf_counter() - assembly_start
        self._direct_cudss_factorizations += len(factors)

        transfer_start = time.perf_counter()
        pcg_x = qd_to_torch(self.pcg_state_v.x, copy=False)
        for i_b, update in enumerate(velocity_updates_device):
            pcg_x[i_b].copy_((h * update).reshape(self.n_vertices, 3))
        torch.cuda.synchronize(device)
        self._direct_transfer_time_s += time.perf_counter() - transfer_start

        force_flat = force.reshape(self._B, -1)
        residuals = torch.stack(
            [
                force_flat[i_b] - factors[i_b].matvec_gpu(velocity_updates_device[i_b]) / h
                for i_b in range(self._B)
            ]
        )
        initial_rtr = torch.sum(force_flat * force_flat, dim=1).cpu().numpy()
        final_rtr = torch.sum(residuals * residuals, dim=1).cpu().numpy()
        self.pcg_state.rTr_initial.from_numpy(initial_rtr)
        self.pcg_state.rTr.from_numpy(final_rtr)
        self.pcg_state.rTz.from_numpy(final_rtr)
        self.pcg_state.termination_threshold.from_numpy(
            np.maximum(self._pcg_threshold, initial_rtr * self._pcg_rtol * self._pcg_rtol)
        )
        self.batch_pcg_active.fill(False)
        self.batch_pcg_iterations.fill(0)

        self._direct_velocity_matrices = tuple(matrices)
        self._direct_velocity_matrix = matrices[0] if self._B == 1 else None
        self._direct_velocity_factors = tuple(factors)
        self._direct_velocity_gpu_factors.clear()
        self._direct_velocity_rhs = np.stack(
            [rhs.cpu().numpy() for rhs in velocity_rhs_device]
        )

    def _finalize_sparse_direct_velocity_system(self, f: qd.i32):
        transfer_start = time.perf_counter()
        positions = np.asarray(self.elements_v.pos.to_numpy(), dtype=np.float64)
        self._direct_transfer_time_s += time.perf_counter() - transfer_start
        h = float(self.substep_dt)
        free_velocity = np.transpose(positions[f + 1] - positions[f], (1, 0, 2)).reshape(self._B, -1) / h
        free_residual = np.stack(
            [
                self._direct_velocity_rhs[i_b] - self._direct_velocity_matrices[i_b] @ free_velocity[i_b]
                for i_b in range(self._B)
            ]
        )
        self._direct_free_velocity = free_velocity
        self._direct_free_residual = free_residual
        self._direct_free_residual_norm = np.linalg.norm(free_residual, axis=1)
        self._direct_free_positions = np.transpose(positions[f + 1], (1, 0, 2)).copy()
        self._direct_geometry_candidates = [{} for _ in range(self._B)]
        position_residual_rtr = np.einsum("bi,bi->b", free_residual, free_residual) / (h * h)
        self.pcg_state.rTr.from_numpy(position_residual_rtr)
        self.pcg_state.rTz.from_numpy(position_residual_rtr)

    def get_sparse_direct_velocity_system(self, i_b: int):
        """Return ``(A_f, b_f, c, b_f - A_f c)`` for one completed free solve."""
        if self._linear_solver != "sparse_direct" or self._direct_velocity_rhs is None:
            raise RuntimeError("sparse-direct FEM velocity system is not available")
        return (
            self._direct_velocity_matrices[i_b],
            self._direct_velocity_rhs[i_b],
            self._direct_free_velocity[i_b],
            self._direct_free_residual[i_b],
        )

    def solve_sparse_direct_velocity_rhs(self, i_b: int, rhs):
        """Apply the current free-FEM ``A_f`` LU factor to one or many FP64 right-hand sides."""
        return np.asarray(self._direct_velocity_factors[i_b].solve(np.asarray(rhs, dtype=np.float64)), dtype=np.float64)

    def solve_sparse_direct_velocity_rhs_gpu(self, i_b: int, rhs) -> torch.Tensor:
        """Solve current ``A_f`` against CUDA FP64 ``(3 * n_vertices, K)`` RHSs.

        ``rhs`` may be a Torch tensor or a CuPy array. The returned Torch tensor
        shares the CUDA solution through DLPack; the RHS and solution never
        pass through host memory. Work runs on the current Torch stream for
        the RHS device. This interface is for substep response preparation,
        and does not provide autograd through the sparse solve.

        CuPy uploads the existing SciPy SuperLU factors lazily, including both
        row and column permutations. These GPU factors are reused within the
        same physical substep, but CuPy's solve still performs its own sparse
        triangular analysis on each call.
        """
        if self._linear_solver != "sparse_direct" or not self._direct_velocity_factors:
            raise RuntimeError("GPU sparse-direct FEM solve requires a completed sparse_direct free-FEM solve")
        if not 0 <= i_b < len(self._direct_velocity_factors):
            raise IndexError(f"FEM batch index {i_b} is outside the current sparse-direct factors")

        try:
            import cupy as cp
            from cupyx.scipy.sparse.linalg import SuperLU
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "The SAP GPU Schur backend requires CuPy with CUDA support for the FEM response solve. "
                "Install a CuPy package matching the CUDA runtime (for CUDA 12, cupy-cuda12x) "
                "in the active Python environment."
            ) from exc

        if isinstance(rhs, torch.Tensor):
            if rhs.device.type != "cuda" or rhs.dtype != torch.float64:
                raise ValueError("GPU sparse-direct FEM RHS must be a CUDA torch.float64 tensor")
            device_index = rhs.device.index
            rhs_torch = rhs.detach()
        elif isinstance(rhs, cp.ndarray):
            if rhs.dtype != cp.float64:
                raise ValueError("GPU sparse-direct FEM RHS must be a CuPy float64 array")
            device_index = rhs.device.id
            # Import before overriding CuPy's stream so DLPack can order its
            # producing stream before the current Torch consumer stream.
            with torch.cuda.device(device_index):
                rhs_torch = torch.from_dlpack(rhs)
        else:
            raise TypeError("GPU sparse-direct FEM RHS must be a CUDA Torch tensor or CuPy array")
        if rhs_torch.ndim != 2 or rhs_torch.shape[0] != 3 * self.n_vertices:
            raise ValueError(f"GPU sparse-direct FEM RHS must have shape ({3 * self.n_vertices}, K)")

        factor = self._direct_velocity_factors[i_b]
        if self._sparse_direct_backend == "cudss":
            return factor.solve_gpu(rhs_torch)

        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            if rhs_torch.shape[1] == 0:
                return torch.empty_like(rhs_torch)
            with cp.cuda.Device(device_index), cp.cuda.ExternalStream(stream.cuda_stream, device_id=device_index):
                rhs_device = cp.from_dlpack(rhs_torch)
                key = (i_b, device_index)
                cached = self._direct_velocity_gpu_factors.get(key)
                if cached is None or cached[0] is not factor:
                    # SuperLU's wrapper applies perm_r/perm_c around the two
                    # GPU triangular solves; the unpermuted factors alone do
                    # not represent A_f^{-1}.
                    gpu_factor = SuperLU(factor)
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    self._direct_velocity_gpu_factors[key] = (factor, gpu_factor, ready)
                else:
                    _, gpu_factor, ready = cached
                    stream.wait_event(ready)
                solution = gpu_factor.solve(rhs_device)
                return torch.from_dlpack(solution)

    def get_material_connected_partition_of_unity(
        self,
        *,
        max_partitions_per_component: int = 8,
        target_tets_per_partition: int = 128,
        smoothing_steps: int = 3,
    ):
        """Return the fixed generic material-connected PoU for this FEM mesh."""
        cache_key = (max_partitions_per_component, target_tets_per_partition, smoothing_steps)
        if cache_key not in self._material_connected_pou_cache:
            material_keys = np.column_stack(
                (
                    np.asarray(self.elements_i.mu.to_numpy(), dtype=np.float64),
                    np.asarray(self.elements_i.lam.to_numpy(), dtype=np.float64),
                )
            )
            self._material_connected_pou_cache[cache_key] = build_material_connected_partition_of_unity(
                np.asarray(self.elements_i.el2v.to_numpy(), dtype=np.int64),
                material_keys,
                self.n_vertices,
                max_partitions_per_component=max_partitions_per_component,
                target_tets_per_partition=target_tets_per_partition,
                smoothing_steps=smoothing_steps,
            )
        return self._material_connected_pou_cache[cache_key]

    def get_sparse_direct_geometry_candidates(
        self,
        i_b: int,
        *,
        max_partitions_per_component: int = 8,
        target_tets_per_partition: int = 128,
        smoothing_steps: int = 3,
    ):
        """Return current-position PoU translation/rotation candidates in vertex-major XYZ order."""
        cache_key = (max_partitions_per_component, target_tets_per_partition, smoothing_steps)
        if cache_key not in self._direct_geometry_candidates[i_b]:
            weights, partition_metadata = self.get_material_connected_partition_of_unity(
                max_partitions_per_component=max_partitions_per_component,
                target_tets_per_partition=target_tets_per_partition,
                smoothing_steps=smoothing_steps,
            )
            self._direct_geometry_candidates[i_b][cache_key] = build_weighted_rigid_motion_candidates(
                self._direct_free_positions[i_b],
                weights,
                vertex_masses=np.asarray(self.elements_v_info.mass.to_numpy(), dtype=np.float64),
                partition_metadata=partition_metadata,
            )
        return self._direct_geometry_candidates[i_b][cache_key]

    def batch_solve(self, f: qd.i32):
        self.batch_active.fill(True)
        self.batch_pcg_budget_exhausted.fill(False)
        self.batch_pcg_breakdown.fill(False)
        self.batch_linesearch_budget_exhausted.fill(False)

        if self._linear_solver == "sparse_direct":
            self._direct_assembly_time_s = 0.0
            self._direct_factor_time_s = 0.0
            self._direct_solve_time_s = 0.0
            self._direct_transfer_time_s = 0.0
        elif self._enable_material_coarse_preconditioner:
            self._refresh_material_coarse_basis(f)

        for i in range(self._n_newton_iterations):
            # compute element energy and gradient
            self.compute_ele_hessian_gradient(f)

            # If the hessian is invariant, we only need to compute it once
            for mat_idx in self._mats_idx:
                if self._mats[mat_idx].hessian_invariant:
                    self._mats[mat_idx]._hessian_ready = True

            # accumulate vertex force and preconditioner
            self.accumulate_vertex_force_preconditioner(f)
            if self._linear_solver == "sparse_direct":
                self._sparse_direct_solve(f)
            else:
                if self._enable_rigid_mode_deflation:
                    self._prepare_rigid_mode_coarse_operator()
                if self._enable_material_coarse_preconditioner:
                    self._prepare_material_coarse_operator()

                # solve for the vertex positions
                self.pcg_solve()
                self._accumulate_pcg_budget_exhaustion()

            # line search
            if self._enable_development_implicit_fem_positive_j_feasible_step:
                if self._enable_development_implicit_fem_positive_j_alpha_one_only:
                    # Alpha=1-only direct replay: commit the unmodified full
                    # PCG update and let setup_pos_vel derive velocity from it.
                    self.skip_linesearch(f)
                else:
                    self._apply_development_implicit_fem_positive_j_feasible_step(f)
            else:
                self.linesearch(f)
            self._accumulate_linesearch_budget_exhaustion()

        if self._linear_solver == "sparse_direct":
            self._finalize_sparse_direct_velocity_system(f)

    @qd.kernel
    def _accumulate_pcg_budget_exhaustion(self):
        for i_b in range(self._B):
            self.batch_pcg_budget_exhausted[i_b] |= self.batch_pcg_active[i_b]

    @qd.kernel
    def _accumulate_linesearch_budget_exhaustion(self):
        for i_b in range(self._B):
            self.batch_linesearch_budget_exhausted[i_b] |= self.batch_linesearch_active[i_b]

    @qd.kernel
    def setup_pos_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # set pos and vel
            self.elements_v[f + 1, i_v, i_b].vel = (
                self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
            ) / self.substep_dt

    def _development_implicit_fem_positive_j_feasible_step_health(self):
        if (
            not self._enable_development_implicit_fem_positive_j_feasible_step
            or self._enable_development_implicit_fem_positive_j_alpha_one_only
        ):
            return None
        return ImplicitFEMPositiveJFeasibleStep(
            pre_update_base_min_j=tuple(
                np.asarray(self._development_implicit_fem_positive_j_base_min_j.to_numpy(), dtype=np.float64).tolist()
            ),
            unfiltered_fem_trial_min_j=tuple(
                np.asarray(self._development_implicit_fem_positive_j_trial_min_j.to_numpy(), dtype=np.float64).tolist()
            ),
            accepted_fem_alpha=tuple(
                np.asarray(self._development_implicit_fem_positive_j_accepted_alpha.to_numpy(), dtype=np.float64).tolist()
            ),
            witness_tet_id=tuple(
                np.asarray(self._development_implicit_fem_positive_j_witness_tet_id.to_numpy(), dtype=np.int64).tolist()
            ),
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip FEM solver step if using IPCCoupler (IPC handles FEM simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim._coupler, IPCCoupler):
                pass  # IPC coupler handles FEM simulation
            elif self._use_implicit_solver:
                self.precompute_material_data(f)
                self.init_pos_and_inertia(f)
                self.batch_solve(f)
                self.setup_pos_vel(f)
            else:
                self.init_pos_and_vel(f)
                self.compute_vel(f)
                self.apply_uniform_force(f)
                if self._constraints_initialized:
                    self.apply_soft_constraints(f)

    def substep_pre_coupling_grad(self, f):
        if self.is_active:
            if self._use_implicit_solver:
                gs.raise_exception("Gradient computation is not supported for implicit solver.")
            self.apply_uniform_force.grad(f)
            self.compute_vel.grad(f)
            self.init_pos_and_vel.grad(f)

    def substep_post_coupling(self, f):
        if self.is_active:
            self.compute_pos(f)
            if self._constraints_initialized and not self._use_implicit_solver:
                self.apply_hard_constraints(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active:
            self.compute_pos.grad(f)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        # Copy pos/vel for all vertices and all batch indices
        for i_v, i_b in qd.ndrange(self.n_vertices_max, self._B):
            self.elements_v[target, i_v, i_b].pos = self.elements_v[source, i_v, i_b].pos
            self.elements_v[target, i_v, i_b].vel = self.elements_v[source, i_v, i_b].vel

        # Copy 'active' for all elements and all batch indices
        for i_e, i_b in qd.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @qd.kernel
    def copy_grad(self, source: qd.i32, target: qd.i32):
        # Copy gradients for vertices
        for i_v, i_b in qd.ndrange(self.n_vertices_max, self._B):
            self.elements_v.grad[target, i_v, i_b].pos = self.elements_v.grad[source, i_v, i_b].pos
            self.elements_v.grad[target, i_v, i_b].vel = self.elements_v.grad[source, i_v, i_b].vel

        # Copy 'active' for elements
        for i_e, i_b in qd.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @qd.kernel
    def reset_grad_till_frame(self, f: qd.i32):
        # Zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
        for frame_i, vert_i, i_b in qd.ndrange(f, self.n_vertices_max, self._B):
            self.elements_v.grad[frame_i, vert_i, i_b].pos = 0
            self.elements_v.grad[frame_i, vert_i, i_b].vel = 0

        # Zero out elements_el.grad in frame 0..(f-1) for all elements, all batch indices
        for frame_i, elem_i, i_b in qd.ndrange(f, self.n_elements_max, self._B):
            self.elements_el.grad[frame_i, elem_i, i_b].actu = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self._kernel_add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self._kernel_add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

    def save_ckpt(self, ckpt_name):
        if self.is_active:
            if self._sim.requires_grad:
                if ckpt_name not in self._ckpt:
                    self._ckpt[ckpt_name] = dict()
                    self._ckpt[ckpt_name]["pos"] = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float)
                    self._ckpt[ckpt_name]["vel"] = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float)
                    self._ckpt[ckpt_name]["active"] = torch.zeros((self._B, self.n_elements), dtype=gs.tc_int)

                self._kernel_get_state(
                    0, self._ckpt[ckpt_name]["pos"], self._ckpt[ckpt_name]["vel"], self._ckpt[ckpt_name]["active"]
                )

            self.copy_frame(self.sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            self._kernel_set_state_envs(f, state.pos, state.vel, state.active, envs_idx)

    def get_state(self, f):
        if self.is_active:
            state = FEMSolverState(self._scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def get_completed_control_step_vertices(self):
        """Return all local completed FEM frames ``1..substeps_local`` at once.

        M3 uses this only where one control step is retained locally
        (``substeps_local == substeps``).  Frame zero may already have been
        refreshed from the final frame by ``save_ckpt``; the completed frames
        themselves remain intact.
        """
        if not self.is_active:
            return None
        frames = torch.empty(
            (self._B, self.sim.substeps_local, self.n_vertices, 3),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self._kernel_get_completed_control_step_vertices(frames)
        return frames

    def get_completed_substep_safety_extrema(
        self, *, completed_frame: int, global_substep_index: int | None = None
    ) -> FEMSubstepSafetyExtrema | None:
        """Return one conservative extrema record for a completed implicit-FEM frame.

        This is intentionally a small solver-health reduction: it exports no
        vertex or element history. It is a B=1 qualification-only CPU
        readback and reduction, not a K-way runtime telemetry API. It supports
        active volumetric ``linear_corotated`` entities, whose energy
        definition matches the implicit FEM solver. Other configurations return
        ``None`` so a caller requiring this safety evidence can fail closed.
        """
        if not self.is_active or not self._use_implicit_solver or self._B != 1:
            return None
        if type(completed_frame) is not int or not (1 <= completed_frame <= self.sim.substeps_local):
            raise ValueError("completed FEM safety frame lies outside the local frame range")
        state = self.get_state(completed_frame)
        if state is None:
            return None
        positions = np.asarray(tensor_to_array(state.pos), dtype=np.float64)
        active = np.asarray(tensor_to_array(state.active), dtype=np.bool_)
        if positions.shape != (self._B, self.n_vertices, 3) or active.shape != (self._B, self.n_elements):
            raise ValueError("implicit FEM completed-state shape does not match solver topology")

        extrema: list[FEMSubstepSafetyExtrema] = []
        for entity_index, entity in enumerate(self._entities):
            material = entity.material
            if getattr(material, "model", None) != "linear_corotated":
                return None
            tetrahedra = np.asarray(entity.elems, dtype=np.int64)
            if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
                return None
            heterogeneous = entity._heterogeneous_material_np
            if heterogeneous is None:
                lame_mu = np.full(entity.n_elements, material.mu, dtype=np.float64)
                lame_lambda = np.full(entity.n_elements, material.lam, dtype=np.float64)
            else:
                lame_mu = np.asarray(heterogeneous.mu, dtype=np.float64)
                lame_lambda = np.asarray(heterogeneous.lam, dtype=np.float64)
            entity_extrema = _linear_corotated_safety_extrema(
                rest_positions=np.asarray(tensor_to_array(entity.init_positions), dtype=np.float64),
                tetrahedra=tetrahedra,
                current_positions=positions[:, entity.v_start : entity.v_start + entity.n_vertices],
                active=active[:, entity.el_start : entity.el_start + entity.n_elements],
                lame_mu=lame_mu,
                lame_lambda=lame_lambda,
                global_substep_index=global_substep_index,
                env_index=0,
                fem_entity_index=entity.idx,
                fem_entity_name=getattr(entity, "name", None),
                vertex_global_offset=entity.v_start,
                tet_global_offset=entity.el_start,
                floor_height_m=float(self._floor_height),
            )
            if entity_extrema is None:
                return None
            extrema.append(entity_extrema)
        if not extrema:
            return None
        max_strain = max(item.max_principal_stretch_strain for item in extrema)
        strain_witnesses = tuple(
            item.principal_strain_witness
            for item in extrema
            if item.principal_strain_witness is not None and item.principal_strain_witness.principal_stretch_strain == max_strain
        )
        witness = min(
            strain_witnesses,
            key=lambda item: (item.fem_entity_index, item.tet_local_index),
        ) if strain_witnesses else None
        return FEMSubstepSafetyExtrema(
            min_j=min(item.min_j for item in extrema),
            max_principal_stretch_strain=max_strain,
            max_tet_elastic_energy_j=max(item.max_tet_elastic_energy_j for item in extrema),
            total_elastic_energy_j=sum(item.total_elastic_energy_j for item in extrema),
            no_inversion=all(item.no_inversion for item in extrema),
            principal_strain_witness=witness,
        )

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        vertices = self.surface_render_v.vertices
        indices = self.surface_render_f.indices
        uvs = self.surface_render_uvs

        return vertices, indices, uvs

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None

        return qd_to_torch(self.elements_v_energy.force, copy=True)

    @qd.kernel
    def _kernel_add_elements(
        self,
        f: qd.i32,
        mat_idx: qd.i32,
        mat_mu_per_el: qd.types.ndarray(),
        mat_lam_per_el: qd.types.ndarray(),
        mat_rho_per_el: qd.types.ndarray(),
        mat_friction_mu_per_el: qd.types.ndarray(),
        n_surfaces: qd.i32,
        v_start: qd.i32,
        el_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        elems: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
        tri2el: qd.types.ndarray(),
        uvs: qd.types.ndarray(),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        # Copy UVs to solver field (skip if no UVs provided)
        n_uvs = uvs.shape[0]
        for i_v in range(n_uvs):
            i_global = i_v + v_start
            self.surface_render_uvs[i_global] = qd.Vector([uvs[i_v, 0], uvs[i_v, 1]])

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 0.0
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = mat_friction_mu_per_el[0]

        dt2_inv = 1.0 / (self.substep_dt**2)
        n_elems_local = elems.shape[0]
        for i_e in range(n_elems_local):
            i_global = i_e + el_start

            a = self.elements_v[f, elems[i_e, 0] + v_start, 0].pos
            b = self.elements_v[f, elems[i_e, 1] + v_start, 0].pos
            c = self.elements_v[f, elems[i_e, 2] + v_start, 0].pos
            d = self.elements_v[f, elems[i_e, 3] + v_start, 0].pos
            B_inv = qd.Matrix.cols([a - d, b - d, c - d])
            self.elements_i[i_global].B = B_inv.inverse()
            det = B_inv.determinant()
            # Determinant should be consistently smaller than 0
            if det >= 0.0:
                self.tet_wrong_order[None] = True
            V = qd.abs(det) / 6.0
            self.elements_i[i_global].V = V
            V_scaled = V * self._vol_scale
            self.elements_i[i_global].V_scaled = V_scaled

            for j in qd.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i_e, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu_per_el[i_e]
            self.elements_i[i_global].lam = mat_lam_per_el[i_e]
            self.elements_i[i_global].friction_mu = mat_friction_mu_per_el[i_e]
            self.elements_i[i_global].mass_scaled = mat_rho_per_el[i_e] * V_scaled
            for j in qd.static(range(4)):
                mass = 0.25 * mat_rho_per_el[i_e] * V
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass += mass
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass_over_dt2 += mass * dt2_inv
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass_inv = 1.0 / self.elements_v_info[i_global].mass

        for i_e, i_b in qd.ndrange(n_elems_local, self._B):
            i_global = i_e + el_start
            self.elements_el[f, i_global, i_b].actu = 0.0
            self.elements_el_ng[f, i_global, i_b].active = True

        for i_s in range(n_surfaces):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = tri2el[i_s] + el_start
            self.surface[i_global].active = True

    @qd.kernel
    def _kernel_add_cloth_for_rendering(
        self,
        f: qd.i32,
        n_surfaces: qd.i32,
        v_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
        uvs: qd.types.ndarray(),
    ):
        """
        Add cloth vertices and surfaces for rendering only (no physics computation).
        Cloth is simulated by IPC, but needs to be in FEM solver's rendering pipeline.
        """
        # Add vertices for rendering
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        # Copy UVs to solver field (skip if no UVs provided)
        n_uvs = uvs.shape[0]
        for i_v in range(n_uvs):
            i_global = i_v + v_start
            self.surface_render_uvs[i_global] = qd.Vector([uvs[i_v, 0], uvs[i_v, 1]])

        # Initialize vertex info (mass will be managed by IPC, set to dummy value)
        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 1.0  # Dummy value, not used for cloth
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = 0.0

        # Add surface triangles for rendering
        for i_s in range(n_surfaces):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            # For cloth, tri2el points to itself (no tetrahedral element)
            self.surface[i_global].tri2el = i_global
            self.surface[i_global].active = True

    @qd.kernel
    def _kernel_set_elements_pos(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        pos: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[k] = pos[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_pos_grad(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        pos_grad: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].pos[k] = pos_grad[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_vel(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].vel[k] = vel[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_vel_grad(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        vel_grad: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].vel[k] = vel_grad[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_actu(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        n_groups: qd.i32,
        actu: qd.types.ndarray(),  # shape [B, n_elements, n_groups]
    ):
        for i_e, j_g, i_b in qd.ndrange(n_elements, n_groups, self._B):
            i_global = i_e + element_el_start
            if self.elements_i[i_global].muscle_group == j_g:
                self.elements_el[f, i_global, i_b].actu = actu[i_b, j_g]

    @qd.kernel
    def _kernel_set_elements_actu_grad(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        actu_grad: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in qd.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el.grad[f, i_global, i_b].actu = actu_grad[i_b, i_e]

    @qd.kernel
    def _kernel_set_active(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in qd.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el_ng[f, i_global, i_b].active = active[i_b, i_e]

    @qd.kernel
    def _kernel_set_muscle_group(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        muscle_group: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            self.elements_i[i_global].muscle_group = muscle_group[i_e]

    @qd.kernel
    def _kernel_set_muscle_direction(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        muscle_direction: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in qd.static(range(3)):
                self.elements_i[i_global].muscle_direction[j] = muscle_direction[i_e, j]

    @qd.kernel
    def _kernel_get_el2v(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        el2v: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in qd.static(range(4)):
                el2v[i_global, j] = self.elements_i[i_global].el2v[j]

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self.elements_v[f, i_v, i_b].pos[j]
                vel[i_b, i_v, j] = self.elements_v[f, i_v, i_b].vel[j]

        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            active[i_b, i_e] = self.elements_el_ng[f, i_e, i_b].active

    @qd.kernel
    def _kernel_get_completed_control_step_vertices(
        self,
        vertices: qd.types.ndarray(),  # [B,S,V,3]
    ):
        for i_b, f, i_v in qd.ndrange(self._B, self.sim.substeps_local, self.n_vertices):
            for j in qd.static(range(3)):
                vertices[i_b, f, i_v, j] = self.elements_v[f + 1, i_v, i_b].pos[j]

    @qd.kernel
    def get_state_render_kernel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.elements_v[f, i_v, i_b].pos[j], qd.f32)
                self.surface_render_v[i_v, i_b].vertices[j] = pos_j + self.envs_offset[i_b][j]

        # Fill triangle indices (flat array, 3 ints per triangle)
        for i_s in range(self.n_surfaces):
            for j in qd.static(range(3)):
                self.surface_render_f[i_s * 3 + j].indices = qd.cast(self.surface[i_s].tri2v[j], qd.i32)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.elements_v[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            self.elements_el_ng[f, i_e, i_b].active = active[i_b, i_e]

    @qd.kernel
    def _kernel_set_state_envs(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        active: qd.types.ndarray(),  # shape [B, n_elements]
        envs_idx: qd.types.ndarray(),  # shape [n_selected]
    ):
        """Restore only the selected batch rows from a full public FEM state."""
        for i_v, i_b_ in qd.ndrange(self.n_vertices, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for j in qd.static(range(3)):
                self.elements_v[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.elements_v[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

        for i_e, i_b_ in qd.ndrange(self.n_elements, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.elements_el_ng[f, i_e, i_b].active = active[i_b, i_e]

    @qd.kernel
    def _kernel_add_grad_from_pos(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].pos[j] += pos_grad[i_b, i_v, j]

    @qd.kernel
    def _kernel_add_grad_from_vel(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].vel[j] += vel_grad[i_b, i_v, j]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height

    @property
    def enable_floor(self):
        return self._enable_floor

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vertices_max(self):
        return self._n_vertices_max

    @property
    def n_elements_max(self):
        return self._n_elements_max

    @property
    def vol_scale(self):
        return self._vol_scale

    @property
    def n_surface_vertices(self):
        return self.surface_vertices.shape[0]

    @property
    def n_surface_elements(self):
        return self.surface_elements.shape[0]

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertex constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_update_linked_vertex_constraints(
        self,
        links_state: array_class.LinksState,
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.link_idx >= 0:
                i_l = vc.link_idx
                pos = links_state.pos[i_l, i_b]
                quat = links_state.quat[i_l, i_b]

                offset_pos = vc.link_offset_pos
                offset_quat = qd_transform_quat_by_quat(vc.link_init_quat, quat)
                self.vertex_constraints[i_v, i_b].target_pos = pos + qd_transform_by_quat(offset_pos, offset_quat)

    @qd.kernel
    def apply_hard_constraints(self, f: qd.i32):
        """Apply hard constraints by directly overriding positions and velocities."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                self.elements_v[f + 1, i_v, i_b].pos = vc.target_pos
                self.elements_v[f + 1, i_v, i_b].vel.fill(0.0)

    @qd.kernel
    def apply_soft_constraints(self, f: qd.i32):
        """Apply soft constraints as spring forces for explicit solver."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.is_soft_constraint:
                pos_error = self.elements_v[f, i_v, i_b].pos - vc.target_pos
                vel_error = self.elements_v[f + 1, i_v, i_b].vel - self.elements_v[f, i_v, i_b].vel
                spring_force = -vc.stiffness * pos_error
                damping_force = -2.0 * qd.math.sqrt(vc.stiffness) * vel_error

                dv = self.substep_dt * (spring_force + damping_force)
                self.elements_v[f + 1, i_v, i_b].vel += dv

    @qd.kernel
    def _kernel_set_vertex_constraints(
        self,
        f: qd.i32,
        verts_idx: qd.types.ndarray(),  # shape [B, V]
        target_poss: qd.types.ndarray(),  # shape [B, V, 3]
        is_soft_constraint: qd.i32,
        stiffness: qd.f32,
        link_idx: qd.i32,
        link_init_pos: qd.types.ndarray(),  # shape [B, 3]
        link_init_quat: qd.types.ndarray(),  # shape [B, 4]
        envs_idx: qd.types.ndarray(),  # shape [B]
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            self.vertex_constraints[i_v, i_b].is_constrained = True
            self.vertex_constraints[i_v, i_b].is_soft_constraint = qd.cast(is_soft_constraint, gs.qd_bool)
            self.vertex_constraints[i_v, i_b].stiffness = stiffness
            self.vertex_constraints[i_v, i_b].link_idx = link_idx

            cur_pos = self.elements_v[f, i_v, i_b].pos
            for j in qd.static(range(3)):
                self.vertex_constraints[i_v, i_b].target_pos[j] = target_poss[i_b_, i_v_, j]
                self.vertex_constraints[i_v, i_b].link_offset_pos[j] = cur_pos[j] - link_init_pos[i_b_, j]
            for j in qd.static(range(4)):
                self.vertex_constraints[i_v, i_b].link_init_quat[j] = link_init_quat[i_b_, j]

    @qd.kernel
    def _kernel_update_constraint_targets(
        self, verts_idx: qd.types.ndarray(), new_target_poss: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            for j in qd.static(range(3)):
                self.vertex_constraints[i_v, i_b].target_pos[j] = new_target_poss[i_b_, i_v_, j]

    @qd.kernel
    def _kernel_remove_specific_constraints(self, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            self.vertex_constraints[i_v, i_b].is_constrained = False
