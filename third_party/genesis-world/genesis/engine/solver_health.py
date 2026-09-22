"""Immutable public health records for completed SAP control steps.

The records are snapshots of the solver fields after each physical substep.
They deliberately distinguish a substep with no contact solve from a solved
substep, and they never turn a failed or partial control step into a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class SolverHealthError(RuntimeError):
    """Base class for public solver-health query failures."""


class SolverHealthUnavailableError(SolverHealthError):
    """Raised when a scene is not backed by the SAP solver-health producer."""


class SolverHealthNotReadyError(SolverHealthError):
    """Raised when no complete successful Scene.step has published a record."""


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")


def _finite_nonnegative(value: object, name: str) -> None:
    if type(value) is not float or not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")


def _float_value(value: object, name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")


@dataclass(frozen=True, slots=True)
class ImplicitFEMTrueResidualSample:
    """One immutable scalar-only true-residual PCG checkpoint."""

    completed_iteration: int
    actual_iterations_by_batch: tuple[int, ...]
    pcg_active_by_batch: tuple[bool, ...]
    true_residual_squared_by_batch: tuple[float, ...]
    recursive_residual_squared_by_batch: tuple[float, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.completed_iteration, "completed_iteration")
        batch_size = len(self.actual_iterations_by_batch)
        if batch_size == 0 or any(
            len(values) != batch_size
            for values in (
                self.pcg_active_by_batch,
                self.true_residual_squared_by_batch,
                self.recursive_residual_squared_by_batch,
            )
        ):
            raise ValueError("true-residual sample fields must have one value per batch")
        for value in self.actual_iterations_by_batch:
            _nonnegative_int(value, "actual_iterations_by_batch value")
        if any(type(value) is not bool for value in self.pcg_active_by_batch):
            raise TypeError("pcg_active_by_batch must contain bool values")
        for name in ("true_residual_squared_by_batch", "recursive_residual_squared_by_batch"):
            for value in getattr(self, name):
                _finite_nonnegative(value, f"{name} value")


@dataclass(frozen=True, slots=True)
class ImplicitFEMTrueResidualProbe:
    """Five ordered scalar-only checkpoints from one implicit FEM PCG solve."""

    global_substep_index: int
    samples: tuple[ImplicitFEMTrueResidualSample, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.global_substep_index, "global_substep_index")
        if tuple(sample.completed_iteration for sample in self.samples) != (0, 50, 100, 250, 500):
            raise ValueError("true-residual probe schedule must be exactly (0, 50, 100, 250, 500)")
        batch_size = len(self.samples[0].actual_iterations_by_batch)
        if any(len(sample.actual_iterations_by_batch) != batch_size for sample in self.samples):
            raise ValueError("true-residual probe batch size must be constant")
        for batch_index in range(batch_size):
            actual = tuple(sample.actual_iterations_by_batch[batch_index] for sample in self.samples)
            if any(next_value < value for value, next_value in zip(actual, actual[1:])):
                raise ValueError("true-residual actual iteration counts must be nondecreasing")

    @property
    def finite(self) -> bool:
        return all(
            isfinite(value) and value >= 0.0
            for sample in self.samples
            for value in (
                *sample.true_residual_squared_by_batch,
                *sample.recursive_residual_squared_by_batch,
            )
        )


@dataclass(frozen=True, slots=True)
class ImplicitFEMSubstepHealth:
    """Actual final state of FEM's independent implicit solve for one substep.

    ``batch_active_by_batch`` is an activity/participation mask.  It is not
    represented as an outer-Newton convergence mask by the current solver, so
    the public qualification check uses the explicit final-update threshold and
    accumulated inner-budget flags below.
    """

    batch_active_by_batch: tuple[bool, ...]
    pcg_active_by_batch: tuple[bool, ...]
    linesearch_active_by_batch: tuple[bool, ...]
    pcg_budget_exhausted_by_batch: tuple[bool, ...]
    pcg_breakdown_by_batch: tuple[bool, ...]
    linesearch_budget_exhausted_by_batch: tuple[bool, ...]
    pcg_initial_residual_squared_by_batch: tuple[float, ...]
    pcg_residual_squared_by_batch: tuple[float, ...]
    pcg_preconditioned_residual_by_batch: tuple[float, ...]
    pcg_relative_residual_norm_by_batch: tuple[float, ...]
    pcg_effective_residual_squared_threshold_by_batch: tuple[float, ...]
    max_newton_update_m_by_batch: tuple[float, ...]
    newton_iteration_budget: int
    pcg_iteration_budget: int
    linesearch_iteration_budget: int
    newton_dx_threshold_m: float
    """Final-update qualification threshold for the independent implicit Newton solve."""
    pcg_threshold: float
    """Legacy name for the squared absolute residual floor; retained for compatibility."""
    pcg_absolute_residual_squared_floor: float
    pcg_rtol: float
    pcg_iterations_by_batch: tuple[int, ...] = (0,)
    rigid_mode_deflation_enabled: bool = False
    rigid_mode_coarse_matrix_finite_by_batch: tuple[bool, ...] | None = None
    rigid_mode_coarse_inverse_finite_by_batch: tuple[bool, ...] | None = None
    true_residual_probe: ImplicitFEMTrueResidualProbe | None = None
    linear_solver: str = "pcg"
    sparse_direct_true_residual_norm_by_batch: tuple[float, ...] = ()
    sparse_direct_assembly_time_s: float = 0.0
    sparse_direct_factor_time_s: float = 0.0
    sparse_direct_solve_time_s: float = 0.0
    sparse_direct_transfer_time_s: float = 0.0

    def __post_init__(self) -> None:
        batch_size = len(self.batch_active_by_batch)
        fields = (
            "pcg_active_by_batch",
            "linesearch_active_by_batch",
            "pcg_budget_exhausted_by_batch",
            "pcg_breakdown_by_batch",
            "linesearch_budget_exhausted_by_batch",
            "pcg_initial_residual_squared_by_batch",
            "pcg_residual_squared_by_batch",
            "pcg_preconditioned_residual_by_batch",
            "pcg_relative_residual_norm_by_batch",
            "pcg_effective_residual_squared_threshold_by_batch",
            "max_newton_update_m_by_batch",
        )
        if batch_size == 0 or any(len(getattr(self, name)) != batch_size for name in fields):
            raise ValueError("implicit FEM fields must have one value per batch")
        if len(self.pcg_iterations_by_batch) != batch_size:
            raise ValueError("pcg_iterations_by_batch must have one value per batch")
        for value in self.pcg_iterations_by_batch:
            _nonnegative_int(value, "pcg_iterations_by_batch value")
        if type(self.rigid_mode_deflation_enabled) is not bool:
            raise TypeError("rigid_mode_deflation_enabled must be a bool")
        coarse_fields = (
            "rigid_mode_coarse_matrix_finite_by_batch",
            "rigid_mode_coarse_inverse_finite_by_batch",
        )
        if self.rigid_mode_deflation_enabled:
            for name in coarse_fields:
                values = getattr(self, name)
                if values is None or len(values) != batch_size:
                    raise ValueError(f"{name} must have one value per batch when rigid-mode deflation is enabled")
                if any(type(value) is not bool for value in values):
                    raise TypeError(f"{name} must contain bool values")
        elif any(getattr(self, name) is not None for name in coarse_fields):
            raise ValueError("rigid-mode coarse finiteness must be None when deflation is disabled")
        if self.true_residual_probe is not None and not isinstance(
            self.true_residual_probe, ImplicitFEMTrueResidualProbe
        ):
            raise TypeError("true_residual_probe must be ImplicitFEMTrueResidualProbe or None")
        if self.linear_solver not in ("pcg", "sparse_direct"):
            raise ValueError("linear_solver must be 'pcg' or 'sparse_direct'")
        if self.linear_solver == "sparse_direct":
            if len(self.sparse_direct_true_residual_norm_by_batch) != batch_size:
                raise ValueError("sparse-direct FEM residuals must have one value per batch")
        elif self.sparse_direct_true_residual_norm_by_batch:
            raise ValueError("PCG FEM health cannot carry sparse-direct residuals")
        for name in (
            "batch_active_by_batch",
            "pcg_active_by_batch",
            "linesearch_active_by_batch",
            "pcg_budget_exhausted_by_batch",
            "pcg_breakdown_by_batch",
            "linesearch_budget_exhausted_by_batch",
        ):
            if any(type(value) is not bool for value in getattr(self, name)):
                raise TypeError(f"{name} must contain bool values")
        for name in (
            "pcg_initial_residual_squared_by_batch",
            "pcg_residual_squared_by_batch",
            "pcg_preconditioned_residual_by_batch",
            "pcg_relative_residual_norm_by_batch",
            "pcg_effective_residual_squared_threshold_by_batch",
            "max_newton_update_m_by_batch",
        ):
            if any(type(value) is not float for value in getattr(self, name)):
                raise TypeError(f"{name} must contain float values")
        for name in ("newton_iteration_budget", "pcg_iteration_budget", "linesearch_iteration_budget"):
            _nonnegative_int(getattr(self, name), name)
        _finite_nonnegative(self.newton_dx_threshold_m, "newton_dx_threshold_m")
        _finite_nonnegative(self.pcg_threshold, "pcg_threshold")
        _finite_nonnegative(self.pcg_absolute_residual_squared_floor, "pcg_absolute_residual_squared_floor")
        _finite_nonnegative(self.pcg_rtol, "pcg_rtol")
        for value in self.sparse_direct_true_residual_norm_by_batch:
            _finite_nonnegative(value, "sparse_direct_true_residual_norm_by_batch value")
        for name in (
            "sparse_direct_assembly_time_s",
            "sparse_direct_factor_time_s",
            "sparse_direct_solve_time_s",
            "sparse_direct_transfer_time_s",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.pcg_threshold != self.pcg_absolute_residual_squared_floor:
            raise ValueError("pcg_threshold must equal the explicit squared absolute residual floor")

    @property
    def finite(self) -> bool:
        return (self.true_residual_probe is None or self.true_residual_probe.finite) and all(
            isfinite(value) and value >= 0.0
            for value in (
                *self.pcg_initial_residual_squared_by_batch,
                *self.pcg_residual_squared_by_batch,
                *self.pcg_preconditioned_residual_by_batch,
                *self.pcg_relative_residual_norm_by_batch,
                *self.pcg_effective_residual_squared_threshold_by_batch,
                *self.max_newton_update_m_by_batch,
                *self.sparse_direct_true_residual_norm_by_batch,
                self.sparse_direct_assembly_time_s,
                self.sparse_direct_factor_time_s,
                self.sparse_direct_solve_time_s,
                self.sparse_direct_transfer_time_s,
            )
        )

    @property
    def converged(self) -> bool:
        return (
            not any(self.pcg_active_by_batch)
            and not any(self.linesearch_active_by_batch)
            and not any(self.pcg_budget_exhausted_by_batch)
            and not any(self.pcg_breakdown_by_batch)
            and not any(self.linesearch_budget_exhausted_by_batch)
            and all(value <= self.newton_dx_threshold_m for value in self.max_newton_update_m_by_batch)
        )

    @property
    def healthy(self) -> bool:
        return self.finite and self.converged


@dataclass(frozen=True, slots=True)
class FEMPrincipalStrainWitness:
    """Immutable public argmax witness for one completed FEM substep.

    This is deliberately one bounded record, not a per-element telemetry
    surface.  The geometry is retained in canonical tetrahedron connectivity
    order so downstream consumers can independently derive the centroid and
    the floor-straddling predicate.
    """

    global_substep_index: int
    env_index: int
    fem_entity_index: int
    fem_entity_name: str
    tet_local_index: int
    tet_global_index: int
    tet_entity_local_vertex_indices: tuple[int, int, int, int]
    tet_global_vertex_indices: tuple[int, int, int, int]
    current_vertex_positions_m: tuple[tuple[float, float, float], ...]
    current_centroid_m: tuple[float, float, float]
    principal_stretch_strain: float
    floor_height_m: float
    floor_tet_candidate_geometrically_possible: bool

    def __post_init__(self) -> None:
        for name in (
            "global_substep_index",
            "env_index",
            "fem_entity_index",
            "tet_local_index",
            "tet_global_index",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.env_index != 0:
            raise ValueError("M2 FEM principal-strain witness requires env_index == 0")
        if not isinstance(self.fem_entity_name, str) or not self.fem_entity_name:
            raise TypeError("fem_entity_name must be a nonempty string")
        for name in ("tet_entity_local_vertex_indices", "tet_global_vertex_indices"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 4
                or any(type(item) is not int or item < 0 for item in value)
            ):
                raise ValueError(f"{name} must contain four nonnegative integer indices")
        positions = self.current_vertex_positions_m
        if not isinstance(positions, tuple) or len(positions) != 4:
            raise ValueError("current_vertex_positions_m must contain four vertices")
        if any(
            not isinstance(vertex, tuple)
            or len(vertex) != 3
            or any(type(component) is not float or not isfinite(component) for component in vertex)
            for vertex in positions
        ):
            raise ValueError("current tetrahedron vertices must be finite float triples")
        centroid = self.current_centroid_m
        if (
            not isinstance(centroid, tuple)
            or len(centroid) != 3
            or any(type(component) is not float or not isfinite(component) for component in centroid)
        ):
            raise ValueError("current_centroid_m must be a finite float triple")
        derived_centroid = tuple(float(sum(vertex[axis] for vertex in positions) / 4.0) for axis in range(3))
        if centroid != derived_centroid:
            raise ValueError("current_centroid_m must be derived from current_vertex_positions_m")
        if (
            type(self.principal_stretch_strain) is not float
            or not isfinite(self.principal_stretch_strain)
            or self.principal_stretch_strain < 0.0
        ):
            raise ValueError("principal_stretch_strain must be finite and nonnegative")
        _float_value(self.floor_height_m, "floor_height_m")
        if not isfinite(self.floor_height_m):
            raise ValueError("floor_height_m must be finite")
        derived_floor_candidate = any(vertex[2] > self.floor_height_m for vertex in positions) and any(
            vertex[2] <= self.floor_height_m for vertex in positions
        )
        if self.floor_tet_candidate_geometrically_possible != derived_floor_candidate:
            raise ValueError("floor_tet_candidate_geometrically_possible must derive from vertex positions")


@dataclass(frozen=True, slots=True)
class FEMSubstepSafetyExtrema:
    """Conservative geometric FEM extrema for one completed physical substep.

    The values reduce every active volumetric implicit-FEM tetrahedron in the
    completed substep.  They are intentionally scalar safety evidence rather
    than a history or a per-element telemetry surface.
    """

    min_j: float
    max_principal_stretch_strain: float
    max_tet_elastic_energy_j: float
    total_elastic_energy_j: float
    no_inversion: bool
    principal_strain_witness: FEMPrincipalStrainWitness | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_j",
            "max_principal_stretch_strain",
            "max_tet_elastic_energy_j",
            "total_elastic_energy_j",
        ):
            _float_value(getattr(self, name), name)
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_principal_stretch_strain < 0.0:
            raise ValueError("max_principal_stretch_strain must be nonnegative")
        if self.max_tet_elastic_energy_j < 0.0 or self.total_elastic_energy_j < 0.0:
            raise ValueError("elastic-energy extrema must be nonnegative")
        if self.no_inversion != bool(self.min_j > 0.0):
            raise ValueError("no_inversion must be derived from min_j")
        if self.principal_strain_witness is not None:
            if not isinstance(self.principal_strain_witness, FEMPrincipalStrainWitness):
                raise TypeError("principal_strain_witness must be FEMPrincipalStrainWitness or None")
            if self.principal_strain_witness.principal_stretch_strain != self.max_principal_stretch_strain:
                raise ValueError("principal-strain witness value must equal substep maximum")


@dataclass(frozen=True, slots=True)
class PositiveJFeasibleStep:
    """Per-batch post-SAP orientation-feasibility evidence for one substep.

    The fixed development floor is intentionally not duplicated as a runtime
    field: a baseline-infeasible substep is evident from ``pre_sap_min_j``
    below the documented 0.20 floor together with ``accepted_alpha == 0``.
    ``witness_tet_id`` is the smallest-index tet attaining the unfiltered
    alpha-one trial minimum when that baseline is feasible; only a baseline
    below the floor reports the corresponding pre-SAP minimum witness.
    """

    pre_sap_min_j: tuple[float, ...]
    unfiltered_post_sap_trial_min_j: tuple[float, ...]
    accepted_alpha: tuple[float, ...]
    witness_tet_id: tuple[int, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.pre_sap_min_j)
        if batch_size == 0 or any(
            len(field) != batch_size
            for field in (
                self.unfiltered_post_sap_trial_min_j,
                self.accepted_alpha,
                self.witness_tet_id,
            )
        ):
            raise ValueError("positive-J feasible-step fields must have one value per batch")
        for name in ("pre_sap_min_j", "unfiltered_post_sap_trial_min_j", "accepted_alpha"):
            for value in getattr(self, name):
                _float_value(value, name)
                if not isfinite(value):
                    raise ValueError(f"{name} must contain finite values")
        if any(value < 0.0 or value > 1.0 for value in self.accepted_alpha):
            raise ValueError("accepted_alpha must lie in [0, 1]")
        if any(type(value) is not int or value < -1 for value in self.witness_tet_id):
            raise ValueError("witness_tet_id must contain tetrahedron indices or -1")


@dataclass(frozen=True, slots=True)
class ImplicitFEMPositiveJFeasibleStep:
    """Per-batch implicit-FEM PCG orientation-feasibility evidence for one substep.

    The documented 0.20 development floor is intentionally not copied into
    this record. A base-infeasible state is instead represented directly by
    ``pre_update_base_min_j`` below that floor and an accepted alpha of zero.
    When the base is feasible, ``witness_tet_id`` identifies the smallest
    global tet index attaining the unfiltered alpha-one PCG-trial minimum;
    otherwise it identifies the smallest base-minimum tet.
    """

    pre_update_base_min_j: tuple[float, ...]
    unfiltered_fem_trial_min_j: tuple[float, ...]
    accepted_fem_alpha: tuple[float, ...]
    witness_tet_id: tuple[int, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.pre_update_base_min_j)
        if batch_size == 0 or any(
            len(field) != batch_size
            for field in (
                self.unfiltered_fem_trial_min_j,
                self.accepted_fem_alpha,
                self.witness_tet_id,
            )
        ):
            raise ValueError("implicit FEM positive-J fields must have one value per batch")
        for name in ("pre_update_base_min_j", "unfiltered_fem_trial_min_j", "accepted_fem_alpha"):
            for value in getattr(self, name):
                _float_value(value, name)
                if not isfinite(value):
                    raise ValueError(f"{name} must contain finite values")
        if any(value < 0.0 or value > 1.0 for value in self.accepted_fem_alpha):
            raise ValueError("accepted_fem_alpha must lie in [0, 1]")
        if any(type(value) is not int or value < -1 for value in self.witness_tet_id):
            raise ValueError("witness_tet_id must contain tetrahedron indices or -1")


@dataclass(frozen=True, slots=True)
class SAPSubstepSolverHealth:
    """Actual SAP and rigid--FEM state for one completed physical substep."""

    global_substep_index: int
    sim_step_index: int
    physical_dt_s: float
    contact_solve_executed: bool
    sap_iteration_budget: int
    pcg_iteration_budget: int
    linesearch_iteration_budget: int
    sap_active_by_batch: tuple[bool, ...]
    pcg_active_by_batch: tuple[bool, ...]
    linesearch_active_by_batch: tuple[bool, ...]
    pcg_budget_exhausted_by_batch: tuple[bool, ...]
    linesearch_budget_exhausted_by_batch: tuple[bool, ...]
    gradient_norm_by_batch: tuple[float, ...]
    momentum_norm_by_batch: tuple[float, ...]
    impulse_norm_by_batch: tuple[float, ...]
    pcg_residual_squared_by_batch: tuple[float, ...]
    pcg_preconditioned_residual_by_batch: tuple[float, ...]
    sap_convergence_atol: float
    sap_convergence_rtol: float
    pcg_threshold: float
    implicit_fem: ImplicitFEMSubstepHealth | None
    rigid_fem_contact_supported: bool
    rigid_fem_contact_pair_count: int
    max_rigid_fem_penetration_m: float | None
    contact_overflow: bool
    unknown_rigid_fem_contact_mode: bool
    unwhitelisted_rigid_fem_link_names: tuple[str, ...]
    fem_safety_extrema: FEMSubstepSafetyExtrema | None = None
    post_final_sap_health_available: bool = False
    post_final_sap_active_by_batch: tuple[bool, ...] = ()
    post_final_gradient_norm_by_batch: tuple[float, ...] = ()
    post_final_momentum_norm_by_batch: tuple[float, ...] = ()
    post_final_impulse_norm_by_batch: tuple[float, ...] = ()
    positive_j_feasible_step: PositiveJFeasibleStep | None = None
    implicit_fem_positive_j_feasible_step: ImplicitFEMPositiveJFeasibleStep | None = None
    rigid_fem_contact_patch_preconditioner_enabled: bool = False
    rigid_fem_contact_patch_min_active_count_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_patch_min_rank_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_patch_max_rank_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_patch_all_coarse_finite_by_batch: tuple[bool, ...] = ()
    rigid_fem_contact_tet_schwarz_preconditioner_enabled: bool = False
    rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_tet_schwarz_max_vertex_overlap_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_tet_schwarz_max_link_overlap_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_tet_schwarz_max_link_rank_by_batch: tuple[int, ...] = ()
    rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch: tuple[float, ...] = ()
    rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch: tuple[bool, ...] = ()
    linear_solver: str = "pcg"
    sparse_direct_sap_iterations_by_batch: tuple[int, ...] = ()
    sparse_direct_final_original_momentum_norm_by_batch: tuple[float, ...] = ()
    sparse_direct_converged_by_batch: tuple[bool, ...] = ()
    sparse_direct_assembly_time_s: float = 0.0
    sparse_direct_factor_time_s: float = 0.0
    sparse_direct_solve_time_s: float = 0.0
    sparse_direct_transfer_time_s: float = 0.0
    enriched_pcg_sap_iterations_by_batch: tuple[int, ...] = ()
    enriched_pcg_final_original_momentum_norm_by_batch: tuple[float, ...] = ()
    enriched_pcg_sap_converged_by_batch: tuple[bool, ...] = ()
    enriched_pcg_geometry_columns_by_batch: tuple[int, ...] = ()
    enriched_pcg_response_columns_by_batch: tuple[int, ...] = ()
    enriched_pcg_retained_columns_by_batch: tuple[int, ...] = ()
    enriched_pcg_iterations_by_batch: tuple[int, ...] = ()
    enriched_pcg_true_residual_norm_by_batch: tuple[float, ...] = ()
    enriched_pcg_linear_converged_by_batch: tuple[bool, ...] = ()
    enriched_pcg_setup_time_s: float = 0.0
    enriched_pcg_solve_time_s: float = 0.0
    enriched_pcg_transfer_time_s: float = 0.0
    contact_schur_sap_iterations_by_batch: tuple[int, ...] = ()
    contact_schur_final_original_momentum_norm_by_batch: tuple[float, ...] = ()
    contact_schur_sap_converged_by_batch: tuple[bool, ...] = ()
    contact_schur_rows_by_batch: tuple[int, ...] = ()
    contact_schur_true_residual_norm_by_batch: tuple[float, ...] = ()
    contact_schur_linear_converged_by_batch: tuple[bool, ...] = ()
    contact_schur_setup_time_s: float = 0.0
    contact_schur_solve_time_s: float = 0.0
    contact_schur_transfer_time_s: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative_int(self.global_substep_index, "global_substep_index")
        _nonnegative_int(self.sim_step_index, "sim_step_index")
        _finite_nonnegative(self.physical_dt_s, "physical_dt_s")
        if self.physical_dt_s == 0.0:
            raise ValueError("physical_dt_s must be positive")
        for name in ("sap_iteration_budget", "pcg_iteration_budget", "linesearch_iteration_budget"):
            _nonnegative_int(getattr(self, name), name)
        for name in (
            "contact_solve_executed",
            "rigid_fem_contact_supported",
            "contact_overflow",
            "unknown_rigid_fem_contact_mode",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        batch_size = len(self.sap_active_by_batch)
        fields = (
            "pcg_active_by_batch",
            "linesearch_active_by_batch",
            "pcg_budget_exhausted_by_batch",
            "linesearch_budget_exhausted_by_batch",
            "gradient_norm_by_batch",
            "momentum_norm_by_batch",
            "impulse_norm_by_batch",
            "pcg_residual_squared_by_batch",
            "pcg_preconditioned_residual_by_batch",
        )
        if any(len(getattr(self, name)) != batch_size for name in fields):
            raise ValueError("all per-batch solver fields must have identical length")
        if self.linear_solver not in ("pcg", "sparse_direct", "enriched_pcg", "contact_schur"):
            raise ValueError("linear_solver must be 'pcg', 'sparse_direct', 'enriched_pcg', or 'contact_schur'")
        direct_fields = (
            "sparse_direct_sap_iterations_by_batch",
            "sparse_direct_final_original_momentum_norm_by_batch",
            "sparse_direct_converged_by_batch",
        )
        if self.linear_solver == "sparse_direct":
            if self.contact_solve_executed and any(
                len(getattr(self, name)) != batch_size for name in direct_fields
            ):
                raise ValueError("sparse-direct SAP health must cover every solved batch")
            if not self.contact_solve_executed and any(getattr(self, name) for name in direct_fields):
                raise ValueError("unexecuted sparse-direct SAP health must be empty")
        elif any(getattr(self, name) for name in direct_fields):
            raise ValueError("non-sparse-direct SAP health cannot carry sparse-direct fields")
        enriched_fields = (
            "enriched_pcg_sap_iterations_by_batch",
            "enriched_pcg_final_original_momentum_norm_by_batch",
            "enriched_pcg_sap_converged_by_batch",
            "enriched_pcg_geometry_columns_by_batch",
            "enriched_pcg_response_columns_by_batch",
            "enriched_pcg_retained_columns_by_batch",
            "enriched_pcg_iterations_by_batch",
            "enriched_pcg_true_residual_norm_by_batch",
            "enriched_pcg_linear_converged_by_batch",
        )
        if self.linear_solver == "enriched_pcg":
            if self.contact_solve_executed and any(
                len(getattr(self, name)) != batch_size for name in enriched_fields
            ):
                raise ValueError("enriched-PCG SAP health must cover every solved batch")
            if not self.contact_solve_executed and any(getattr(self, name) for name in enriched_fields):
                raise ValueError("unexecuted enriched-PCG SAP health must be empty")
        elif any(getattr(self, name) for name in enriched_fields):
            raise ValueError("non-enriched-PCG SAP health cannot carry enriched-PCG fields")
        schur_fields = (
            "contact_schur_sap_iterations_by_batch",
            "contact_schur_final_original_momentum_norm_by_batch",
            "contact_schur_sap_converged_by_batch",
            "contact_schur_rows_by_batch",
            "contact_schur_true_residual_norm_by_batch",
            "contact_schur_linear_converged_by_batch",
        )
        if self.linear_solver == "contact_schur":
            if self.contact_solve_executed and any(len(getattr(self, name)) != batch_size for name in schur_fields):
                raise ValueError("contact-Schur SAP health must cover every solved batch")
            if not self.contact_solve_executed and any(getattr(self, name) for name in schur_fields):
                raise ValueError("unexecuted contact-Schur SAP health must be empty")
        elif any(getattr(self, name) for name in schur_fields):
            raise ValueError("non-contact-Schur SAP health cannot carry contact-Schur fields")
        if any(
            type(value) is not bool
            for name in (
                "sap_active_by_batch",
                "pcg_active_by_batch",
                "linesearch_active_by_batch",
                "pcg_budget_exhausted_by_batch",
                "linesearch_budget_exhausted_by_batch",
            )
            for value in getattr(self, name)
        ):
            raise TypeError("per-batch active fields must contain bool values")
        for name in (
            "gradient_norm_by_batch",
            "momentum_norm_by_batch",
            "impulse_norm_by_batch",
            "pcg_residual_squared_by_batch",
            "pcg_preconditioned_residual_by_batch",
        ):
            for value in getattr(self, name):
                _float_value(value, name)
        for name in ("sap_convergence_atol", "sap_convergence_rtol", "pcg_threshold"):
            _finite_nonnegative(getattr(self, name), name)
        for value in self.sparse_direct_sap_iterations_by_batch:
            _nonnegative_int(value, "sparse_direct_sap_iterations_by_batch value")
        for value in self.sparse_direct_final_original_momentum_norm_by_batch:
            _finite_nonnegative(value, "sparse_direct_final_original_momentum_norm_by_batch value")
        if any(type(value) is not bool for value in self.sparse_direct_converged_by_batch):
            raise TypeError("sparse_direct_converged_by_batch must contain bool values")
        for name in (
            "sparse_direct_assembly_time_s",
            "sparse_direct_factor_time_s",
            "sparse_direct_solve_time_s",
            "sparse_direct_transfer_time_s",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in (
            "enriched_pcg_sap_iterations_by_batch",
            "enriched_pcg_geometry_columns_by_batch",
            "enriched_pcg_response_columns_by_batch",
            "enriched_pcg_retained_columns_by_batch",
            "enriched_pcg_iterations_by_batch",
        ):
            for value in getattr(self, name):
                _nonnegative_int(value, f"{name} value")
        for name in (
            "enriched_pcg_final_original_momentum_norm_by_batch",
            "enriched_pcg_true_residual_norm_by_batch",
        ):
            for value in getattr(self, name):
                _finite_nonnegative(value, f"{name} value")
        if any(type(value) is not bool for value in self.enriched_pcg_sap_converged_by_batch):
            raise TypeError("enriched_pcg_sap_converged_by_batch must contain bool values")
        if any(type(value) is not bool for value in self.enriched_pcg_linear_converged_by_batch):
            raise TypeError("enriched_pcg_linear_converged_by_batch must contain bool values")
        for name in (
            "enriched_pcg_setup_time_s",
            "enriched_pcg_solve_time_s",
            "enriched_pcg_transfer_time_s",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in ("contact_schur_sap_iterations_by_batch", "contact_schur_rows_by_batch"):
            for value in getattr(self, name):
                _nonnegative_int(value, f"{name} value")
        for name in (
            "contact_schur_final_original_momentum_norm_by_batch",
            "contact_schur_true_residual_norm_by_batch",
        ):
            for value in getattr(self, name):
                _finite_nonnegative(value, f"{name} value")
        for name in ("contact_schur_sap_converged_by_batch", "contact_schur_linear_converged_by_batch"):
            if any(type(value) is not bool for value in getattr(self, name)):
                raise TypeError(f"{name} must contain bool values")
        for name in ("contact_schur_setup_time_s", "contact_schur_solve_time_s", "contact_schur_transfer_time_s"):
            _finite_nonnegative(getattr(self, name), name)
        if self.implicit_fem is not None and not isinstance(self.implicit_fem, ImplicitFEMSubstepHealth):
            raise TypeError("implicit_fem must be ImplicitFEMSubstepHealth or None")
        if self.fem_safety_extrema is not None and not isinstance(self.fem_safety_extrema, FEMSubstepSafetyExtrema):
            raise TypeError("fem_safety_extrema must be FEMSubstepSafetyExtrema or None")
        if self.fem_safety_extrema is not None and self.fem_safety_extrema.principal_strain_witness is not None:
            if self.fem_safety_extrema.principal_strain_witness.global_substep_index != self.global_substep_index:
                raise ValueError("principal-strain witness must belong to its owning global substep")
        if self.positive_j_feasible_step is not None:
            if not isinstance(self.positive_j_feasible_step, PositiveJFeasibleStep):
                raise TypeError("positive_j_feasible_step must be PositiveJFeasibleStep or None")
            if self.contact_solve_executed and len(self.positive_j_feasible_step.pre_sap_min_j) != batch_size:
                raise ValueError("positive-J feasible-step evidence must cover every batch")
        if self.implicit_fem_positive_j_feasible_step is not None:
            if not isinstance(self.implicit_fem_positive_j_feasible_step, ImplicitFEMPositiveJFeasibleStep):
                raise TypeError(
                    "implicit_fem_positive_j_feasible_step must be ImplicitFEMPositiveJFeasibleStep or None"
                )
            if (
                self.contact_solve_executed
                and len(self.implicit_fem_positive_j_feasible_step.pre_update_base_min_j) != batch_size
            ):
                raise ValueError("implicit FEM positive-J evidence must cover every batch")
        _nonnegative_int(self.rigid_fem_contact_pair_count, "rigid_fem_contact_pair_count")
        if self.max_rigid_fem_penetration_m is not None:
            _float_value(self.max_rigid_fem_penetration_m, "max_rigid_fem_penetration_m")
        if not self.rigid_fem_contact_supported:
            if self.rigid_fem_contact_pair_count != 0 or self.max_rigid_fem_penetration_m is not None:
                raise ValueError("unsupported rigid--FEM telemetry cannot report contact values")
        elif self.max_rigid_fem_penetration_m is None:
            raise ValueError("supported rigid--FEM telemetry must report a penetration extrema")
        if tuple(sorted(set(self.unwhitelisted_rigid_fem_link_names))) != self.unwhitelisted_rigid_fem_link_names:
            raise ValueError("unwhitelisted_rigid_fem_link_names must be sorted and unique")
        if type(self.post_final_sap_health_available) is not bool:
            raise TypeError("post_final_sap_health_available must be bool")
        post_final_fields = (
            "post_final_sap_active_by_batch",
            "post_final_gradient_norm_by_batch",
            "post_final_momentum_norm_by_batch",
            "post_final_impulse_norm_by_batch",
        )
        if self.post_final_sap_health_available:
            if not self.contact_solve_executed or any(
                len(getattr(self, name)) != batch_size for name in post_final_fields
            ):
                raise ValueError("available post-final SAP health must cover every solved batch")
            if any(type(value) is not bool for value in self.post_final_sap_active_by_batch):
                raise TypeError("post_final_sap_active_by_batch must contain bool values")
            for name in post_final_fields[1:]:
                for value in getattr(self, name):
                    _float_value(value, name)
        elif any(getattr(self, name) for name in post_final_fields):
            raise ValueError("unavailable post-final SAP health must be empty")
        if type(self.rigid_fem_contact_patch_preconditioner_enabled) is not bool:
            raise TypeError("rigid_fem_contact_patch_preconditioner_enabled must be bool")
        patch_fields = (
            "rigid_fem_contact_patch_min_active_count_by_batch",
            "rigid_fem_contact_patch_min_rank_by_batch",
            "rigid_fem_contact_patch_max_rank_by_batch",
            "rigid_fem_contact_patch_all_coarse_finite_by_batch",
        )
        if self.rigid_fem_contact_patch_preconditioner_enabled and self.contact_solve_executed:
            if any(len(getattr(self, name)) != batch_size for name in patch_fields):
                raise ValueError("enabled rigid--FEM contact-patch health must cover every solved batch")
            for name in patch_fields[:3]:
                for value in getattr(self, name):
                    _nonnegative_int(value, f"{name} value")
            if any(
                value > 6
                for name in patch_fields[1:3]
                for value in getattr(self, name)
            ):
                raise ValueError("rigid--FEM contact-patch ranks must lie in [0, 6]")
            if any(
                minimum > maximum
                for minimum, maximum in zip(
                    self.rigid_fem_contact_patch_min_rank_by_batch,
                    self.rigid_fem_contact_patch_max_rank_by_batch,
                )
            ):
                raise ValueError("rigid--FEM contact-patch minimum rank cannot exceed maximum rank")
            if any(
                type(value) is not bool
                for value in self.rigid_fem_contact_patch_all_coarse_finite_by_batch
            ):
                raise TypeError("rigid_fem_contact_patch_all_coarse_finite_by_batch must contain bool values")
        elif any(len(getattr(self, name)) != 0 for name in patch_fields):
            raise ValueError("unexecuted or disabled rigid--FEM contact-patch health must be empty")
        if type(self.rigid_fem_contact_tet_schwarz_preconditioner_enabled) is not bool:
            raise TypeError("rigid_fem_contact_tet_schwarz_preconditioner_enabled must be bool")
        schwarz_fields = (
            "rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch",
            "rigid_fem_contact_tet_schwarz_max_vertex_overlap_by_batch",
            "rigid_fem_contact_tet_schwarz_max_link_overlap_by_batch",
            "rigid_fem_contact_tet_schwarz_max_link_rank_by_batch",
            "rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch",
            "rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch",
        )
        if self.rigid_fem_contact_tet_schwarz_preconditioner_enabled and self.contact_solve_executed:
            if any(len(getattr(self, name)) != batch_size for name in schwarz_fields):
                raise ValueError("enabled rigid--FEM contact-tet Schwarz health must cover every solved batch")
            for name in schwarz_fields[:4]:
                for value in getattr(self, name):
                    _nonnegative_int(value, f"{name} value")
            if any(value > 6 for value in self.rigid_fem_contact_tet_schwarz_max_link_rank_by_batch):
                raise ValueError("rigid--FEM contact-tet Schwarz link ranks must lie in [0, 6]")
            for value in self.rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch:
                _finite_nonnegative(value, "rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch value")
            if any(
                active > 0 and (vertex_overlap < 1 or link_overlap < 1)
                for active, vertex_overlap, link_overlap in zip(
                    self.rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch,
                    self.rigid_fem_contact_tet_schwarz_max_vertex_overlap_by_batch,
                    self.rigid_fem_contact_tet_schwarz_max_link_overlap_by_batch,
                )
            ):
                raise ValueError("positive Schwarz active-block count requires positive vertex and link overlap")
            if any(
                type(value) is not bool
                for value in self.rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch
            ):
                raise TypeError("rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch must contain bool values")
            if any(
                active > 0 and valid and pivot <= 0.0
                for active, valid, pivot in zip(
                    self.rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch,
                    self.rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch,
                    self.rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch,
                )
            ):
                raise ValueError("valid active Schwarz factors require a positive minimum pivot")
        elif any(len(getattr(self, name)) != 0 for name in schwarz_fields):
            raise ValueError("unexecuted or disabled rigid--FEM contact-tet Schwarz health must be empty")

    @property
    def solver_converged(self) -> bool:
        """Whether every real SAP iterative mask was inactive after its budget."""
        if not self.contact_solve_executed:
            return True
        if self.linear_solver == "sparse_direct":
            return all(self.sparse_direct_converged_by_batch)
        if self.linear_solver == "enriched_pcg":
            return all(self.enriched_pcg_sap_converged_by_batch) and all(
                self.enriched_pcg_linear_converged_by_batch
            )
        if self.linear_solver == "contact_schur":
            return all(self.contact_schur_sap_converged_by_batch) and all(self.contact_schur_linear_converged_by_batch)
        return not any(
            self.sap_active_by_batch
            + self.pcg_active_by_batch
            + self.linesearch_active_by_batch
            + self.pcg_budget_exhausted_by_batch
            + self.linesearch_budget_exhausted_by_batch
        )

    @property
    def finite(self) -> bool:
        """Whether the exported solver residual and contact extrema are usable."""
        values = (
            *self.gradient_norm_by_batch,
            *self.momentum_norm_by_batch,
            *self.impulse_norm_by_batch,
            *self.pcg_residual_squared_by_batch,
            *self.pcg_preconditioned_residual_by_batch,
            *self.sparse_direct_final_original_momentum_norm_by_batch,
            self.sparse_direct_assembly_time_s,
            self.sparse_direct_factor_time_s,
            self.sparse_direct_solve_time_s,
            self.sparse_direct_transfer_time_s,
            *self.enriched_pcg_final_original_momentum_norm_by_batch,
            *self.enriched_pcg_true_residual_norm_by_batch,
            self.enriched_pcg_setup_time_s,
            self.enriched_pcg_solve_time_s,
            self.enriched_pcg_transfer_time_s,
            *self.contact_schur_final_original_momentum_norm_by_batch,
            *self.contact_schur_true_residual_norm_by_batch,
            self.contact_schur_setup_time_s,
            self.contact_schur_solve_time_s,
            self.contact_schur_transfer_time_s,
        )
        if self.max_rigid_fem_penetration_m is not None:
            values = (*values, self.max_rigid_fem_penetration_m)
        if not all(isfinite(value) and value >= 0.0 for value in values):
            return False
        if self.rigid_fem_contact_patch_preconditioner_enabled and not all(
            self.rigid_fem_contact_patch_all_coarse_finite_by_batch
        ):
            return False
        if self.rigid_fem_contact_tet_schwarz_preconditioner_enabled and not all(
            self.rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch
        ):
            return False
        if self.rigid_fem_contact_tet_schwarz_preconditioner_enabled and not all(
            isfinite(value) and value >= 0.0
            for value in self.rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch
        ):
            return False
        if self.positive_j_feasible_step is None:
            return True
        return all(
            isfinite(value)
            for value in (
                *self.positive_j_feasible_step.pre_sap_min_j,
                *self.positive_j_feasible_step.unfiltered_post_sap_trial_min_j,
                *self.positive_j_feasible_step.accepted_alpha,
            )
        )

    @property
    def healthy(self) -> bool:
        return (
            self.solver_converged
            and self.finite
            and self.implicit_fem is not None
            and self.implicit_fem.healthy
            and self.fem_safety_extrema is not None
            and not self.contact_overflow
            and not self.unknown_rigid_fem_contact_mode
            and not self.unwhitelisted_rigid_fem_link_names
        )

    @property
    def post_final_sap_converged(self) -> bool:
        """Qualification-only completed-state SAP verdict, when available."""
        return self.post_final_sap_health_available and not any(self.post_final_sap_active_by_batch)


@dataclass(frozen=True, slots=True)
class SAPControlStepSolverHealth:
    """All completed physical substeps for one successful public ``Scene.step``."""

    scene_step_index: int
    first_global_substep_index: int
    last_global_substep_index: int
    substeps: tuple[SAPSubstepSolverHealth, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.scene_step_index, "scene_step_index")
        if self.scene_step_index == 0:
            raise ValueError("scene_step_index must be the post-increment positive Scene step index")
        _nonnegative_int(self.first_global_substep_index, "first_global_substep_index")
        _nonnegative_int(self.last_global_substep_index, "last_global_substep_index")
        if not self.substeps or not all(isinstance(item, SAPSubstepSolverHealth) for item in self.substeps):
            raise TypeError("substeps must be a nonempty tuple of SAPSubstepSolverHealth")
        indices = tuple(item.global_substep_index for item in self.substeps)
        if indices != tuple(range(self.first_global_substep_index, self.last_global_substep_index + 1)):
            raise ValueError("substeps must exactly cover the completed global-substep interval")
        if any(item.sim_step_index != self.substeps[0].sim_step_index for item in self.substeps):
            raise ValueError("substeps must belong to one simulator control step")
        if self.substeps[0].sim_step_index != self.scene_step_index - 1:
            raise ValueError("scene_step_index must be exactly one greater than the zero-based simulator step index")

    @property
    def all_substeps_healthy(self) -> bool:
        return all(item.healthy for item in self.substeps)

    @property
    def first_failed_global_substep_index(self) -> int | None:
        return next((item.global_substep_index for item in self.substeps if not item.healthy), None)

    @property
    def max_rigid_fem_penetration_m(self) -> float | None:
        extrema = tuple(item.max_rigid_fem_penetration_m for item in self.substeps)
        if any(value is None for value in extrema):
            return None
        return max(float(value) for value in extrema)

    @property
    def contact_overflow(self) -> bool:
        return any(item.contact_overflow for item in self.substeps)

    @property
    def unknown_rigid_fem_contact_mode(self) -> bool:
        return any(item.unknown_rigid_fem_contact_mode for item in self.substeps)

    @property
    def unwhitelisted_rigid_fem_link_names(self) -> tuple[str, ...]:
        return tuple(sorted({name for item in self.substeps for name in item.unwhitelisted_rigid_fem_link_names}))

    @property
    def fem_safety_extrema(self) -> FEMSubstepSafetyExtrema | None:
        """Conservative reduction over every physical substep in this control step."""
        extrema = tuple(item.fem_safety_extrema for item in self.substeps)
        if any(item is None for item in extrema):
            return None
        values = tuple(item for item in extrema if item is not None)
        max_strain = max(item.max_principal_stretch_strain for item in values)
        witness_candidates = tuple(
            (substep.global_substep_index, item.principal_strain_witness)
            for substep, item in zip(self.substeps, values)
            if item.principal_strain_witness is not None
            and item.principal_strain_witness.principal_stretch_strain == max_strain
        )
        witness = min(witness_candidates, key=lambda candidate: candidate[0])[1] if witness_candidates else None
        return FEMSubstepSafetyExtrema(
            min_j=min(item.min_j for item in values),
            max_principal_stretch_strain=max_strain,
            max_tet_elastic_energy_j=max(item.max_tet_elastic_energy_j for item in values),
            total_elastic_energy_j=max(item.total_elastic_energy_j for item in values),
            no_inversion=all(item.no_inversion for item in values),
            principal_strain_witness=witness,
        )
