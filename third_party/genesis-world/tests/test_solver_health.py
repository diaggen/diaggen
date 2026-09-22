from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import genesis as gs
from genesis.engine.couplers.sap_coupler import SAPCoupler


class _Field:
    def __init__(self, value):
        self._value = np.asarray(value)

    def to_numpy(self):
        return self._value.copy()


def _extrema(
    *,
    min_j: float = 1.0,
    strain: float = 0.0,
    max_energy: float = 0.0,
    total_energy: float = 0.0,
    global_substep_index: int = 0,
):
    witness = gs.FEMPrincipalStrainWitness(
        global_substep_index=global_substep_index,
        env_index=0,
        fem_entity_index=0,
        fem_entity_name="clawhauser",
        tet_local_index=0,
        tet_global_index=0,
        tet_entity_local_vertex_indices=(0, 1, 2, 3),
        tet_global_vertex_indices=(0, 1, 2, 3),
        current_vertex_positions_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.2), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        current_centroid_m=(0.25, 0.25, 0.05),
        principal_stretch_strain=float(strain),
        floor_height_m=0.0,
        floor_tet_candidate_geometrically_possible=True,
    )
    return gs.FEMSubstepSafetyExtrema(
        min_j=min_j,
        max_principal_stretch_strain=strain,
        max_tet_elastic_energy_j=max_energy,
        total_elastic_energy_j=total_energy,
        no_inversion=min_j > 0.0,
        principal_strain_witness=witness,
    )


def _substep(
    index: int,
    *,
    sim_step: int = 4,
    active: bool = False,
    penetration: float = 0.0,
    extrema: gs.FEMSubstepSafetyExtrema | None = None,
):
    if extrema is None:
        extrema = _extrema(global_substep_index=index)
    elif extrema.principal_strain_witness is not None:
        extrema = replace(
            extrema,
            principal_strain_witness=replace(extrema.principal_strain_witness, global_substep_index=index),
        )
    return gs.SAPSubstepSolverHealth(
        global_substep_index=index,
        sim_step_index=sim_step,
        physical_dt_s=0.002,
        contact_solve_executed=True,
        sap_iteration_budget=3,
        pcg_iteration_budget=5,
        linesearch_iteration_budget=7,
        sap_active_by_batch=(active,),
        pcg_active_by_batch=(False,),
        linesearch_active_by_batch=(False,),
        pcg_budget_exhausted_by_batch=(False,),
        linesearch_budget_exhausted_by_batch=(False,),
        gradient_norm_by_batch=(1.0,),
        momentum_norm_by_batch=(2.0,),
        impulse_norm_by_batch=(3.0,),
        pcg_residual_squared_by_batch=(4.0,),
        pcg_preconditioned_residual_by_batch=(5.0,),
        sap_convergence_atol=1.0e-6,
        sap_convergence_rtol=1.0e-4,
        pcg_threshold=1.0e-8,
        implicit_fem=gs.ImplicitFEMSubstepHealth(
            batch_active_by_batch=(True,),
            pcg_active_by_batch=(False,),
            linesearch_active_by_batch=(False,),
            pcg_budget_exhausted_by_batch=(False,),
            pcg_breakdown_by_batch=(False,),
            linesearch_budget_exhausted_by_batch=(False,),
            pcg_initial_residual_squared_by_batch=(16.0,),
            pcg_residual_squared_by_batch=(4.0,),
            pcg_preconditioned_residual_by_batch=(5.0,),
            pcg_relative_residual_norm_by_batch=(0.5,),
            pcg_effective_residual_squared_threshold_by_batch=(1.0e-8,),
            max_newton_update_m_by_batch=(1.0e-7,),
            newton_iteration_budget=2,
            pcg_iteration_budget=5,
            linesearch_iteration_budget=3,
            newton_dx_threshold_m=1.0e-6,
            pcg_threshold=1.0e-8,
            pcg_absolute_residual_squared_floor=1.0e-8,
            pcg_rtol=0.0,
        ),
        rigid_fem_contact_supported=True,
        rigid_fem_contact_pair_count=1,
        max_rigid_fem_penetration_m=penetration,
        contact_overflow=False,
        unknown_rigid_fem_contact_mode=False,
        unwhitelisted_rigid_fem_link_names=(),
        fem_safety_extrema=extrema,
    )


def test_control_step_health_keeps_early_substep_nonconvergence_and_worst_penetration():
    status = gs.SAPControlStepSolverHealth(
        scene_step_index=5,
        first_global_substep_index=12,
        last_global_substep_index=14,
        substeps=(
            _substep(
                12,
                active=True,
                penetration=0.003,
                extrema=_extrema(min_j=0.84, strain=0.07, max_energy=0.3, total_energy=1.2),
            ),
            _substep(
                13, penetration=0.001, extrema=_extrema(min_j=0.96, strain=0.02, max_energy=0.1, total_energy=0.8)
            ),
            _substep(14, extrema=_extrema(min_j=0.99, strain=0.01, max_energy=0.2, total_energy=0.9)),
        ),
    )
    assert not status.all_substeps_healthy
    assert status.first_failed_global_substep_index == 12
    assert status.max_rigid_fem_penetration_m == pytest.approx(0.003)
    assert status.fem_safety_extrema == _extrema(
        min_j=0.84, strain=0.07, max_energy=0.3, total_energy=1.2, global_substep_index=12
    )


def test_completed_fem_extrema_cross_entity_argmax_uses_authoritative_entity_idx():
    from genesis.engine.solvers.fem_solver import FEMSolver

    rest_one = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    rest = np.concatenate((rest_one, rest_one), axis=0)
    tets = np.asarray(((0, 1, 2, 3), (0, 1, 2, 3)), dtype=np.int64)
    current = rest.copy()
    current[1, 0] = 1.2
    current[5, 0] = 1.2

    def _entity(*, idx: int, v_start: int, el_start: int):
        return SimpleNamespace(
            idx=idx,
            name=f"entity-{idx}",
            v_start=v_start,
            el_start=el_start,
            n_vertices=4,
            n_elements=1,
            elems=np.asarray(((0, 1, 2, 3),), dtype=np.int64),
            init_positions=rest_one.copy(),
            _heterogeneous_material_np=None,
            material=SimpleNamespace(model="linear_corotated", mu=2.0, lam=3.0),
        )

    state = {"positions": current}
    stub = SimpleNamespace(
        is_active=True,
        _use_implicit_solver=True,
        _B=1,
        sim=SimpleNamespace(substeps_local=1),
        n_vertices=8,
        n_elements=2,
        _floor_height=0.0,
        _entities=(_entity(idx=9, v_start=0, el_start=0), _entity(idx=3, v_start=4, el_start=1)),
    )
    stub.get_state = lambda _frame: SimpleNamespace(
        pos=state["positions"][None], active=np.asarray(((True, True),), dtype=np.bool_)
    )
    equal = FEMSolver.get_completed_substep_safety_extrema(stub, completed_frame=1, global_substep_index=11)
    assert equal is not None and equal.principal_strain_witness is not None
    assert equal.principal_strain_witness.fem_entity_index == 3

    state["positions"] = current.copy()
    state["positions"][1, 0] = 1.3
    larger = FEMSolver.get_completed_substep_safety_extrema(stub, completed_frame=1, global_substep_index=11)
    assert larger is not None and larger.principal_strain_witness is not None
    assert larger.principal_strain_witness.fem_entity_index == 9
    assert larger.max_principal_stretch_strain > equal.max_principal_stretch_strain


def test_sap_capture_reads_actual_masks_and_nonfinite_residual_fail_closed():
    coupler = object.__new__(SAPCoupler)
    coupler.sim = SimpleNamespace(_B=1, cur_substep_global=8, cur_step_global=3, _substep_dt=0.002)
    coupler.has_contact = True
    coupler.batch_active = _Field([False])
    coupler.batch_pcg_active = _Field([True])
    coupler.batch_linesearch_active = _Field([False])
    coupler.batch_pcg_budget_exhausted = _Field([True])
    coupler.batch_linesearch_budget_exhausted = _Field([False])
    coupler.sap_state = SimpleNamespace(
        gradient_norm=_Field([1.0]), momentum_norm=_Field([2.0]), impulse_norm=_Field([3.0])
    )
    coupler.pcg_state = SimpleNamespace(rTr=_Field([np.nan]), rTz=_Field([4.0]))
    coupler._n_sap_iterations = 3
    coupler._n_pcg_iterations = 5
    coupler._n_linesearch_iterations = 7
    coupler._sap_convergence_atol = 1.0e-6
    coupler._sap_convergence_rtol = 1.0e-4
    coupler._pcg_threshold = 1.0e-8
    coupler._enable_rigid_fem_contact = False
    coupler._enable_development_positive_j_feasible_step = False
    coupler._enable_development_positive_j_alpha_one_only = False
    coupler.fem_solver = SimpleNamespace(
        is_active=False,
        _use_implicit_solver=False,
        _development_implicit_fem_positive_j_feasible_step_health=lambda: None,
    )
    coupler._last_contact_overflow = False
    status = coupler._capture_completed_solver_health()
    assert status.pcg_active_by_batch == (True,)
    assert np.isnan(status.pcg_residual_squared_by_batch[0])
    assert not status.finite
    assert not status.healthy
    assert not status.post_final_sap_health_available
    assert not status.post_final_sap_converged


def test_sap_capture_retains_early_inner_budget_exhaustion_after_later_success():
    coupler = object.__new__(SAPCoupler)
    coupler.sim = SimpleNamespace(_B=1, cur_substep_global=8, cur_step_global=3, _substep_dt=0.002)
    coupler.has_contact = True
    coupler.batch_active = _Field([False])
    coupler.batch_pcg_active = _Field([False])
    coupler.batch_linesearch_active = _Field([False])
    coupler.batch_pcg_budget_exhausted = _Field([True])
    coupler.batch_linesearch_budget_exhausted = _Field([False])
    coupler.sap_state = SimpleNamespace(
        gradient_norm=_Field([1.0]), momentum_norm=_Field([2.0]), impulse_norm=_Field([3.0])
    )
    coupler.pcg_state = SimpleNamespace(rTr=_Field([4.0]), rTz=_Field([5.0]))
    coupler._n_sap_iterations = 3
    coupler._n_pcg_iterations = 5
    coupler._n_linesearch_iterations = 7
    coupler._sap_convergence_atol = 1.0e-6
    coupler._sap_convergence_rtol = 1.0e-4
    coupler._pcg_threshold = 1.0e-8
    coupler._enable_rigid_fem_contact = False
    coupler._enable_development_positive_j_feasible_step = False
    coupler._enable_development_positive_j_alpha_one_only = False
    coupler.fem_solver = SimpleNamespace(
        is_active=False,
        _use_implicit_solver=False,
        _development_implicit_fem_positive_j_feasible_step_health=lambda: None,
    )
    coupler._last_contact_overflow = False
    status = coupler._capture_completed_solver_health()
    assert status.pcg_active_by_batch == (False,)
    assert status.pcg_budget_exhausted_by_batch == (True,)
    assert not status.solver_converged


def test_implicit_fem_health_exports_final_inner_relative_pcg_detail_and_accumulated_breakdown():
    coupler = object.__new__(SAPCoupler)
    coupler.sim = SimpleNamespace(_B=1)
    coupler.fem_solver = SimpleNamespace(
        is_active=True,
        _use_implicit_solver=True,
        batch_active=_Field([True]),
        batch_pcg_active=_Field([False]),
        batch_linesearch_active=_Field([False]),
        batch_pcg_budget_exhausted=_Field([True]),
        batch_pcg_breakdown=_Field([False]),
        batch_linesearch_budget_exhausted=_Field([False]),
        pcg_state=SimpleNamespace(
            rTr_initial=_Field([100.0]),
            rTr=_Field([1.0e-8]),
            rTz=_Field([2.0e-9]),
            termination_threshold=_Field([1.0e-8]),
        ),
        pcg_state_v=SimpleNamespace(x=_Field([[[0.0, 0.0, 0.0]]])),
        n_vertices=1,
        _n_newton_iterations=2,
        _n_pcg_iterations=500,
        _n_linesearch_iterations=0,
        _newton_dx_threshold=1.0e-6,
        _pcg_threshold=1.0e-10,
        _pcg_rtol=1.0e-5,
    )
    health = coupler._implicit_fem_health_state()
    assert health is not None
    assert health.pcg_initial_residual_squared_by_batch == (100.0,)
    assert health.pcg_residual_squared_by_batch == (1.0e-8,)
    assert health.pcg_relative_residual_norm_by_batch == pytest.approx((1.0e-5,))
    assert health.pcg_effective_residual_squared_threshold_by_batch == (1.0e-8,)
    assert health.pcg_absolute_residual_squared_floor == pytest.approx(1.0e-10)
    assert health.pcg_rtol == pytest.approx(1.0e-5)
    assert health.pcg_budget_exhausted_by_batch == (True,)
    assert not health.healthy
    assert not replace(health, pcg_breakdown_by_batch=(True,)).healthy


def test_scene_public_query_rejects_missing_or_stale_control_step_status():
    scene = object.__new__(gs.Scene)
    scene._is_built = True
    scene._t = 5
    scene._last_completed_solver_health = None
    with pytest.raises(gs.SolverHealthNotReadyError):
        scene.get_last_completed_solver_health()


def test_scene_step_exception_invalidates_prior_successful_health_record():
    class _FailingSimulator:
        def step(self):
            raise RuntimeError("solver failed after prior success")

        def destroy(self):
            pass

    scene = object.__new__(gs.Scene)
    scene._is_built = True
    scene._t = 5
    scene._forward_ready = True
    scene._pre_step_callbacks = []
    scene._sim = _FailingSimulator()
    scene._last_completed_solver_health = gs.SAPControlStepSolverHealth(5, 0, 0, (_substep(0),))
    with pytest.raises(RuntimeError, match="solver failed"):
        scene.step(update_visualizer=False)
    with pytest.raises(gs.SolverHealthNotReadyError):
        scene.get_last_completed_solver_health()
    scene._last_completed_solver_health = gs.SAPControlStepSolverHealth(4, 0, 0, (_substep(0, sim_step=3),))
    with pytest.raises(gs.SolverHealthNotReadyError):
        scene.get_last_completed_solver_health()


def test_simulator_partial_substep_failure_clears_the_publish_slot():
    initialized_here = not gs._initialized
    if initialized_here:
        gs.init(backend=gs.cpu, precision="64", logging_level="error")
    from genesis.engine.simulator import Simulator

    simulator = object.__new__(Simulator)
    coupler = object.__new__(SAPCoupler)
    coupler.get_last_completed_solver_health = lambda: _substep(simulator._cur_substep_global, sim_step=0)
    finalized_extrema = []
    coupler.finalize_completed_solver_health = lambda *, fem_safety_extrema: finalized_extrema.append(
        fem_safety_extrema
    )
    simulator._coupler = coupler
    simulator._last_completed_sap_substeps = (_substep(0),)
    simulator._cur_substep_global = 0
    simulator._substeps = 2
    simulator._substeps_local = 2
    simulator._rigid_only = False
    simulator._requires_grad = False
    simulator._completed_solver_health_enabled = True
    simulator._completed_physical_substep_geometry_trace_enabled = False
    simulator.rigid_solver = SimpleNamespace(is_active=False)
    simulator.fem_solver = SimpleNamespace(
        _enable_qualification_safety_extrema=True,
        get_completed_substep_safety_extrema=lambda **kwargs: _extrema(
            global_substep_index=kwargs.get("global_substep_index", 0)
        ),
    )
    simulator.process_input = lambda **_kwargs: None
    substep_calls = []

    def _advance_substep(_local):
        substep_calls.append(_local)
        if len(substep_calls) == 2:
            raise RuntimeError("second physical substep failed")

    simulator.substep = _advance_substep
    try:
        with pytest.raises(RuntimeError, match="second physical"):
            simulator.step()
        assert substep_calls == [0, 1]
        assert finalized_extrema == [_extrema()]
        assert simulator._last_completed_sap_substeps is None
    finally:
        if initialized_here:
            gs.destroy()
