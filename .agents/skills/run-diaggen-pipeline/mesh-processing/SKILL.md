---
name: mesh-processing
description: Choose per-part mesh fidelities and target metric size for HAG4R volumetric-only filled/hollow meshing.
---

# Mesh Processing

Use this skill when the mesh processing agent converts OmniPart parts and Feature-1 material predictions into final HAG4R volumetric simulation artifacts.

## Objective

Choose one `low | medium | high` mesh fidelity for every color-locked part and estimate one object-level `target_max_dimension_m`. Fidelity describes how much geometric detail to retain; it does not control execution or scheduling order. The runtime then deterministically builds a metric monolithic tet mesh using per-part `solid_fill | hollow_wall` topology and a checked Manifold Boolean union of the reconstructed watertight part volumes. The agent does not author numeric mesh parameters, target volume, wall thickness, or simulator representation.

## Input

- OmniPart output directory.
- `part_labels.npz`.
- Feature-1 `inferred_params.json` with exact `part_index`, `part_color_rgb`, `volume_fill_mode`, density, Young's modulus, Poisson ratio, and friction.
- Object description, semantic context, visual evidence, and mesh-processing diagnostic cues.
- When this stage is the designated diagnostic repair destination, a structured diagnostic repair brief in the user message.

## Expected Output

- Per-part material table from `run_assign_params_to_prims_stage`.
- Metric monolithic tetrahedral mesh.
- Owner-derived tet-only heterogeneous parameters.
- `metric_mesh_scaling.json`.
- `volume_topology.json`.
- Advisory tet budget status; over `40,000` tets is warning-only and does not fail, reroute, or retry.
- Immediate code-owned hard validation of Genesis orientation and determinant compatibility, zero non-manifold tet faces, and exactly one face-connected tetrahedral material component before material assignment.
- Exact material-label coverage before success artifacts are advertised on the normal single-component path.
- If tetrahedralization yields multiple face-connected tetrahedral material components, a retryable `needs_component_resolution` state plus a code-owned component-resolution candidate/report. The resolver deterministically keeps the maximum-volume component, allows final tet labels to be a subset of the original part domain, and re-runs all hard validation before success artifacts are advertised.
- Diagnostic `boundary_surface_component_count`; multiple cavity boundaries are allowed and do not relax the one-component material-domain invariant.

## Available Tools

- `run_assign_params_to_prims_stage`
- `select_mesh_processing_fidelities_stage`
- `run_combined_to_monolithic_stage`
- `resolve_disconnected_tet_components_stage` only after `run_combined_to_monolithic_stage` returns `needs_component_resolution`

## Fidelity Table

| fidelity | use when | compression | keep ratio | absolute cap | voxel resolution |
|---|---|---|---:|---:|---:|
| `high` | Thin/hollow, contact-critical, small-feature, or diagnostics-detail-sensitive part. | aggressive | 0.02 | 10,000 faces | longest axis 96 voxels |
| `medium` | Ordinary structure or visible but non-contact-critical part. | super-aggressive | 0.005 | 5,000 faces | longest axis 96 voxels |
| `low` | Simple bulky, low-visible, low-interaction, or safely extreme-compression part. | extreme-aggressive | 0.001 | 2,000 faces | longest axis 96 voxels |

The runtime computes target faces using:

```text
target_faces =
    min(
        input_faces,
        max(32, min(ceil(input_faces * keep_ratio), absolute_cap))
    )
```

All fidelity tiers use the same longest-axis voxel resolution of 96. Fidelity changes only the post-voxelization surface face budget. The agent must not pass `target_faces`, `keep_ratio`, `absolute_cap`, voxel resolution, voxel pitch, edge length, or any other numeric mesh knob.

## Workflow

1. Read material predictions and their `volume_fill_mode` values.
2. For every exact indexed part, choose `mesh_fidelity` and write a concise `fidelity_rationale`.
3. Estimate exactly one positive `target_max_dimension_m` with a concise `estimate_rationale`.
4. Call `select_mesh_processing_fidelities_stage` with exact `part_index` and `part_color_rgb` coverage.
5. Run `run_assign_params_to_prims_stage`.
6. Run `run_combined_to_monolithic_stage`.
7. If the result status is `needs_component_resolution`, read the component-resolution report, then call `resolve_disconnected_tet_components_stage` exactly once with a concise rationale. Do not choose or suggest a component ID; deterministic code keeps the maximum `volume_m3` component, breaking ties by smallest component ID.
9. Do not call unscaled-volume or volume-based scaling stages.
10. Do not request wall thickness. `hollow_wall_band_layers = 2` is fixed runtime configuration. Derived band width is provenance only.
11. If a diagnostic repair brief is present, summarize how it affected per-part fidelities or `target_max_dimension_m` only.

## Failure Cases

- Using representation or old asset-level fidelity language.
- Passing a target material-volume value.
- Passing agent-authored numeric mesh flags.
- Changing `hollow_wall_band_layers = 2`.
- Treating the `40,000` tet advisory budget as a failure.
- Disabling or bypassing the code-owned Genesis-compatibility, non-manifold-face, or material-domain gates.
- Agent-authored deletion of disconnected components, agent choice of a component ID, silently bridging gaps, or falling back from a failed checked Boolean union to concatenation. Use only `resolve_disconnected_tet_components_stage` after the runtime reports `needs_component_resolution`.
- Treating multiple boundary-surface components as proof of multiple material bodies.
- Relaxing the monolithic contract for intentional independent bodies; those require an explicit multi-body contract.
- Adding agent-authored geometry repairs, retries, or hard gates for cavity emptiness, center fill, tet leakage, band width, or general tet-quality heuristics.
- Calling final-mesh proximity transfer instead of using owner-derived `tet_part_labels`.
