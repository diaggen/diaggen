---
name: final-export
description: Export the completed HAG4R asset bundle, manifest, cleanup artifacts, and diagnostics artifacts without changing upstream stage outputs.
---

# Final Export

Use this skill when the final export stage packages a completed HAG4R run into the downstream asset bundle.

## Objective

Run the final export bundle tool after the required upstream stages have completed.
The canonical export is tet-only and contains the six physical/manifest files plus
a required `appearance/` directory. Preserve upstream
artifacts by copying them; do not regenerate, rescale, or reinterpret mesh/material
outputs during final export.

## Inputs

- Runtime `run_root`.
- Canonical final tet mesh and material/topology artifacts:
  `final_mesh.mesh`, `heterogeneous_params.npz`, `inferred_params.json`,
  `volume_topology.json`, and `metric_mesh_scaling.json`.
- Cleanup artifacts when the run started from a source image.
- Genesis diagnostics artifacts when diagnostics ran.
- Runtime state paths for `final_export_dir`, `final_export_manifest_path`, and `final_asset_refinement_dir`.
- A complete, validated active-revision post-mesh visual bundle.

## Expected Outputs

- `final_export_manifest.json`.
- `final_mesh.mesh`.
- `heterogeneous_params.npz`.
- `inferred_params.json`.
- `volume_topology.json`.
- `metric_mesh_scaling.json`.
- `appearance/`, containing an exact recursive copy of the working visual bundle.

No other files or directories are exported into the canonical final-export root.
Cleanup, asset-refinement, and diagnostics artifacts remain provenance inputs only:
the manifest records their source paths/directories when available, but the final
export directory itself contains the six canonical files above plus `appearance/`.
The manifest recursively lists and hashes every regular file under `appearance/`.
Final export never rebakes, repairs, merges, or otherwise modifies the working
visual bundle; every re-export fully replaces stale canonical appearance content.

The manifest must not include representation decisions, homogeneous parameters, or
surface audit artifacts. Reject canonical JSON containing representation,
shell-thickness, wall-thickness, or triangle-array contract fields, and reject
canonical parameter archives containing legacy triangle arrays.

## Required Tool Order

1. Call `run_final_export_bundle_stage` exactly once with the supplied `run_root`.
2. If the export succeeds and the manifest, six physical/manifest files, and required validated `appearance/` exist, call `submit_final_export_stage` exactly once.
3. If required upstream artifacts are missing or the final export tool fails, call `halt_final_export_stage` exactly once with the failure reason.

## Failure Cases

- Missing final mesh or parameter artifacts.
- Missing, incomplete, or corrupt post-mesh visual bundle.
- Missing final export manifest.
- Export output written outside the runtime state's declared final export directory.
- Copying cleanup, diagnostics, asset-refinement, homogeneous-parameter, surface-audit, or other noncanonical artifacts into the final export directory.
- Dropping cleanup, diagnostics, or asset-refinement provenance records from the manifest when those source artifacts are present in runtime state.
- Changing the relative output layout without an explicit later migration feature.
