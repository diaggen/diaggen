---
name: omnipart
description: Run OmniPart part-segmented mesh generation from the segmentation agent's 2D mask bundle and the cleanup agent's object description.
---

# OmniPart Part Generation

Use this skill when the OmniPart stage agent turns HAG4R-owned 2D segmentation outputs into OmniPart part-segmented mesh artifacts.

## Objective

Run OmniPart generation from the segmentation agent's validated 2D segmentation manifest and the cleanup agent's concise object description. The stage should preserve the object identity and part semantics already established upstream, while delegating 3D part generation to OmniPart.

This agent does not create a new object description, author SAM3 prompts, edit images, infer materials, or post-process meshes. Its job is to invoke the OmniPart generation stage on the existing segmentation bundle and verify that OmniPart artifacts are produced.

## Inputs

- Runtime `run_root`.
- `segmentation_manifest_path` from the segmentation agent.
- Processed 2D masks, labels, and view inputs referenced by the segmentation manifest.
- `object_description_path` from the image cleanup agent.
- `inferred_object_name` and `object_description` from the cleanup-generated object-description JSON.
- Output paths from runtime state.
- Optional diagnostic suggestions routed to the OmniPart stage.

The segmentation manifest is the authoritative machine-readable input for OmniPart. The cleanup-generated object description is concise semantic context only; do not expand it into extra segmentation hints, joint hints, material cues, or a new schema.

## Expected Outputs

The OmniPart stage must produce the artifacts declared by `run_omnipart_generate_parts_stage`, including:

- `part_labels.npz`
- `mesh_combined_colored_by_parts.glb`
- `mesh_combined.mesh`
- per-view `mesh_combined_part_labels_<view>.png` segmented-view previews.
- stage metadata recorded in runtime state for `omnipart_generate_parts`.

These are OmniPart part-geometry, label, and preview artifacts. They are consumed by
the later volumetric mesh-processing stage; the old combined OmniPart mesh is not
the canonical final simulation mesh.

## Available Tools

- `run_omnipart_generate_parts_stage`
- `submit_omnipart_stage`
- `halt_omnipart_stage`

## Required Tool Order

1. Call `run_omnipart_generate_parts_stage` exactly once with the supplied `run_root`.
2. If OmniPart generation succeeds and the required artifacts are present, call `submit_omnipart_stage` exactly once.
3. If the manifest, upstream mask bundle, object description, OmniPart environment, or expected artifacts are missing or invalid, call `halt_omnipart_stage` exactly once with the failure reason.

## Workflow

1. Treat the segmentation agent's manifest and referenced 2D masks as the only mask inputs to OmniPart.
2. Treat the cleanup agent's `object_description` as semantic context for object identity and part meaning, not as a prompt to regenerate upstream artifacts.
3. Run the single OmniPart generation tool for this stage.
4. Confirm the stage result reports the expected OmniPart artifacts.
5. Submit or halt the stage; do not call downstream material or mesh-processing tools.

## Failure Cases

- Missing `segmentation_manifest_path`.
- Missing processed 2D mask data referenced by the segmentation manifest.
- Missing cleanup-generated `object_description_path` or empty `object_description`.
- Attempting to rerun image cleanup, object-description generation, or SAM3 segmentation inside this stage.
- Falling back to OmniPart's old native pre-3D segmentation route.
- Missing part labels, combined mesh, volumetric mesh, or segmented-view preview outputs.
