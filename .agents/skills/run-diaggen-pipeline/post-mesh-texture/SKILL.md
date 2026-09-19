---
name: post-mesh-texture
description: Build and validate the active revision's exact monolithic textured visual bundle after diagnostics acceptance, or after mesh processing when diagnostics are bypassed, and before final export.
---

# Post-mesh Texture

Use this skill only for the planned `hag4r_post_mesh_texture` stage. Diagnostics
remain segmentation-only and never read or render this stage's outputs.

## Inputs and environment

- Use the supplied runtime `run_root`.
- Read only the active revision paths declared by runtime state: OmniPart appearance
  manifest, monolithic `.mesh`, heterogeneous parameters, metric scaling, and
  inferred material JSON.
- The deterministic runner manages `.conda/omnipart`; preserve the runtime worker's
  `CUDA_VISIBLE_DEVICES`. Do not install dependencies or invoke system Python.

## Expected outputs

The active revision's `mesh_processing/post_mesh_texture/` directory must match the
frozen exact inventory: embedded-texture `visual_mesh.glb`, `albedo.png`, exact
`visual_to_physics.npz`, `visual_manifest.json`,
`post_mesh_texture_request.json`, and the five files under `qa/`.

Coverage and fill provenance are QA metrics only. They are never pass/degraded
gates and never require human approval.

## Required Tool Order

1. Call `run_post_mesh_texture_stage` exactly once with the supplied `run_root`.
2. Only after runner success and validation of the GLB, binding, manifest, QA
   report, exact inventory, and hashes, call `submit_post_mesh_texture_stage`
   exactly once.
3. On any prerequisite, runner, artifact, or validation failure, call
   `halt_post_mesh_texture_stage` exactly once with the failure reason and stop.
4. Never call final export when the runner or terminal action failed.
5. After runner success and successful submission, return control to the root
   orchestrator; only it may consume the next ordered `final-export` skill.

The submit terminal writes `pending_validation`; it does not execute this stage,
advance revision status, or invoke final export.
