### Plan Revision: Post-3D Six-View Rendering Only

#### Summary
- Drop all 2D-segmentation-related work from this plan.
- Do not do 2D↔3D matching, label transfer, color policy reuse, or mask-based logic.
- Only change behavior after 3D parts (`part*.glb`) are already generated.

#### Implementation Changes
- Keep `merge_parts(save_dir)` as the integration point.
- Keep random per-part colors as-is (`get_random_color(...)` in the existing loop).
- Keep all existing merged outputs (`mesh_combined.glb`, `mesh_combined_textured.glb`, `mesh_combined.mesh`, `mesh_combined_colored.vtu`, `part_labels.npz`).

- Replace the current single-view screenshot in `merge_parts()` with six enclosing views:
  - `front`
  - `back`
  - `left`
  - `right`
  - `top`
  - `bottom`
- Ensure camera framing encloses the whole object for each view.
- Export six PNGs in `save_dir`:
  - `mesh_combined_part_labels_front.png`
  - `mesh_combined_part_labels_back.png`
  - `mesh_combined_part_labels_left.png`
  - `mesh_combined_part_labels_right.png`
  - `mesh_combined_part_labels_top.png`
  - `mesh_combined_part_labels_bottom.png`

- Renderer backend policy for this iteration:
  - Primary: PyVista off-screen rendering (minimal code change from current implementation).
  - Optional alternative: MeshRenderer path can be added later if needed. Don't add anything MeshRenderer-specific code for now; just ensure the design allows for it to be added later without affecting the PyVista path.

#### Test Plan
- Run normal inference once and verify all six view images are produced.
- Verify each image is non-empty and contains rendered object pixels.
- Verify no new 2D-mask/segment artifacts are introduced by this change.
- Perform test (and debug) only when explicitly instructed by human user.

#### Defaults Locked
- Random colors are acceptable even if outputs are not deterministic across runs.
- Six enclosing views are sufficient for now.
- Scope is post-3D-parts only; 2D mask/segment pipeline remains untouched.
