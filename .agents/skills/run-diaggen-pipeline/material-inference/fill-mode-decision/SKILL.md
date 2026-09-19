---
name: material-inference-fill-mode-decision
description: Decide per-part solid_fill or hollow_wall topology after geometry analysis.
---

# Fill-Mode Decision

Use this mini skill after geometry analysis and before material semantics.

Decide `volume_fill_mode` for each indexed part:

- `solid_fill`: the part should become filled tetrahedral material.
- `hollow_wall`: the part should remain a hollow volumetric band or shell-like structure with empty interior space.

This is a topology phase only. Do not estimate density, stiffness, Poisson ratio, friction, material names, part texture, or thickness. Hollow geometry is later derived mechanically from fixed voxel-band layers.

Preserve exact `part_index` and `part_color_rgb`. Output exactly one part entry per indexed part.

Payload contract:

```json
{
  "schema_version": "hag4r-material-fill-mode-decision-v1",
  "inferred_object_name": "object name",
  "parts": [
    {
      "part_index": 0,
      "part_color_rgb": [255.0, 0.0, 0.0],
      "volume_fill_mode": "solid_fill",
      "fill_mode_rationale": "non-empty explanation",
      "fill_mode_evidence": ["non-empty geometry evidence"]
    }
  ]
}
```

`fill_mode_rationale` and `fill_mode_evidence` must explain geometry/topology evidence from the indexed views and prior geometry analysis. If evidence is ambiguous, choose the mode whose downstream volumetric mesh is more likely to preserve the object behavior.
