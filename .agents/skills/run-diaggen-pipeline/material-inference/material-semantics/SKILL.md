---
name: material-inference-material-semantics
description: Infer indexed part role, material class, texture, and four numeric fields for the combined HAG4R material tool.
---

# Material Semantics

Use this mini skill with `physical-parameter-estimation/SKILL.md` before authoring `material_payload` for `run_material_semantics_inference_stage`.

The fill-mode decision is already fixed. Do not output or change `volume_fill_mode`, `fill_mode_rationale`, or `fill_mode_evidence` in `material_payload`.

Each part entry must preserve exact `part_index` and `part_color_rgb`, and must include:

- `part_name`
- `part_semantics`
- `part_texture`
- `major_material_name`
- `density_kg_m3`
- `youngs_modulus_pa`
- `poisson_ratio`
- `friction_coefficient`

For the combined HAG4R tool, material semantics and numeric parameters are authored together. The standalone semantic analysis should inform the numeric choice, but the final payload must include all four numeric fields.

Do not output the forbidden placeholder cluster `major_material_name: plastic`, `density_kg_m3: 1050`, `youngs_modulus_pa: 0`, `poisson_ratio: 0,5`, and `friction_coefficient: 0` for any real indexed part. Uncertainty must be resolved by choosing a plausible physical material, not by keeping placeholder values.

Every combined-tool `part_semantics` string must end with a schema-compatible binding clause: `profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, poisson_ratio=<nu>, friction_coefficient=<mu>`. The numeric fields for that part must exactly match the tuple values in the string.

Do not estimate thickness. Hollow geometry is derived later by mesh processing.
