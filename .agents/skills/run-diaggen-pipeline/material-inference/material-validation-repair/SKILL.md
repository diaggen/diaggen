---
name: material-inference-material-validation-repair
description: Repair invalid material inference outputs and require full revalidation before final write.
---

# Material Validation Repair

Use this mini skill only after `validate_repair_material_predictions_stage` reports that the Codex-authored `material_payload` failed validation and asks for a full replacement `repair_payload`.

Repair cannot change fill mode. If `volume_fill_mode` is wrong, rerun the fill-mode phase instead of editing the material payload.

Repair must return a complete replacement JSON object with one entry for every indexed part. Preserve exact `part_index` and `part_color_rgb`.

Fix these failures by choosing a plausible physical material and replacing the full field cluster for the affected part:

- missing, duplicate, extra, renumbered, or recolored part entries;
- missing required text or numeric fields;
- `density_kg_m3 <= 0`;
- `youngs_modulus_pa <= 0`;
- `poisson_ratio <= -1` or `poisson_ratio >= 0.5`;
- `friction_coefficient < 0`;
- non-physical material names such as `void`, `artifact`, `unknown`, `air`, or `none`;
- missing or mismatched `profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, poisson_ratio=<nu>, friction_coefficient=<mu>` binding.

Do not preserve the forbidden placeholder cluster `major_material_name: plastic`, `density_kg_m3: 1050`, `youngs_modulus_pa: 0`, `poisson_ratio: 0,5`, and `friction_coefficient: 0`. Repair by replacing the whole material hypothesis and numeric field set with material-consistent JSON-valid values.

Do not estimate thickness. Hollow geometry is derived later by mesh processing.

After repair, revalidate mentally before calling the tool again:

- exact part coverage and colors;
- plausible physical material names;
- strictly positive density and Young's modulus;
- valid Poisson ratio;
- non-negative friction;
- tuple values exactly match the numeric fields;
- no topology, old representation, or thickness fields appear in the material payload.
