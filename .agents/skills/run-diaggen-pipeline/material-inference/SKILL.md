---
name: material-inference
description: Run HAG4R material inference through ordered color-locked geometry, fill-mode, semantics, numeric, validation, and write tools.
---

# Material Inference

Use this stage to author material-inference payloads for HAG4R. The Codex stage agent authors every topology and material judgment. Tools only load context, validate exact schemas, record scratch/history, and write files; they do not call hidden VLM, model, or network helpers.

If this stage is the designated diagnostic repair destination, digest the diagnostic repair brief at the stage-agent planning level. Pass only the resulting stage choices through the optional `diagnostic_repair_plan` argument.

## Required Order

1. Call `load_material_part_labels_stage`.
2. Call `build_material_indexed_parts_stage`.
3. Read `geometry-analysis/SKILL.md`.
4. Author geometry JSON and call `run_material_geometry_analysis_stage`.
5. Read `fill-mode-decision/SKILL.md`.
6. Author fill-mode JSON and call `decide_material_fill_mode_stage`.
7. Read `material-semantics/SKILL.md` and `physical-parameter-estimation/SKILL.md`.
8. Author the combined material-class and numeric-parameter JSON and call `run_material_semantics_inference_stage`.
9. Read `material-validation-repair/SKILL.md`.
10. Call `validate_repair_material_predictions_stage`; provide a full replacement `repair_payload` only after a validation failure.
11. Call `write_inferred_material_params_stage`.
12. Submit the material stage only after the final write succeeds.

Do not skip a mini skill before its matching tool phase. Do not call semantics, validation, repair, or final write before `decide_material_fill_mode_stage` succeeds.

## Color Lock

Every payload must contain exactly one entry per indexed part. Preserve `part_index` and `part_color_rgb` exactly as returned by `build_material_indexed_parts_stage`.

Missing parts, duplicate indices, extra parts, renumbered indices, or changed colors are invalid. Repair payloads must preserve the same color lock.

## Payload Ownership

`fill_mode_payload` contains only topology fields:

- `part_index`
- `part_color_rgb`
- `volume_fill_mode`
- `fill_mode_rationale`
- `fill_mode_evidence`

`material_payload` and `repair_payload` have no topology, old representation fields, or thickness fields. They contain:

- `part_index`
- `part_name`
- `part_semantics`
- `part_texture`
- `major_material_name`
- `part_color_rgb`
- `density_kg_m3`
- `youngs_modulus_pa`
- `poisson_ratio`
- `friction_coefficient`

If a fill mode must change, rerun the fill-mode phase. Do not use material repair as a topology patch.

## Physical Validation

Every real indexed part must receive final candidate values, not placeholders. The strictly positive fields are not optional:

- `density_kg_m3 > 0`
- `youngs_modulus_pa > 0`
- `-1 < poisson_ratio < 0.5`
- `friction_coefficient >= 0`
- `major_material_name` must name a plausible physical material, not void, unknown, air, artifact, or none
- every `part_semantics` string includes a schema-compatible `profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, poisson_ratio=<nu>, friction_coefficient=<mu>` binding clause whose values match the numeric fields

Do not output the forbidden plastic-plus-zero placeholder cluster: `major_material_name: plastic`, `density_kg_m3: 1050`, `youngs_modulus_pa: 0`, `poisson_ratio: 0,5`, and `friction_coefficient: 0`.

Use the Material Profile Bank in `physical-parameter-estimation/SKILL.md` and the worked toilet-plunger output pattern as style references. The semantics output numeric fields as final candidate values.

Final `inferred_params.json["predictions"]` is written by the tool after validation. It merges the fixed `volume_fill_mode` entries with the validated material entries.

## Exit Criteria

- Geometry, fill-mode, semantics, validation, and final write phases succeeded in order.
- The final payload has no old representation decision and no thickness fields.
- `predictions` contains exact index/color coverage, `volume_fill_mode`, rationale/evidence, material semantics, texture, material name, and the four numeric fields.
