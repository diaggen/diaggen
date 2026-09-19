---
name: material-inference-geometry-analysis
description: Analyze indexed part geometry without making material, topology, or numeric parameter claims.
---

# Geometry Analysis

Use this mini skill immediately before `run_material_geometry_analysis_stage`.

Author one geometry-only entry for every indexed part. Preserve exact `part_index` and `part_color_rgb`.

Required fields per part:

- `part_index`
- `part_color_rgb`
- `geometric_category`
- `geometry_description`
- `relative_size`
- `attachment_pattern`

Forbidden outputs:

- `volume_fill_mode`
- old representation fields
- `part_name`
- `part_semantics`
- `part_texture`
- `major_material_name`
- `density_kg_m3`
- `youngs_modulus_pa`
- `poisson_ratio`
- `friction_coefficient`
- thickness fields

Reason only from visible segmented geometry, relative part size, placement, and attachment. Defer hollow-vs-filled topology to the fill-mode phase and defer material/numeric choices to the later material phase.
