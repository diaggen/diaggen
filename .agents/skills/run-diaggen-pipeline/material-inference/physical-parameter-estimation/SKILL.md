---
name: material-inference-physical-parameter-estimation
description: Estimate Genesis-practical material parameters for locked indexed parts.
---

# Physical Parameter Estimation

Use this mini skill together with `material-semantics/SKILL.md` before authoring `material_payload`.

Required numeric fields:

- `density_kg_m3`: strictly positive.
- `youngs_modulus_pa`: strictly positive.
- `poisson_ratio`: greater than `-1` and less than `0.5`.
- `friction_coefficient`: non-negative.

Invalid sentinel values are forbidden: `0` for density or Young's modulus; negative density, modulus, or friction; `poisson_ratio: 0.5`; comma-decimal values such as `poisson_ratio: 0,5`; and non-physical material names such as `void`, `artifact`, `unknown`, `air`, or `none`.

Do not estimate thickness. Hollow geometry is derived later by mesh processing.

## Material Profile Bank

When uncertain, choose the closest profile row and copy all four numeric fields before adjusting only when visual evidence strongly supports a different physical value.

| profile | density_kg_m3 | youngs_modulus_pa | poisson_ratio | friction_coefficient |
|---|---:|---:|---:|---:|
| thin_plastic_sheet | 1050 | 2000000000 | 0.38 | 0.35 |
| rigid_plastic_hub | 1100 | 2200000000 | 0.37 | 0.35 |
| wooden_stick | 650 | 9000000000 | 0.35 | 0.45 |
| paper_cardboard_sheet | 700 | 3000000000 | 0.30 | 0.55 |
| rubber_flexible | 1100 | 5000000 | 0.49 | 0.90 |
| metal_connector | 7850 | 200000000000 | 0.30 | 0.40 |

Every `part_semantics` string must end with:

```text
profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, poisson_ratio=<nu>, friction_coefficient=<mu>
```

Worked example:

```json
{
  "inferred_object_name": "toilet plunger",
  "parts": [
    {
      "part_index": 0,
      "part_name": "painted_wood_handle",
      "part_semantics": "rigid long handle used to push and pull the plunger; profile=wooden_stick; tuple density_kg_m3=650, youngs_modulus_pa=9000000000, poisson_ratio=0.35, friction_coefficient=0.45",
      "part_texture": "smooth painted wood with light grip wear",
      "major_material_name": "wood",
      "part_color_rgb": [141.0, 211.0, 199.0],
      "density_kg_m3": 650,
      "youngs_modulus_pa": 9000000000,
      "poisson_ratio": 0.35,
      "friction_coefficient": 0.45
    }
  ]
}
```

Use profile rows as practical Genesis values, not exact real-world material certificates.
