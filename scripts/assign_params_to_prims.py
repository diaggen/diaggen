# Given E (Young's Modulus), nu (Poisson's Ratio), density, and volumetric
# fill mode for each color-locked part, write the per-part material table
# consumed by HAG4R's owner-derived volumetric meshing route.


import argparse
import json
import os

import numpy as np


FILL_MODE_CODES = {"solid_fill": 0, "hollow_wall": 1}
FORBIDDEN_LEGACY_FIELDS = {
    "representation",
    "representation_decision",
    "shell_" + "thickness_m",
    "wall_" + "thickness_m",
}
LEGACY_ARRAY_PREFIX = "t" + "ri_"


def _colors_equal(a, b):
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    if left.shape == right.shape:
        return bool(np.array_equal(left, right))
    if left.shape == (3,) and right.shape == (4,) and right[3] == 255:
        return bool(np.array_equal(left, right[:3]))
    if left.shape == (4,) and right.shape == (3,) and left[3] == 255:
        return bool(np.array_equal(left[:3], right))
    return False


def _reject_legacy_fields(pred):
    for key in pred:
        if key in FORBIDDEN_LEGACY_FIELDS or key.startswith(LEGACY_ARRAY_PREFIX):
            raise ValueError(f"legacy material field is not supported by volumetric-only assignment: {key}")


def load_part_properties(inferred_params_path, part_colors):
    with open(inferred_params_path, "r", encoding="utf-8") as f:
        inferred_params = json.load(f)

    predictions = inferred_params.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("inferred_params.json must contain a predictions list")
    n_parts = int(part_colors.shape[0])
    part_E_nu = np.zeros((n_parts, 2), dtype=float)
    part_density = np.zeros((n_parts, 1), dtype=float)
    part_indices = np.arange(n_parts, dtype=np.int64)
    part_fill_mode_codes = np.full((n_parts,), -1, dtype=np.int64)
    part_fill_mode_names = np.full((n_parts,), "", dtype="<U16")
    seen = set()
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError("each prediction must be an object")
        _reject_legacy_fields(pred)
        idx = int(pred["part_index"])
        if idx in seen:
            raise ValueError(f"duplicate prediction for part_index {idx}")
        if idx < 0 or idx >= n_parts:
            raise ValueError(f"part_index {idx} is outside part_colors range")
        if not _colors_equal(pred.get("part_color_rgb"), part_colors[idx]):
            raise ValueError(f"part_color_rgb mismatch for part_index {idx}")
        fill_mode = str(pred["volume_fill_mode"])
        if fill_mode not in FILL_MODE_CODES:
            raise ValueError(f"volume_fill_mode must be one of {sorted(FILL_MODE_CODES)} for part_index {idx}")
        part_E_nu[idx, 0] = pred["youngs_modulus_pa"]
        part_E_nu[idx, 1] = pred["poisson_ratio"]
        part_density[idx, 0] = pred["density_kg_m3"]
        part_fill_mode_codes[idx] = FILL_MODE_CODES[fill_mode]
        part_fill_mode_names[idx] = fill_mode
        seen.add(idx)
    expected = set(range(n_parts))
    if seen != expected:
        raise ValueError(f"predictions must cover exact part indices {sorted(expected)}; got {sorted(seen)}")
    return part_indices, part_E_nu, part_density, part_fill_mode_codes, part_fill_mode_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--part_labels',
        type=str,
        help='path to OmniPart part_labels.npz containing the canonical part_colors table',
        required=True)
    p.add_argument(
        '--inferred_params_path',
        type=str,
        help='path to the json file containing per-part E and nu',
        required=True)
    p.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='explicit output .npz path; overrides --output_dir and the legacy outputs/assign_params_to_prims/<json_stem> layout')
    p.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='explicit output directory; output filename is volumetric_params.npz')

    args = p.parse_args()
    part_labels_path = args.part_labels
    inferred_params_path = args.inferred_params_path

    # set output dir
    script_name = "assign_params_to_prims"
    json_stem = os.path.splitext(os.path.basename(inferred_params_path))[0]
    output_dir = args.output_dir or f"outputs/{script_name}/{json_stem}"
    os.makedirs(output_dir, exist_ok=True)

    part_labels_dict = np.load(part_labels_path, allow_pickle=True)
    part_colors = part_labels_dict["part_colors"]

    part_indices, part_E_nu, part_density, part_fill_mode_codes, part_fill_mode_names = load_part_properties(
        inferred_params_path, part_colors)

    output_path = args.output_path or os.path.join(output_dir, "volumetric_params.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        part_indices=part_indices,
        part_E_nu=part_E_nu,
        part_density=part_density,
        part_colors=part_colors,
        part_volume_fill_mode_codes=part_fill_mode_codes,
        part_volume_fill_mode_names=part_fill_mode_names,
    )
    print(f"Saved primitive-wise parameters to {output_path}")


if __name__ == "__main__":
    main()
