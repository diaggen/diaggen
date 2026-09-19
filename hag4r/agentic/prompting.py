from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HETEROGENEOUS_CUES = (
    "metal rings, clips, screws, buckles, pins, springs, rods, and hinges",
    "rubber tires, grips, feet, seals, bands, straps, and soft pads",
    "fabric, leather, paper, foam, plush, fur, sponge, and other highly deformable regions",
    "transparent, glossy, ceramic, glass, wooden, and dense-looking rigid inserts",
    "thin shells, handles, pull tabs, cords, and flexible connectors",
)

DEFAULT_JOINT_CUES = (
    "hinges",
    "revolute pivots",
    "ball joints",
    "rings through holes",
    "straps through loops",
    "axles",
    "fold lines",
    "thin flexible necks",
)


@dataclass(frozen=True)
class SegmentationPromptStrategy:
    object_description: str
    system_prompt: str
    user_prompt: str
    merge_rules: tuple[str, ...]
    preserve_rules: tuple[str, ...]
    joint_rules: tuple[str, ...]
    fallback_triggers: tuple[str, ...]
    target_part_count_hint: str


@dataclass(frozen=True)
class MaterialPromptStrategy:
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, str]
    evidence_rules: tuple[str, ...]
    guardrails: tuple[str, ...]


def build_segmentation_prompt_strategy(
    object_description: str,
    *,
    known_heterogeneous_parts: tuple[str, ...] = (),
    joint_hints: tuple[str, ...] = (),
) -> SegmentationPromptStrategy:
    hetero_cues = known_heterogeneous_parts or DEFAULT_HETEROGENEOUS_CUES
    joints = joint_hints or DEFAULT_JOINT_CUES
    preserve_rules = tuple(
        f"Preserve {cue} as separate parts when visible or physically plausible."
        for cue in hetero_cues
    )
    joint_rules = tuple(
        f"Segment {joint} separately when it can create relative motion or localized compliance."
        for joint in joints
    )
    merge_rules = (
        "Merge texture-only color changes, logos, shallow grooves, and decorative speckles.",
        "Merge tiny repeated elements unless they change stiffness, density, contact, or articulation.",
        "Prefer coarse simulation parts over visual over-segmentation.",
        "Keep boundaries simple enough for downstream meshing and part-label transfer.",
    )
    fallback_triggers = (
        "OmniPart cannot accept or preserve language-conditioned part hints.",
        "Automatic masks split one material into many decorative fragments.",
        "Automatic masks merge visibly different stiff/soft materials.",
        "Joint-like regions are not segmented after one prompt refinement.",
    )
    target_part_count_hint = (
        "Aim for 3-12 simulation-relevant parts unless the object has more true material "
        "or articulation groups."
    )
    system_prompt = (
        "You are planning 2D segmentation for physics-aware asset generation. "
        "Favor simulation-relevant material and articulation boundaries over visual detail."
    )
    user_prompt = (
        f"Segment `{object_description}` for simulation. {target_part_count_hint} "
        "Separate stiff/soft/dense/thin/flexible regions and joint-like structures. "
        "Merge decorative or texture-only details that do not affect geometry or material behavior."
    )
    return SegmentationPromptStrategy(
        object_description=object_description,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        merge_rules=merge_rules,
        preserve_rules=preserve_rules,
        joint_rules=joint_rules,
        fallback_triggers=fallback_triggers,
        target_part_count_hint=target_part_count_hint,
    )


def build_material_prompt_strategy(
    object_description: str,
    *,
    part_names: tuple[str, ...] = (),
) -> MaterialPromptStrategy:
    part_text = ", ".join(part_names) if part_names else "all segmented parts"
    system_prompt = (
        "You infer physically plausible material parameters for robotics simulation. "
        "Use visual evidence and part semantics, return conservative values, and output only "
        "the requested structured fields."
    )
    user_prompt = (
        f"Infer material parameters for `{object_description}` using the volumetric-only "
        f"pipeline. Estimate parameters for {part_text}. Compare relative stiffness, "
        "density, and friction only as needed to choose plausible numeric values. Preserve "
        "part identity exactly, preserve per-part `volume_fill_mode` context exactly, and "
        "return one prediction per provided part index."
    )
    output_schema = {
        "predictions": (
            "array of per-part objects. Each object must include part_index, part_name, "
            "part_semantics, part_texture, major_material_name, part_color_rgb, "
            "volume_fill_mode, density_kg_m3, youngs_modulus_pa, poisson_ratio, and "
            "friction_coefficient."
        ),
        "inferred_object_name": "short object name inferred from the image and prompt",
    }
    evidence_rules = (
        "Use part labels, the processed source image, rendered views, and material contrast cues.",
        "Reason about relative stiffness before assigning absolute values.",
        "Flag parts where visual evidence is weak and simulator feedback should decide.",
        "Keep values internally consistent across neighboring materials.",
    )
    guardrails = (
        "Do not invent invisible internal mechanisms unless the prompt or views support them.",
        "Do not average all parts into one material when heterogeneous regions are visible.",
        "Do not make decorative texture-only regions separate material entries.",
        "Keep image evidence limited to the processed source plus six views or fewer.",
    )
    return MaterialPromptStrategy(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=output_schema,
        evidence_rules=evidence_rules,
        guardrails=guardrails,
    )


__all__ = [
    "DEFAULT_HETEROGENEOUS_CUES",
    "DEFAULT_JOINT_CUES",
    "MaterialPromptStrategy",
    "SegmentationPromptStrategy",
    "build_material_prompt_strategy",
    "build_segmentation_prompt_strategy",
]
