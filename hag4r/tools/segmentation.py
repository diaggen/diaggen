from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from hag4r.agentic.state import (
    ArtifactRole,
    Stage,
    StageRunResult,
    read_git_branch,
    to_json_dict,
)
from hag4r.tools.common import (
    EnvName,
    _artifact,
    _conda_module_argv,
    _repo_root as _tools_repo_root,
    _run_subprocess_stage,
    resolve_conda_env,
)


SAM3_OMNIPART_2D_SEGMENTATION_STAGE = "sam3_omnipart_2d_segmentation"
SEGMENTATION_SCHEMA_VERSION = "hag4r-sam3-omnipart-2d-v1"
SAM3_PROMPT_SCHEMA_VERSION = "hag4r-sam3-prompt-v1"
SAM3_RAW_SCHEMA_VERSION = "hag4r-sam3-raw-v1"
SAM3_CHECKPOINT_ENV_VAR = "HAG4R_SAM3_CHECKPOINT_PATH"
TARGET_SIZE = 518
ORDERED_MASK_SIZE = 37
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_SIZE_THRESHOLD = 2000
MAX_SAM3_CONCEPTS = 20
MAX_SAM3_BACKGROUND_RATIO = 0.35
SCORE_THRESHOLD_SCOPE = "post_sam3_mask_score_filter_only"
SAM3_RAW_OUTPUT_STAGE = "pre_postprocess_sam3_output"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _object_name_from_paths(image_path: Path, output_dir: Path, object_name: str | None = None) -> str:
    if object_name:
        return object_name
    if output_dir.name:
        return output_dir.name
    return image_path.stem


@dataclass(frozen=True)
class Sam3OmniPart2DSegmentationRequest:
    image_path: Path
    sam3_prompt: str
    output_dir: Path
    object_name: str
    object_description_path: Path | None
    sam3_root: Path
    post_sam3_score_threshold: float
    size_threshold: int
    sam3_prompt_path: Path
    sam3_raw_npz_path: Path
    sam3_raw_metadata_path: Path
    processed_rgba_path: Path
    processed_white_bg_png_path: Path
    processed_black_bg_png_path: Path
    image_white_bg_tensor_path: Path
    image_black_bg_tensor_path: Path
    group_ids_path: Path
    mask_exr_path: Path
    ordered_mask_input_path: Path
    ordered_mask_vis_path: Path
    segmentation_overlay_path: Path
    segmentation_manifest_path: Path
    expected_outputs: tuple[Path, ...]

    @property
    def score_threshold(self) -> float:
        return self.post_sam3_score_threshold


def _resolve_post_sam3_score_threshold(
    *,
    post_sam3_score_threshold: float | None,
    score_threshold: float | None,
) -> float:
    if (
        post_sam3_score_threshold is not None
        and score_threshold is not None
        and post_sam3_score_threshold != score_threshold
    ):
        raise ValueError(
            "post_sam3_score_threshold and score_threshold are aliases for the same post-SAM3 "
            "mask score filter; provide one value or matching values."
        )
    if post_sam3_score_threshold is not None:
        return post_sam3_score_threshold
    if score_threshold is not None:
        return score_threshold
    return DEFAULT_SCORE_THRESHOLD


def build_sam3_omnipart_2d_segmentation_request(
    *,
    image_path: Path,
    sam3_prompt: str,
    output_dir: Path,
    object_description_path: Path | None = None,
    object_name: str | None = None,
    sam3_root: Path | None = None,
    post_sam3_score_threshold: float | None = None,
    score_threshold: float | None = None,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> Sam3OmniPart2DSegmentationRequest:
    resolved_image_path = _path_for_repo(image_path)
    resolved_output_dir = _path_for_repo(output_dir)
    resolved_object_name = _object_name_from_paths(resolved_image_path, resolved_output_dir, object_name)
    resolved_object_description_path = (
        _path_for_repo(object_description_path) if object_description_path is not None else None
    )
    resolved_sam3_root = _path_for_repo(sam3_root or Path("third_party/sam3"))
    resolved_post_sam3_score_threshold = _resolve_post_sam3_score_threshold(
        post_sam3_score_threshold=post_sam3_score_threshold,
        score_threshold=score_threshold,
    )
    prefix = resolved_output_dir / resolved_object_name
    expected_outputs = (
        prefix.with_name(f"{resolved_object_name}_sam3_prompt.json"),
        prefix.with_name(f"{resolved_object_name}_sam3_raw.npz"),
        prefix.with_name(f"{resolved_object_name}_sam3_raw.json"),
        prefix.with_name(f"{resolved_object_name}_processed.png"),
        prefix.with_name(f"{resolved_object_name}_white_bg.png"),
        prefix.with_name(f"{resolved_object_name}_black_bg.png"),
        prefix.with_name(f"{resolved_object_name}_image_white_bg.npy"),
        prefix.with_name(f"{resolved_object_name}_image_black_bg.npy"),
        prefix.with_name(f"{resolved_object_name}_group_ids.npy"),
        prefix.with_name(f"{resolved_object_name}_mask.exr"),
        prefix.with_name(f"{resolved_object_name}_ordered_mask_input.npy"),
        prefix.with_name(f"{resolved_object_name}_ordered_mask_vis.png"),
        prefix.with_name(f"{resolved_object_name}_segmentation_overlay.png"),
        resolved_output_dir / "segmentation_manifest.json",
    )
    return Sam3OmniPart2DSegmentationRequest(
        image_path=resolved_image_path,
        sam3_prompt=sam3_prompt,
        output_dir=resolved_output_dir,
        object_name=resolved_object_name,
        object_description_path=resolved_object_description_path,
        sam3_root=resolved_sam3_root,
        post_sam3_score_threshold=resolved_post_sam3_score_threshold,
        size_threshold=size_threshold,
        sam3_prompt_path=expected_outputs[0],
        sam3_raw_npz_path=expected_outputs[1],
        sam3_raw_metadata_path=expected_outputs[2],
        processed_rgba_path=expected_outputs[3],
        processed_white_bg_png_path=expected_outputs[4],
        processed_black_bg_png_path=expected_outputs[5],
        image_white_bg_tensor_path=expected_outputs[6],
        image_black_bg_tensor_path=expected_outputs[7],
        group_ids_path=expected_outputs[8],
        mask_exr_path=expected_outputs[9],
        ordered_mask_input_path=expected_outputs[10],
        ordered_mask_vis_path=expected_outputs[11],
        segmentation_overlay_path=expected_outputs[12],
        segmentation_manifest_path=expected_outputs[13],
        expected_outputs=expected_outputs,
    )


def sam3_omnipart_2d_segmentation_request_payload(
    image_path: str,
    sam3_prompt: str,
    output_dir: str,
    *,
    object_description_path: str | None = None,
    object_name: str | None = None,
    post_sam3_score_threshold: float | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    request = build_sam3_omnipart_2d_segmentation_request(
        image_path=Path(image_path),
        sam3_prompt=sam3_prompt,
        output_dir=Path(output_dir),
        object_description_path=Path(object_description_path) if object_description_path else None,
        object_name=object_name,
        post_sam3_score_threshold=post_sam3_score_threshold,
        score_threshold=score_threshold,
    )
    payload = to_json_dict(request)
    payload["score_threshold"] = request.post_sam3_score_threshold
    payload["score_threshold_scope"] = SCORE_THRESHOLD_SCOPE
    payload["status"] = "planned"
    payload["stage_name"] = SAM3_OMNIPART_2D_SEGMENTATION_STAGE
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_object_description(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"object-description artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _dominant_corner_color(rgb: np.ndarray) -> np.ndarray:
    samples = np.concatenate(
        (
            rgb[:8, :8].reshape(-1, 3),
            rgb[-8:, :8].reshape(-1, 3),
            rgb[:8, -8:].reshape(-1, 3),
            rgb[-8:, -8:].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(samples.astype(np.float32), axis=0)


def _estimate_alpha_from_rgb(rgb: np.ndarray) -> np.ndarray:
    background = _dominant_corner_color(rgb)
    color_distance = np.linalg.norm(rgb.astype(np.float32) - background[None, None, :], axis=-1)
    alpha = np.where(color_distance > 18.0, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(alpha)) < max(64, alpha.size // 200):
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return alpha


def _resize_and_pad_rgba(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    square = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
    paste_xy = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
    square.paste(resized, paste_xy, resized)
    return square


def preprocess_cleaned_image(image_path: Path, *, target_size: int = TARGET_SIZE) -> tuple[Image.Image, dict[str, Any]]:
    image = Image.open(image_path)
    original_mode = image.mode
    rgba = image.convert("RGBA")
    rgba_array = np.array(rgba)
    if original_mode != "RGBA" or np.all(rgba_array[..., 3] == 255):
        rgba_array[..., 3] = _estimate_alpha_from_rgb(rgba_array[..., :3])
        rgba = Image.fromarray(rgba_array, mode="RGBA")
    processed = _resize_and_pad_rgba(rgba, target_size=target_size)
    processed_alpha = np.array(processed.getchannel("A"))
    bbox = _alpha_bbox(processed_alpha)
    metadata = {
        "cleaned_image_size": list(image.size),
        "cleaned_image_mode": original_mode,
        "cleaned_image_foreground_bbox_xyxy": list(bbox) if bbox is not None else None,
        "processed_size": [target_size, target_size],
    }
    return processed, metadata


def composite_rgba(rgba: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    background = Image.new("RGBA", rgba.size, (*color, 255))
    return Image.alpha_composite(background, rgba.convert("RGBA")).convert("RGB")


def image_to_chw_float(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def build_cleaned_image_visual_summary(
    *,
    image_sha256: str,
    image_metadata: dict[str, Any],
    processed_rgba: Image.Image,
) -> str:
    rgba = np.asarray(processed_rgba, dtype=np.uint8)
    alpha = rgba[..., 3] > 10
    if np.any(alpha):
        mean_rgb = rgba[..., :3][alpha].mean(axis=0)
        mean_text = ",".join(str(int(round(value))) for value in mean_rgb)
        fg_pixels = int(np.count_nonzero(alpha))
    else:
        mean_text = "none"
        fg_pixels = 0
    return (
        f"sha256={image_sha256[:12]}; size={image_metadata['cleaned_image_size']}; "
        f"mode={image_metadata['cleaned_image_mode']}; foreground_bbox="
        f"{image_metadata['cleaned_image_foreground_bbox_xyxy']}; foreground_pixels={fg_pixels}; "
        f"mean_foreground_rgb={mean_text}"
    )


def _normalize_sam3_concept_phrase(text: str) -> str:
    phrase = text.strip().strip("`'\"")
    phrase = phrase.replace("_", " ")
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = phrase.rstrip(".,;:!?").strip().strip("`'\"")
    return phrase


def _is_instruction_like_sam3_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "segment ",
            "return separate masks",
            "runtime prompt context",
            "downstream runtime contract",
            "sha256",
            "/",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".json",
            ".npy",
            ".npz",
            ".exr",
            "cleaned image path",
            "image evidence",
        )
    )


def _append_unique_concept(concepts: list[str], phrase: str) -> None:
    normalized = _normalize_sam3_concept_phrase(phrase)
    if not normalized:
        return
    existing = {concept.casefold() for concept in concepts}
    if normalized.casefold() not in existing:
        concepts.append(normalized)


def _word_count(text: str) -> int:
    return len(text.split())


def _trim_short_concept_tail(text: str) -> str:
    return re.sub(r"\b(?:and|or|with|including|include|includes)$", "", text, flags=re.IGNORECASE).strip()


def _short_concept_candidates(text: str, *, max_words: int = 8) -> tuple[str, ...]:
    normalized = _normalize_sam3_concept_phrase(text)
    if not normalized or _is_instruction_like_sam3_text(normalized):
        return ()
    normalized = _trim_short_concept_tail(
        re.sub(r"^(?:including|include|includes|and|or|with)\s+", "", normalized, flags=re.IGNORECASE).strip()
    )
    if not normalized or _is_instruction_like_sam3_text(normalized):
        return ()
    if 1 <= _word_count(normalized) <= max_words:
        return (normalized,)

    candidates: list[str] = []
    pending = [
        chunk.strip()
        for chunk in re.split(r"[,;:\n()]+", normalized)
        if chunk.strip()
    ]
    for pattern in (r"\b(?:including|include|includes|with)\b", r"\b(?:and|or)\b"):
        next_pending: list[str] = []
        for chunk in pending:
            if _word_count(chunk) <= max_words:
                next_pending.append(chunk)
                continue
            next_pending.extend(part.strip() for part in re.split(pattern, chunk, flags=re.IGNORECASE) if part.strip())
        pending = next_pending

    for chunk in pending:
        cleaned = _trim_short_concept_tail(
            re.sub(r"^(?:including|include|includes|and|or|with)\s+", "", chunk, flags=re.IGNORECASE).strip()
        )
        if not cleaned:
            continue
        if _is_instruction_like_sam3_text(cleaned):
            continue
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = _trim_short_concept_tail(" ".join(words[:max_words]))
        if cleaned and 1 <= _word_count(cleaned) <= max_words:
            _append_unique_concept(candidates, cleaned)
    return tuple(candidates)


def _append_short_concept(concepts: list[str], phrase: str, *, max_words: int = 8) -> None:
    for candidate in _short_concept_candidates(phrase, max_words=max_words):
        _append_unique_concept(concepts, candidate)


def _split_short_concept_hints(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    concepts: list[str] = []
    for phrase in re.split(r"[;,\n]+", text):
        _append_short_concept(concepts, phrase)
    return tuple(concepts)


def build_sam3_concepts_from_agent_prompt(sam3_prompt: str) -> tuple[str, ...]:
    concepts = _split_short_concept_hints(sam3_prompt)
    if not concepts:
        raise ValueError("agent-provided SAM3 prompt is empty or unusable")
    return tuple(concepts[:MAX_SAM3_CONCEPTS])


def write_sam3_prompt_artifact(request: Sam3OmniPart2DSegmentationRequest, processed_rgba: Image.Image, image_metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned_image_bytes = request.image_path.read_bytes()
    cleaned_image_sha256 = _sha256_bytes(cleaned_image_bytes)
    object_payload = _load_object_description(request.object_description_path)
    object_description = str(object_payload.get("object_description", "") or "")
    visual_summary = build_cleaned_image_visual_summary(
        image_sha256=cleaned_image_sha256,
        image_metadata=image_metadata,
        processed_rgba=processed_rgba,
    )
    sam3_prompt_concepts = build_sam3_concepts_from_agent_prompt(request.sam3_prompt)
    generated_prompt = ", ".join(sam3_prompt_concepts)
    prompt_inputs = {
        "cleaned_image_path": str(request.image_path),
        "cleaned_image_sha256": cleaned_image_sha256,
        "cleaned_image_metadata": image_metadata,
        "object_description_path": str(request.object_description_path) if request.object_description_path else None,
        "object_description": object_description,
        "visual_summary": visual_summary,
        "sam3_prompt": request.sam3_prompt,
    }
    payload = {
        "schema_version": SAM3_PROMPT_SCHEMA_VERSION,
        "object_name": request.object_name,
        "cleaned_image_path": str(request.image_path),
        "cleaned_image_sha256": cleaned_image_sha256,
        "cleaned_image_size": image_metadata["cleaned_image_size"],
        "cleaned_image_mode": image_metadata["cleaned_image_mode"],
        "cleaned_image_foreground_bbox_xyxy": image_metadata["cleaned_image_foreground_bbox_xyxy"],
        "cleaned_image_visual_summary": visual_summary,
        "object_description_path": str(request.object_description_path) if request.object_description_path else None,
        "object_description": object_description,
        "sam3_prompt": request.sam3_prompt,
        "sam3_prompt_concepts": list(sam3_prompt_concepts),
        "generated_prompt": generated_prompt,
        "prompt_author": "stage-agent",
        "prompt_stage_effort": "stage-agent",
        "prompt_generation_method": "agent-provided-short-concept-phrases",
        "prompt_inputs_sha256": _sha256_bytes(json.dumps(to_json_dict(prompt_inputs), sort_keys=True).encode("utf-8")),
    }
    _write_json(request.sam3_prompt_path, payload)
    return payload


def execute_sam3_image_prompt(
    image: Image.Image,
    prompt: str,
    *,
    sam3_root: Path | None = None,
) -> dict[str, Any]:
    return execute_sam3_image_prompt_concepts(image, (prompt,), sam3_root=sam3_root)[0]


def execute_sam3_image_prompt_concepts(
    image: Image.Image,
    concepts: tuple[str, ...],
    *,
    sam3_root: Path | None = None,
) -> list[dict[str, Any]]:
    if sam3_root is not None and sam3_root.exists():
        sys.path.insert(0, str(sam3_root))
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "SAM3 is not importable in the segmentation environment. Install third_party/sam3 "
            "and authenticate/download gated checkpoints before running this stage."
        ) from exc

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = os.environ.get(SAM3_CHECKPOINT_ENV_VAR)
        if checkpoint_path:
            resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            if not resolved_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"{SAM3_CHECKPOINT_ENV_VAR} does not exist: {resolved_checkpoint_path}"
                )
            model = build_sam3_image_model(
                checkpoint_path=str(resolved_checkpoint_path),
                device=device,
            )
        else:
            model = build_sam3_image_model(device=device)
        processor = Sam3Processor(model, device=device)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            state = processor.set_image(image)
            outputs = []
            reset_all_prompts = getattr(processor, "reset_all_prompts", None)
            for concept_index, concept in enumerate(concepts):
                if concept_index > 0 and callable(reset_all_prompts):
                    reset_all_prompts(state)
                outputs.append(dict(processor.set_text_prompt(state=state, prompt=concept)))
            return outputs
    except Exception as exc:
        raise RuntimeError(
            "SAM3 inference failed, likely because the gated SAM3 checkpoints are unavailable "
            "or the segmentation environment is incomplete. No fallback segmentation model was invoked."
        ) from exc


def _tensor_like_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        tensor = value.detach()
        if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            tensor = tensor.float()
        value = tensor.cpu().numpy()
    return np.asarray(value)


def _resize_mask(mask: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    if mask.shape == (size, size):
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(pil.resize((size, size), Image.Resampling.NEAREST)) > 0


def normalize_sam3_output(output: dict[str, Any], *, target_size: int = TARGET_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = _tensor_like_to_numpy(output.get("masks", np.zeros((0, 1, target_size, target_size), dtype=bool)))
    boxes = _tensor_like_to_numpy(output.get("boxes", np.zeros((0, 4), dtype=np.float32))).astype(np.float32)
    scores = _tensor_like_to_numpy(output.get("scores", np.zeros((0,), dtype=np.float32))).astype(np.float32).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, None, :, :]
    elif masks.ndim == 3:
        masks = masks[:, None, :, :]
    elif masks.ndim != 4:
        raise ValueError(f"SAM3 masks must have rank 2, 3, or 4, got {masks.shape}")
    resized = np.stack([_resize_mask(mask.squeeze(), target_size) for mask in masks], axis=0) if len(masks) else np.zeros((0, target_size, target_size), dtype=bool)
    return resized[:, None, :, :].astype(bool), boxes.reshape((-1, 4)), scores


def _normalize_sam3_concept_outputs(
    raw_outputs_by_concept: tuple[dict[str, Any], ...],
    concepts: tuple[str, ...],
    *,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    mask_parts: list[np.ndarray] = []
    box_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    mask_provenance: list[dict[str, Any]] = []
    for concept_index, (concept, raw_output) in enumerate(zip(concepts, raw_outputs_by_concept)):
        masks, boxes_xyxy, scores = normalize_sam3_output(raw_output, target_size=target_size)
        mask_parts.append(masks)
        box_parts.append(boxes_xyxy)
        score_parts.append(scores)
        for local_mask_index in range(int(masks.shape[0])):
            mask_provenance.append(
                {
                    "source_concept_index": int(concept_index),
                    "source_concept": concept,
                    "source_concept_mask_index": int(local_mask_index),
                }
            )
    if not mask_parts:
        return (
            np.zeros((0, 1, target_size, target_size), dtype=bool),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            mask_provenance,
        )
    return (
        np.concatenate(mask_parts, axis=0).astype(bool),
        np.concatenate(box_parts, axis=0).astype(np.float32),
        np.concatenate(score_parts, axis=0).astype(np.float32),
        mask_provenance,
    )


def _attach_sam3_concept_provenance(
    mask_records: list[dict[str, Any]],
    mask_provenance: list[dict[str, Any]],
) -> None:
    for record in mask_records:
        record.update(mask_provenance[int(record["source_mask_index"])])


def _label_connected_components(binary_mask: np.ndarray) -> tuple[int, np.ndarray]:
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(binary_mask.astype(bool), structure=structure)
    return int(count), labels


def split_disconnected_parts(group_ids: np.ndarray, *, size_threshold: int = DEFAULT_SIZE_THRESHOLD) -> np.ndarray:
    output = np.full(group_ids.shape, -1, dtype=np.int32)
    next_id = 0
    for part_id in [int(value) for value in np.unique(group_ids) if value >= 0]:
        count, labels = _label_connected_components(group_ids == part_id)
        for label in range(1, count + 1):
            region = labels == label
            if int(region.sum()) >= max(1, size_threshold // 5):
                output[region] = next_id
                next_id += 1
    return output


def fill_missed_foreground(group_ids: np.ndarray, alpha: np.ndarray, *, size_threshold: int = DEFAULT_SIZE_THRESHOLD) -> np.ndarray:
    filled = group_ids.copy()
    missed = (alpha > 10) & (filled < 0)
    count, labels = _label_connected_components(missed)
    next_id = int(filled.max()) + 1 if np.any(filled >= 0) else 0
    for label in range(1, count + 1):
        region = labels == label
        if int(region.sum()) >= size_threshold:
            filled[region] = next_id
            next_id += 1
    return filled


def clean_segment_edges(group_ids: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    cleaned = np.full(group_ids.shape, -1, dtype=np.int32)
    structure = np.ones((3, 3), dtype=bool)
    for part_id in [int(value) for value in np.unique(group_ids) if value >= 0]:
        mask = group_ids == part_id
        mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        cleaned[mask] = part_id
    return cleaned


def reindex_group_ids(group_ids: np.ndarray) -> np.ndarray:
    output = np.full(group_ids.shape, -1, dtype=np.int32)
    for new_id, old_id in enumerate([int(value) for value in np.unique(group_ids) if value >= 0]):
        output[group_ids == old_id] = new_id
    return output


@dataclass(frozen=True)
class _Sam3PostprocessResult:
    group_ids: np.ndarray
    mask_records: list[dict[str, Any]]
    part_sources: dict[int, dict[str, Any]]
    postprocess_summary: dict[str, Any]


def _unknown_part_source() -> dict[str, Any]:
    return {"source": "unknown", "is_segmentation_evidence": False}


def _part_source(part_sources: dict[int, dict[str, Any]], part_id: int) -> dict[str, Any]:
    return dict(part_sources.get(part_id, _unknown_part_source()))


def _split_disconnected_parts_with_sources(
    group_ids: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
    *,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    output = np.full(group_ids.shape, -1, dtype=np.int32)
    output_sources: dict[int, dict[str, Any]] = {}
    next_id = 0
    for part_id in [int(value) for value in np.unique(group_ids) if value >= 0]:
        count, labels = _label_connected_components(group_ids == part_id)
        for label in range(1, count + 1):
            region = labels == label
            if int(region.sum()) >= max(1, size_threshold // 5):
                output[region] = next_id
                output_sources[next_id] = _part_source(part_sources, part_id)
                next_id += 1
    return output, output_sources


def _fill_missed_foreground_with_sources(
    group_ids: np.ndarray,
    alpha: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
    *,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    filled = group_ids.copy()
    output_sources = {
        int(part_id): _part_source(part_sources, int(part_id))
        for part_id in np.unique(filled)
        if int(part_id) >= 0
    }
    missed = (alpha > 10) & (filled < 0)
    count, labels = _label_connected_components(missed)
    next_id = int(filled.max()) + 1 if np.any(filled >= 0) else 0
    for label in range(1, count + 1):
        region = labels == label
        if int(region.sum()) >= size_threshold:
            filled[region] = next_id
            output_sources[next_id] = {
                "source": "foreground_fill_fallback",
                "is_segmentation_evidence": False,
            }
            next_id += 1
    return filled, output_sources


def _drop_sources_without_pixels(
    group_ids: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        int(part_id): _part_source(part_sources, int(part_id))
        for part_id in np.unique(group_ids)
        if int(part_id) >= 0
    }


def _reindex_group_ids_with_sources(
    group_ids: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    output = np.full(group_ids.shape, -1, dtype=np.int32)
    output_sources: dict[int, dict[str, Any]] = {}
    for new_id, old_id in enumerate([int(value) for value in np.unique(group_ids) if value >= 0]):
        output[group_ids == old_id] = new_id
        output_sources[new_id] = _part_source(part_sources, old_id)
    return output, output_sources


def _build_postprocess_summary(
    *,
    post_sam3_score_threshold: float,
    sam3_pre_postprocess_mask_count: int,
    mask_records: list[dict[str, Any]],
    group_ids: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    final_part_ids = [int(value) for value in np.unique(group_ids) if value >= 0]
    fallback_count = sum(
        1 for part_id in final_part_ids if _part_source(part_sources, part_id)["source"] == "foreground_fill_fallback"
    )
    evidence_count = sum(
        1 for part_id in final_part_ids if bool(_part_source(part_sources, part_id)["is_segmentation_evidence"])
    )
    kept_count = sum(1 for record in mask_records if bool(record["kept_after_alpha_filter"]))
    return {
        "score_threshold_scope": SCORE_THRESHOLD_SCOPE,
        "post_sam3_score_threshold": post_sam3_score_threshold,
        "sam3_pre_postprocess_mask_count": sam3_pre_postprocess_mask_count,
        "sam3_pre_postprocess_zero_masks": sam3_pre_postprocess_mask_count == 0,
        "sam3_masks_kept_after_postprocess_count": kept_count,
        "sam3_masks_rejected_after_postprocess_count": sam3_pre_postprocess_mask_count - kept_count,
        "foreground_fill_fallback_used": fallback_count > 0,
        "foreground_fill_fallback_part_count": fallback_count,
        "final_part_count": len(final_part_ids),
        "final_segmentation_evidence_part_count": evidence_count,
    }


def _build_group_ids_from_sam3_with_metadata(
    *,
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    alpha: np.ndarray,
    post_sam3_score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> _Sam3PostprocessResult:
    group_ids = np.full(alpha.shape, -1, dtype=np.int32)
    mask_records: list[dict[str, Any]] = []
    part_sources: dict[int, dict[str, Any]] = {}
    flat_masks = masks[:, 0] if masks.ndim == 4 else masks
    areas = flat_masks.reshape((flat_masks.shape[0], -1)).sum(axis=1) if len(flat_masks) else np.zeros((0,))
    order = list(np.argsort(-areas))
    group_counter = 0
    background = alpha <= 10
    for mask_index in order:
        mask = flat_masks[mask_index].astype(bool)
        area = int(mask.sum())
        score = float(scores[mask_index]) if mask_index < len(scores) else 0.0
        background_ratio = float(np.sum(mask & background) / max(area, 1))
        keep = bool(
            area >= size_threshold
            and score >= post_sam3_score_threshold
            and background_ratio <= MAX_SAM3_BACKGROUND_RATIO
        )
        assigned_group_id = group_counter if keep else None
        if keep:
            group_ids[mask & (alpha > 10)] = group_counter
            part_sources[group_counter] = {
                "source": "sam3",
                "is_segmentation_evidence": True,
            }
            group_counter += 1
        box = boxes_xyxy[mask_index].astype(float).tolist() if mask_index < len(boxes_xyxy) else [0.0, 0.0, 0.0, 0.0]
        mask_records.append(
            {
                "source_mask_index": int(mask_index),
                "score": score,
                "box_xyxy": box,
                "area_px": area,
                "background_ratio": background_ratio,
                "kept_after_alpha_filter": keep,
                "assigned_group_id": assigned_group_id,
                "source": "sam3",
                "is_segmentation_evidence": keep,
            }
        )
    group_ids, part_sources = _split_disconnected_parts_with_sources(
        group_ids,
        part_sources,
        size_threshold=size_threshold,
    )
    group_ids, part_sources = _fill_missed_foreground_with_sources(
        group_ids,
        alpha,
        part_sources,
        size_threshold=size_threshold,
    )
    group_ids = clean_segment_edges(group_ids)
    part_sources = _drop_sources_without_pixels(group_ids, part_sources)
    group_ids, part_sources = _reindex_group_ids_with_sources(group_ids, part_sources)
    postprocess_summary = _build_postprocess_summary(
        post_sam3_score_threshold=post_sam3_score_threshold,
        sam3_pre_postprocess_mask_count=int(flat_masks.shape[0]),
        mask_records=mask_records,
        group_ids=group_ids,
        part_sources=part_sources,
    )
    return _Sam3PostprocessResult(
        group_ids=group_ids,
        mask_records=mask_records,
        part_sources=part_sources,
        postprocess_summary=postprocess_summary,
    )


def build_group_ids_from_sam3(
    *,
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    alpha: np.ndarray,
    post_sam3_score_threshold: float | None = None,
    score_threshold: float | None = None,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    resolved_post_sam3_score_threshold = _resolve_post_sam3_score_threshold(
        post_sam3_score_threshold=post_sam3_score_threshold,
        score_threshold=score_threshold,
    )
    result = _build_group_ids_from_sam3_with_metadata(
        masks=masks,
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        alpha=alpha,
        post_sam3_score_threshold=resolved_post_sam3_score_threshold,
        size_threshold=size_threshold,
    )
    return result.group_ids, result.mask_records


def save_mask_exr(path: Path, group_ids: np.ndarray) -> np.ndarray:
    mask_values = (group_ids + 1).astype(np.float32)
    mask_exr = np.repeat(mask_values[:, :, None], 3, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    if not cv2.imwrite(str(path), mask_exr):
        raise RuntimeError(f"OpenCV failed to write EXR mask: {path}")
    return mask_exr


def smart_downsample_mask(mask: np.ndarray, target_size: tuple[int, int] = (ORDERED_MASK_SIZE, ORDERED_MASK_SIZE)) -> np.ndarray:
    h, w = mask.shape[:2]
    target_h, target_w = target_size
    h_ratio = h / target_h
    w_ratio = w / target_w
    downsampled = np.zeros((target_h, target_w), dtype=mask.dtype)
    for row in range(target_h):
        for col in range(target_w):
            y_start = int(row * h_ratio)
            y_end = min(int((row + 1) * h_ratio), h)
            x_start = int(col * w_ratio)
            x_end = min(int((col + 1) * w_ratio), w)
            region = mask[y_start:y_end, x_start:x_end]
            values, counts = np.unique(region.reshape(-1), return_counts=True)
            foreground = values > 0
            if np.any(foreground):
                fg_values = values[foreground]
                fg_counts = counts[foreground]
                downsampled[row, col] = fg_values[int(np.argmax(fg_counts))]
            else:
                downsampled[row, col] = values[int(np.argmax(counts))]
    return downsampled


def build_ordered_mask(group_ids: np.ndarray) -> np.ndarray:
    mask_input = smart_downsample_mask((group_ids + 1).astype(np.int64))
    part_positions: dict[int, int] = {}
    for idx in [int(value) for value in np.unique(mask_input) if value > 0]:
        y_coords, _ = np.where(mask_input == idx)
        if len(y_coords) > 0:
            part_positions[idx] = int(np.max(y_coords))
    sorted_parts = sorted(part_positions.items(), key=lambda item: -item[1])
    index_map = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_parts, 1)}
    ordered = np.zeros_like(mask_input, dtype=np.int64)
    for old_idx, new_idx in index_map.items():
        ordered[mask_input == old_idx] = new_idx
    return ordered.astype(np.int64)


def _palette(index: int) -> np.ndarray:
    colors = np.array(
        [
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105],
            [252, 205, 229],
            [188, 128, 189],
            [102, 194, 165],
            [252, 141, 98],
            [141, 160, 203],
        ],
        dtype=np.uint8,
    )
    return colors[index % len(colors)]


def colorize_mask(mask: np.ndarray, *, size: tuple[int, int] = (TARGET_SIZE, TARGET_SIZE)) -> Image.Image:
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for part_id in [int(value) for value in np.unique(mask) if value > 0]:
        image[mask == part_id] = _palette(part_id - 1)
    pil = Image.fromarray(image, mode="RGB")
    return pil.resize(size, Image.Resampling.NEAREST) if pil.size != size else pil


def overlay_group_ids(image: Image.Image, group_ids: np.ndarray, *, alpha: float = 0.42) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for part_id in [int(value) for value in np.unique(group_ids) if value >= 0]:
        mask = group_ids == part_id
        color = _palette(part_id).astype(np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _part_records(
    group_ids: np.ndarray,
    *,
    part_sources: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    part_sources = part_sources or {}
    for part_id in [int(value) for value in np.unique(group_ids) if value >= 0]:
        mask = group_ids == part_id
        ys, xs = np.where(mask)
        source_metadata = _part_source(part_sources, part_id)
        records.append(
            {
                "part_id": part_id,
                "pixel_count": int(mask.sum()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "source": source_metadata["source"],
                "is_segmentation_evidence": bool(source_metadata["is_segmentation_evidence"]),
            }
        )
    return records


def _write_raw_sam3_artifacts(
    request: Sam3OmniPart2DSegmentationRequest,
    *,
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    prompt_payload: dict[str, Any],
    mask_records: list[dict[str, Any]],
    postprocess_summary: dict[str, Any],
) -> dict[str, Any]:
    request.sam3_raw_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        request.sam3_raw_npz_path,
        masks=masks.astype(bool),
        boxes_xyxy=boxes_xyxy.astype(np.float32),
        scores=scores.astype(np.float32),
        mask_source_size=np.array([TARGET_SIZE, TARGET_SIZE], dtype=np.int32),
    )
    metadata = {
        "schema_version": SAM3_RAW_SCHEMA_VERSION,
        "sam3_repo": str(request.sam3_root),
        "sam3_branch": read_git_branch(request.sam3_root),
        "sam3_checkpoint": "default_build_sam3_image_model",
        "generated_prompt": prompt_payload["generated_prompt"],
        "sam3_execution_mode": "per_concept_prompt",
        "sam3_prompt_concepts": list(prompt_payload["sam3_prompt_concepts"]),
        "post_sam3_score_threshold": request.post_sam3_score_threshold,
        "score_threshold": request.post_sam3_score_threshold,
        "score_threshold_scope": SCORE_THRESHOLD_SCOPE,
        "sam3_raw_output_stage": SAM3_RAW_OUTPUT_STAGE,
        "sam3_pre_postprocess_mask_count": int(masks.shape[0]),
        "sam3_pre_postprocess_zero_masks": int(masks.shape[0]) == 0,
        "mask_count": int(masks.shape[0]),
        "postprocess_summary": postprocess_summary,
        "box_format": "xyxy_pixel",
        "masks": mask_records,
    }
    _write_json(request.sam3_raw_metadata_path, metadata)
    return metadata


def _write_manifest(
    request: Sam3OmniPart2DSegmentationRequest,
    *,
    prompt_payload: dict[str, Any],
    raw_metadata: dict[str, Any],
    raw_mask_shape: tuple[int, ...],
    group_ids: np.ndarray,
    part_sources: dict[int, dict[str, Any]],
    postprocess_summary: dict[str, Any],
) -> dict[str, Any]:
    part_records = _part_records(group_ids, part_sources=part_sources)
    manifest = {
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "stage_name": SAM3_OMNIPART_2D_SEGMENTATION_STAGE,
        "object_name": request.object_name,
        "source_image": str(request.image_path),
        "cleaned_image_path": str(request.image_path),
        "cleaned_image_sha256": prompt_payload["cleaned_image_sha256"],
        "sam3_prompt_path": str(request.sam3_prompt_path),
        "sam3_raw_npz_path": str(request.sam3_raw_npz_path),
        "sam3_raw_metadata_path": str(request.sam3_raw_metadata_path),
        "processed_rgba_path": str(request.processed_rgba_path),
        "processed_white_bg_png_path": str(request.processed_white_bg_png_path),
        "processed_black_bg_png_path": str(request.processed_black_bg_png_path),
        "image_white_bg_tensor_path": str(request.image_white_bg_tensor_path),
        "image_black_bg_tensor_path": str(request.image_black_bg_tensor_path),
        "group_ids_path": str(request.group_ids_path),
        "mask_exr_path": str(request.mask_exr_path),
        "ordered_mask_input_path": str(request.ordered_mask_input_path),
        "ordered_mask_vis_path": str(request.ordered_mask_vis_path),
        "segmentation_overlay_path": str(request.segmentation_overlay_path),
        "segmentation_manifest_path": str(request.segmentation_manifest_path),
        "post_sam3_score_threshold": request.post_sam3_score_threshold,
        "score_threshold": request.post_sam3_score_threshold,
        "score_threshold_scope": SCORE_THRESHOLD_SCOPE,
        "sam3_raw_output_stage": SAM3_RAW_OUTPUT_STAGE,
        "sam3_pre_postprocess_mask_count": raw_metadata["sam3_pre_postprocess_mask_count"],
        "sam3_pre_postprocess_zero_masks": raw_metadata["sam3_pre_postprocess_zero_masks"],
        "postprocess_summary": postprocess_summary,
        "label_conventions": {
            "group_ids": "background=-1, foreground parts=0..N-1",
            "ordered_mask_input": "background=0, bottom-up foreground parts=1..M",
        },
        "exr_channel_semantics": "float32 RGB channels all store group_ids + 1",
        "tensor_shapes": {
            "sam3_raw_masks": [int(value) for value in raw_mask_shape],
            "image_white_bg_tensor": [3, TARGET_SIZE, TARGET_SIZE],
            "image_black_bg_tensor": [3, TARGET_SIZE, TARGET_SIZE],
            "group_ids": [TARGET_SIZE, TARGET_SIZE],
            "ordered_mask_input": [ORDERED_MASK_SIZE, ORDERED_MASK_SIZE],
        },
        "tensor_dtypes": {
            "sam3_raw_masks": "bool",
            "boxes_xyxy": "float32",
            "scores": "float32",
            "image_tensors": "float32",
            "group_ids": "int32",
            "ordered_mask_input": "int64",
            "mask_exr": "float32",
        },
        "part_count": len(part_records),
        "final_part_count": len(part_records),
        "parts": part_records,
        "sam3_metadata_path": str(request.sam3_raw_metadata_path),
        "prompt_path": str(request.sam3_prompt_path),
        "hag4r_branch": read_git_branch(_repo_root()),
        "sam3_branch": read_git_branch(request.sam3_root),
        "conda_env": EnvName.SEGMENTATION,
        "conda_env_prefix": resolve_conda_env(EnvName.SEGMENTATION),
        "omnipart_step16_inputs": {
            "image_white_bg_tensor_path": str(request.image_white_bg_tensor_path),
            "image_black_bg_tensor_path": str(request.image_black_bg_tensor_path),
            "ordered_mask_input_path": str(request.ordered_mask_input_path),
        },
    }
    _write_json(request.segmentation_manifest_path, manifest)
    return manifest


def execute_sam3_omnipart_2d_segmentation(
    request: Sam3OmniPart2DSegmentationRequest,
    *,
    sam3_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    processed_rgba, image_metadata = preprocess_cleaned_image(request.image_path)
    processed_rgba.save(request.processed_rgba_path)
    white_bg = composite_rgba(processed_rgba, (255, 255, 255))
    black_bg = composite_rgba(processed_rgba, (0, 0, 0))
    white_bg.save(request.processed_white_bg_png_path)
    black_bg.save(request.processed_black_bg_png_path)
    np.save(request.image_white_bg_tensor_path, image_to_chw_float(white_bg))
    np.save(request.image_black_bg_tensor_path, image_to_chw_float(black_bg))

    prompt_payload = write_sam3_prompt_artifact(request, processed_rgba, image_metadata)
    concepts = tuple(prompt_payload["sam3_prompt_concepts"])
    if sam3_runner is not None:
        raw_outputs_by_concept = tuple(
            sam3_runner(white_bg, concept, sam3_root=request.sam3_root) for concept in concepts
        )
    else:
        raw_outputs_by_concept = tuple(
            execute_sam3_image_prompt_concepts(white_bg, concepts, sam3_root=request.sam3_root)
        )
    masks, boxes_xyxy, scores, mask_provenance = _normalize_sam3_concept_outputs(raw_outputs_by_concept, concepts)
    alpha = np.asarray(processed_rgba.getchannel("A"), dtype=np.uint8)
    post_result = _build_group_ids_from_sam3_with_metadata(
        masks=masks,
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        alpha=alpha,
        post_sam3_score_threshold=request.post_sam3_score_threshold,
        size_threshold=request.size_threshold,
    )
    _attach_sam3_concept_provenance(post_result.mask_records, mask_provenance)
    raw_metadata = _write_raw_sam3_artifacts(
        request,
        masks=masks,
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        prompt_payload=prompt_payload,
        mask_records=post_result.mask_records,
        postprocess_summary=post_result.postprocess_summary,
    )
    np.save(request.group_ids_path, post_result.group_ids.astype(np.int32))
    save_mask_exr(request.mask_exr_path, post_result.group_ids)
    ordered_mask = build_ordered_mask(post_result.group_ids)
    np.save(request.ordered_mask_input_path, ordered_mask.astype(np.int64))
    colorize_mask(ordered_mask).save(request.ordered_mask_vis_path)
    overlay_group_ids(white_bg, post_result.group_ids).save(request.segmentation_overlay_path)
    manifest = _write_manifest(
        request,
        prompt_payload=prompt_payload,
        raw_metadata=raw_metadata,
        raw_mask_shape=tuple(int(value) for value in masks.shape),
        group_ids=post_result.group_ids,
        part_sources=post_result.part_sources,
        postprocess_summary=post_result.postprocess_summary,
    )
    payload = to_json_dict(request)
    payload["status"] = "executed"
    payload["manifest"] = manifest
    payload["score_threshold"] = request.post_sam3_score_threshold
    payload["score_threshold_scope"] = SCORE_THRESHOLD_SCOPE
    payload["sam3_raw_output_stage"] = SAM3_RAW_OUTPUT_STAGE
    payload["sam3_pre_postprocess_mask_count"] = raw_metadata["sam3_pre_postprocess_mask_count"]
    payload["sam3_pre_postprocess_zero_masks"] = raw_metadata["sam3_pre_postprocess_zero_masks"]
    payload["postprocess_summary"] = post_result.postprocess_summary
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM3-owned OmniPart Step 16 segmentation artifacts.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam3_prompt", "--prompt", dest="sam3_prompt", required=True)
    parser.add_argument("--object_description_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_name")
    parser.add_argument("--sam3_root", default="third_party/sam3")
    parser.add_argument(
        "--post_sam3_score_threshold",
        type=float,
        default=None,
        help=(
            "Post-SAM3 mask score filter applied after SAM3 returns raw masks; "
            "does not change SAM3 internal retrieval or proposal thresholds."
        ),
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help=(
            "Compatibility alias for --post_sam3_score_threshold; post-SAM3 only, "
            "not a SAM3 internal retrieval threshold."
        ),
    )
    parser.add_argument("--size_threshold", type=int, default=DEFAULT_SIZE_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        post_sam3_score_threshold = _resolve_post_sam3_score_threshold(
            post_sam3_score_threshold=args.post_sam3_score_threshold,
            score_threshold=args.score_threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))
    request = build_sam3_omnipart_2d_segmentation_request(
        image_path=Path(args.image),
        sam3_prompt=args.sam3_prompt,
        output_dir=Path(args.output_dir),
        object_description_path=Path(args.object_description_path) if args.object_description_path else None,
        object_name=args.object_name,
        sam3_root=Path(args.sam3_root),
        post_sam3_score_threshold=post_sam3_score_threshold,
        size_threshold=args.size_threshold,
    )
    payload = execute_sam3_omnipart_2d_segmentation(request)
    print(json.dumps(to_json_dict(payload), indent=2, sort_keys=True))
    return 0


def run_sam3_omnipart_2d_segmentation(
    image_path: Path,
    sam3_prompt: str,
    output_dir: Path,
    *,
    object_description_path: Path | None = None,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> StageRunResult:
    root = repo_root or _tools_repo_root()
    request = build_sam3_omnipart_2d_segmentation_request(
        image_path=image_path,
        sam3_prompt=sam3_prompt,
        output_dir=output_dir,
        object_description_path=object_description_path,
    )
    args: list[object] = [
        "--image",
        request.image_path,
        "--sam3_prompt",
        request.sam3_prompt,
        "--output_dir",
        request.output_dir,
        "--object_name",
        request.object_name,
        "--sam3_root",
        request.sam3_root,
        "--post_sam3_score_threshold",
        request.post_sam3_score_threshold,
        "--size_threshold",
        request.size_threshold,
    ]
    if request.object_description_path is not None:
        args.extend(["--object_description_path", request.object_description_path])
    return _run_subprocess_stage(
        name=SAM3_OMNIPART_2D_SEGMENTATION_STAGE,
        stage=Stage.SEGMENTATION,
        argv=_conda_module_argv(EnvName.SEGMENTATION, "hag4r.tools.segmentation", args),
        cwd=root,
        conda_env=resolve_conda_env(EnvName.SEGMENTATION),
        timeout_s=3600,
        expected_artifacts=tuple(
            _artifact(
                ArtifactRole.REPORT
                if path.suffix == ".json"
                else ArtifactRole.SEGMENTATION_MASK
                if path.suffix in {".npy", ".npz", ".exr"}
                else ArtifactRole.SEGMENTED_VIEW,
                path,
                Stage.SEGMENTATION,
            )
            for path in request.expected_outputs
        ),
        read_paths=(
            request.image_path,
            request.sam3_root,
            *((request.object_description_path,) if request.object_description_path else ()),
        ),
        write_paths=(request.output_dir,),
        env=env,
        log_dir=log_dir,
        repo_root=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ORDERED_MASK_SIZE",
    "SAM3_RAW_OUTPUT_STAGE",
    "SAM3_CHECKPOINT_ENV_VAR",
    "SAM3_OMNIPART_2D_SEGMENTATION_STAGE",
    "SCORE_THRESHOLD_SCOPE",
    "SEGMENTATION_SCHEMA_VERSION",
    "TARGET_SIZE",
    "Sam3OmniPart2DSegmentationRequest",
    "build_group_ids_from_sam3",
    "build_ordered_mask",
    "build_sam3_concepts_from_agent_prompt",
    "build_sam3_omnipart_2d_segmentation_request",
    "execute_sam3_image_prompt",
    "execute_sam3_image_prompt_concepts",
    "execute_sam3_omnipart_2d_segmentation",
    "preprocess_cleaned_image",
    "run_sam3_omnipart_2d_segmentation",
    "sam3_omnipart_2d_segmentation_request_payload",
    "smart_downsample_mask",
]
