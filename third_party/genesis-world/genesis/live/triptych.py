from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


PANEL_ORDER = ("top", "northeast", "southwest")
HAG4R_LABELS = {"top": "top", "northeast": "ne_3q", "southwest": "sw_3q"}


def png_record(path: Path, *, label: str, hag4r_label: str, frame_index: int, simulation_step: int) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        width, height = image.size
    return {
        "label": label,
        "hag4r_label": hag4r_label,
        "frame_index": int(frame_index),
        "simulation_step": int(simulation_step),
        "path": str(path),
        "width": int(width),
        "height": int(height),
        "byte_size": int(path.stat().st_size),
        "sha256": digest,
    }


def stitch_triptych(panel_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    try:
        widths, heights = zip(*(image.size for image in images), strict=True)
        canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
        x = 0
        for image in images:
            canvas.paste(image, (x, 0))
            x += image.size[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
