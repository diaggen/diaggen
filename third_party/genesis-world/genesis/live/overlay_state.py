from __future__ import annotations

from typing import Any


def anchor_overlay_records(anchor_records: dict[str, list[Any]]) -> list[dict[str, Any]]:
    records = []
    for entity_name, anchors in anchor_records.items():
        for anchor in anchors:
            data = anchor.to_dict() if hasattr(anchor, "to_dict") else dict(anchor)
            data["entity"] = entity_name
            data["kind"] = "static_anchor"
            records.append(data)
    return records


def controller_overlay_records(controllers: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for controller_id, controller in controllers.items():
        data = controller.snapshot()
        data["controller_id"] = controller_id
        data["kind"] = "live_box_controller"
        records.append(data)
    return records
