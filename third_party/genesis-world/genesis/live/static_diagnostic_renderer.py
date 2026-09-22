from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import igl
import numpy as np
import trimesh
from PIL import Image

import genesis as gs
import genesis.utils.mesh as mu
from genesis.ext import pyrender
from genesis.ext.pyrender.jit_render import JITRenderer
from genesis.vis.rasterizer_context import (
    PART_SHADED_CREASE_DEGREES,
    _crease_surface_corner_normals,
    _part_segmentation_wireframe_box_mesh,
)

REQUEST_SCHEMA = "hag4r-genesis-diagnostic-static-render-request-v1"
RESULT_SCHEMA = "genesis-diagnostic-static-render-result-v1"


def _z_up_rotation(z: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    z = z / np.linalg.norm(z)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(z[2])) < 0.99 else np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
    x = np.cross(np.asarray(up, dtype=np.float32), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _load_surface(asset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_path = Path(asset["mesh_path"])
    primitive_kind = asset["primitive_kind"]
    if primitive_kind == "tetrahedron":
        vertices, tets, _faces = igl.readMESH(str(mesh_path))
        tets = np.asarray(tets, dtype=np.int64)
        tet_faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        )
        all_faces = tets[:, tet_faces].reshape((-1, 3))
        with np.load(asset["primitive_labels_path"], allow_pickle=False) as payload:
            primitive_labels = np.asarray(payload[asset["primitive_labels_key"]], dtype=np.int64)
        all_labels = np.repeat(primitive_labels, 4)
        canonical = np.sort(all_faces, axis=1)
        _unique, first, counts = np.unique(canonical, axis=0, return_index=True, return_counts=True)
        boundary = first[counts == 1]
        faces = all_faces[boundary]
        face_labels = all_labels[boundary]
    elif primitive_kind == "triangle":
        mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        with np.load(asset["primitive_labels_path"], allow_pickle=False) as payload:
            face_labels = np.asarray(payload[asset["primitive_labels_key"]], dtype=np.int64)
    else:
        raise ValueError(f"unsupported static diagnostic primitive kind: {primitive_kind}")

    transform = np.asarray(asset["mesh_to_env_local"], dtype=np.float64)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    vertices = (homogeneous @ transform.T)[:, :3].astype(np.float32)
    return vertices, np.asarray(faces, dtype=np.int64), np.asarray(face_labels, dtype=np.int64)


def _material(color_rgb: list[int], *, metallic: float, roughness: float):
    color = np.asarray(color_rgb, dtype=np.float32) / 255.0
    return pyrender.MetallicRoughnessMaterial(
        alphaMode="OPAQUE",
        baseColorFactor=(*color.tolist(), 1.0),
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )


def _add_part_nodes(
    scene: pyrender.Scene,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    parts: list[dict[str, Any]],
    surface: dict[str, Any],
) -> tuple[list[Any], dict[Any, np.ndarray]]:
    corner_normals = _crease_surface_corner_normals(vertices, faces).reshape((-1, 3))
    part_nodes = []
    segmentation_colors = {}
    for part in parts:
        part_id = int(part["part_id"])
        face_indices = np.flatnonzero(face_labels == part_id)
        if len(face_indices) == 0:
            continue
        source_vertices = faces[face_indices].reshape(-1)
        local_faces = np.arange(len(source_vertices), dtype=np.int64).reshape((-1, 3))
        corner_indices = (face_indices[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
        mesh = trimesh.Trimesh(
            vertices=vertices[source_vertices],
            faces=local_faces,
            vertex_normals=corner_normals[corner_indices],
            process=False,
        )
        render_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=True,
            double_sided=True,
            material=_material(
                part["part_color_rgb"],
                metallic=float(surface["metallic"]),
                roughness=float(surface["roughness"]),
            ),
        )
        node = scene.add(render_mesh, name=f"part_{part_id}")
        part_nodes.append(node)
        segmentation_colors[node] = np.asarray(part["part_color_rgb"], dtype=np.uint8)
    return part_nodes, segmentation_colors


def _add_overlay_nodes(scene: pyrender.Scene, overlays: list[dict[str, Any]]) -> list[Any]:
    nodes = []
    for index, overlay in enumerate(overlays):
        box = np.asarray(overlay["box"], dtype=np.float32)
        mesh = _part_segmentation_wireframe_box_mesh(np.stack((box[:3], box[3:])))
        render_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=True,
            double_sided=True,
            material=_material(overlay["color_rgb"], metallic=0.0, roughness=0.65),
            is_marker=True,
        )
        nodes.append(scene.add(render_mesh, name=f"overlay_{index}"))
        motion_axis = overlay.get("motion_axis")
        if motion_axis in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            axis = {
                "+X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
                "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            }[motion_axis]
            length = max(float(np.max(box[3:] - box[:3])) * 0.30, 0.01)
            arrow = mu.create_arrow(length=length, radius=0.004)
            arrow_render_mesh = pyrender.Mesh.from_trimesh(
                arrow,
                smooth=True,
                double_sided=True,
                material=_material(overlay["color_rgb"], metallic=0.0, roughness=0.65),
                is_marker=True,
            )
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = _z_up_rotation(axis)
            pose[:3, 3] = (box[:3] + box[3:]) * 0.5
            nodes.append(scene.add(arrow_render_mesh, name=f"overlay_axis_{index}", pose=pose))
    return nodes


def _camera_pose(spec: dict[str, Any]) -> np.ndarray:
    position = np.asarray(spec["position"], dtype=np.float32)
    target = np.asarray(spec["target"], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _z_up_rotation(position - target, np.asarray(spec["up"], dtype=np.float32))
    pose[:3, 3] = position
    return pose


def _roi_camera(source: dict[str, Any], box: list[float], target_coverage: float) -> dict[str, Any]:
    bounds = np.asarray(box, dtype=np.float32)
    center = (bounds[:3] + bounds[3:]) * 0.5
    corners = np.asarray(
        [
            [x, y, z]
            for x in (bounds[0], bounds[3])
            for y in (bounds[1], bounds[4])
            for z in (bounds[2], bounds[5])
        ],
        dtype=np.float32,
    )
    source_position = np.asarray(source["position"], dtype=np.float32)
    source_target = np.asarray(source["target"], dtype=np.float32)
    view_direction = source_target - source_position
    view_direction /= np.linalg.norm(view_direction)
    nominal_up = np.asarray(source["up"], dtype=np.float32)
    right = np.cross(view_direction, nominal_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_direction)
    relative = corners - center
    tangent = np.tan(np.deg2rad(float(source["fov_degrees"])) * 0.5) * target_coverage
    along_view = relative @ view_direction
    distance = float(
        np.max(np.maximum(np.abs(relative @ right), np.abs(relative @ up)) / tangent - along_view)
    )
    extent = float(np.max(bounds[3:] - bounds[:3]))
    distance = max(distance, max(extent, 1.0e-4))
    radius = float(np.max(np.linalg.norm(relative, axis=1)))
    return {
        "position": (center - view_direction * distance).tolist(),
        "target": center.tolist(),
        "up": up.tolist(),
        "fov_degrees": float(source["fov_degrees"]),
        "resolution": list(source["resolution"]),
        "near": max(1.0e-5, distance - radius * 1.25),
        "far": distance + radius * 1.25,
    }


def _render(
    scene: pyrender.Scene,
    camera_node: Any,
    *,
    size: tuple[int, int],
    active_nodes: list[Any],
    segmentation: bool,
    segmentation_colors: dict[Any, np.ndarray],
) -> np.ndarray:
    encoded_segmentation = {
        node: np.asarray(color, dtype=np.uint8) for node, color in segmentation_colors.items()
    }
    offscreen = pyrender.OffscreenRenderer(
        pyopengl_platform=os.environ.get("PYOPENGL_PLATFORM", "egl"),
        seg_node_map=encoded_segmentation,
    )
    offscreen.make_current()
    jit = JITRenderer(scene, [], [])
    renderer = pyrender.Renderer(size[0], size[1], jit)
    try:
        result = offscreen.render(
            scene,
            renderer,
            rgb=not segmentation,
            seg=segmentation,
            camera_node=camera_node,
            shadow=not segmentation,
            active_nodes=active_nodes,
            render_pass="part_segmentation" if segmentation else "part_shaded",
        )
        return np.asarray(result[0], dtype=np.uint8)
    finally:
        renderer.delete()
        offscreen.make_uncurrent()
        offscreen.delete()


def render_static_request(request: dict[str, Any]) -> dict[str, Any]:
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ValueError(f"unsupported static diagnostic render request: {request['schema_version']!r}")
    render = request["render"]
    vertices, faces, face_labels = _load_surface(request["asset"])
    background = np.asarray(render["background_rgb"], dtype=np.float32) / 255.0
    ambient = np.asarray(render["ambient_light_rgb"], dtype=np.float32) / 255.0
    scene = pyrender.Scene(bg_color=(*background.tolist(), 1.0), ambient_light=ambient)
    part_nodes, segmentation_colors = _add_part_nodes(
        scene,
        vertices,
        faces,
        face_labels,
        request["asset"]["parts"],
        render["surface"],
    )
    overlay_nodes = _add_overlay_nodes(scene, request["overlays"])
    light_spec = render["directional_light"]
    direction = np.asarray(light_spec["direction"], dtype=np.float32)
    light_pose = np.eye(4, dtype=np.float32)
    light_pose[:3, :3] = _z_up_rotation(-direction)
    light_color = np.asarray(light_spec["color_rgb"], dtype=np.float32) / 255.0
    scene.add(
        pyrender.DirectionalLight(color=light_color, intensity=float(light_spec["intensity"])),
        pose=light_pose,
    )

    camera_nodes = {}
    for view_name, camera_spec in request["cameras"].items():
        camera_nodes[view_name] = scene.add(
            pyrender.PerspectiveCamera(
                yfov=np.deg2rad(float(camera_spec["fov_degrees"])),
                znear=float(camera_spec.get("near", 1.0e-4)),
                zfar=float(camera_spec.get("far", 100.0)),
                aspectRatio=float(camera_spec["resolution"][0] / camera_spec["resolution"][1]),
            ),
            pose=_camera_pose(camera_spec),
        )

    outputs = request["outputs"]
    views = {}
    part_id_png_paths = {}
    selected_part_id = int(request["roi"]["selected_part_id"])
    selected_color = next(
        np.asarray(part["part_color_rgb"], dtype=np.uint8)
        for part in request["asset"]["parts"]
        if int(part["part_id"]) == selected_part_id
    )
    for view_name, camera_node in camera_nodes.items():
        camera_spec = request["cameras"][view_name]
        size = tuple(int(value) for value in camera_spec["resolution"])
        rgb = _render(
            scene,
            camera_node,
            size=size,
            active_nodes=part_nodes + overlay_nodes,
            segmentation=False,
            segmentation_colors=segmentation_colors,
        )
        path = Path(outputs["view_png_paths"][view_name])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(path)
        part_ids = _render(
            scene,
            camera_node,
            size=size,
            active_nodes=part_nodes,
            segmentation=True,
            segmentation_colors=segmentation_colors,
        )
        part_id_path = Path(outputs["part_id_png_paths"][view_name])
        part_id_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(part_ids, mode="RGB").save(part_id_path)
        selected_pixels = int(np.count_nonzero(np.all(part_ids == selected_color, axis=2)))
        object_pixels = int(np.count_nonzero(np.any(part_ids != 0, axis=2)))
        views[view_name] = {
            "path": str(path),
            "visible_pixel_count": object_pixels,
            "selected_part_visible_pixel_count": selected_pixels,
            "camera": camera_spec,
        }
        part_id_png_paths[view_name] = str(part_id_path)

    source_view = max(
        request["roi"]["source_view_order"],
        key=lambda name: views[name]["selected_part_visible_pixel_count"],
    )
    roi_spec = _roi_camera(
        request["cameras"][source_view],
        request["roi"]["box"],
        float(request["roi"]["target_coverage"]),
    )
    roi_camera_node = scene.add(
        pyrender.PerspectiveCamera(
            yfov=np.deg2rad(float(roi_spec["fov_degrees"])),
            znear=float(roi_spec["near"]),
            zfar=float(roi_spec["far"]),
            aspectRatio=1.0,
        ),
        pose=_camera_pose(roi_spec),
    )
    roi_rgb = _render(
        scene,
        roi_camera_node,
        size=tuple(int(value) for value in roi_spec["resolution"]),
        active_nodes=part_nodes + overlay_nodes,
        segmentation=False,
        segmentation_colors=segmentation_colors,
    )
    roi_path = Path(outputs["roi_png_path"])
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(roi_rgb, mode="RGB").save(roi_path)
    clearest = dict(views[source_view])
    clearest["view_name"] = source_view
    return {
        "schema_version": RESULT_SCHEMA,
        "renderer": {
            "backend": "genesis_native_rasterizer",
            "mode": "part_shaded",
            "resolution": list(render["resolution"]),
            "fov_degrees": float(render["fov_degrees"]),
            "surface": dict(render["surface"]),
            "crease_degrees": PART_SHADED_CREASE_DEGREES,
        },
        "views": views,
        "clearest_overview": clearest,
        "roi": {
            "path": str(roi_path),
            "source_view": source_view,
            "camera": roi_spec,
            "selected_part_id": selected_part_id,
        },
        "part_id_png_paths": part_id_png_paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render static HAG4R diagnostic views with Genesis Rasterizer")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    result = render_static_request(request)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
