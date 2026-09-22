import glob
import json
import os

import numpy as np
import pyvista as pv
import pytetwild
import torch
import trimesh
from tqdm import tqdm


def load_hag4r_step16_inputs(segmentation_manifest_path, device="cuda"):
    with open(segmentation_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    step16 = manifest["omnipart_step16_inputs"]
    image_white_bg = torch.from_numpy(np.load(step16["image_white_bg_tensor_path"]).astype(np.float32)).to(device)
    image_black_bg = torch.from_numpy(np.load(step16["image_black_bg_tensor_path"]).astype(np.float32)).to(device)
    ordered_mask_input = torch.from_numpy(np.load(step16["ordered_mask_input_path"]).astype(np.int64)).long().to(device)
    return manifest, image_white_bg, image_black_bg, ordered_mask_input


def get_random_color(index=None, use_float=False):
    palette = np.array(
        [
            [141, 211, 199, 255],
            [255, 255, 179, 255],
            [190, 186, 218, 255],
            [251, 128, 114, 255],
            [128, 177, 211, 255],
            [253, 180, 98, 255],
            [179, 222, 105, 255],
            [252, 205, 229, 255],
            [217, 217, 217, 255],
            [188, 128, 189, 255],
            [204, 235, 197, 255],
            [255, 237, 111, 255],
            [102, 194, 165, 255],
            [252, 141, 98, 255],
            [141, 160, 203, 255],
            [231, 138, 195, 255],
            [166, 216, 84, 255],
            [255, 217, 47, 255],
            [229, 196, 148, 255],
            [179, 179, 179, 255],
            [228, 26, 28, 255],
            [55, 126, 184, 255],
            [77, 175, 74, 255],
            [152, 78, 163, 255],
            [255, 127, 0, 255],
            [255, 255, 51, 255],
            [166, 86, 40, 255],
            [247, 129, 191, 255],
            [153, 153, 153, 255],
        ],
        dtype=np.uint8,
    )
    if index is None:
        index = np.random.randint(0, len(palette))
    if index >= len(palette):
        index = index % len(palette)
    return palette[index].astype(np.float32) / 255 if use_float else palette[index]


def change_pcd_range(pcd, from_rg=(-1, 1), to_rg=(-1, 1)):
    return (pcd - (from_rg[0] + from_rg[1]) / 2) / (from_rg[1] - from_rg[0]) * (
        to_rg[1] - to_rg[0]
    ) + (to_rg[0] + to_rg[1]) / 2


def prepare_bbox_gen_input(voxel_coords_path, img_white_bg, ordered_mask_input, bins=64, device="cuda"):
    whole_voxel = np.load(voxel_coords_path)
    whole_voxel = whole_voxel[:, 1:]
    whole_voxel = (whole_voxel + 0.5) / bins - 0.5
    whole_voxel_index = change_pcd_range(whole_voxel, from_rg=(-0.5, 0.5), to_rg=(0.5 / bins, 1 - 0.5 / bins))
    whole_voxel_index = (whole_voxel_index * bins).astype(np.int32)

    points = torch.from_numpy(whole_voxel).to(torch.float16).unsqueeze(0).to(device)
    whole_voxel_index = torch.from_numpy(whole_voxel_index).long().unsqueeze(0).to(device)
    images = img_white_bg.unsqueeze(0).to(device)
    masks = ordered_mask_input.unsqueeze(0).to(device)

    return {
        "points": points,
        "whole_voxel_index": whole_voxel_index,
        "images": images,
        "masks": masks,
    }


def vis_voxel_coords(voxel_coords, bins=64):
    voxel_coords = voxel_coords[:, 1:]
    voxel_coords = (voxel_coords + 0.5) / bins - 0.5
    voxel_coords_ply = trimesh.PointCloud(voxel_coords)
    rot_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    voxel_coords_ply.apply_transform(rot_matrix)
    return voxel_coords_ply


def gen_mesh_from_bounds(bounds):
    bboxes = []
    rot_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for index in range(bounds.shape[0]):
        bbox = trimesh.primitives.Box(bounds=bounds[index])
        bbox.visual.vertex_colors = get_random_color(index, use_float=True)
        bboxes.append(bbox)
    mesh = trimesh.Scene(bboxes)
    mesh.apply_transform(rot_matrix)
    return mesh


def prepare_part_synthesis_input(voxel_coords_path, bbox_depth_path, ordered_mask_input, padding_size=2, bins=64, device="cuda"):
    overall_coords = np.load(voxel_coords_path)
    overall_coords = overall_coords[:, 1:]
    bbox_scene = np.load(bbox_depth_path)
    all_coords_wnoise = []
    part_layouts = []
    start_idx = 0

    part_layouts.append(slice(start_idx, start_idx + overall_coords.shape[0]))
    start_idx += overall_coords.shape[0]
    assigned_points = np.zeros(overall_coords.shape[0], dtype=bool)
    bbox_coords_list = []

    for bbox in bbox_scene:
        points = change_pcd_range(bbox, from_rg=(-0.5, 0.5), to_rg=(0.5 / bins, 1 - 0.5 / bins))
        bbox_min = np.floor(points[0] * bins).astype(np.int32)
        bbox_max = np.ceil(points[1] * bins).astype(np.int32)
        bbox_min = np.clip(bbox_min - padding_size, 0, bins - 1)
        bbox_max = np.clip(bbox_max + padding_size, 0, bins - 1)
        bbox_mask = np.all((overall_coords >= bbox_min) & (overall_coords <= bbox_max), axis=1)
        if np.sum(bbox_mask) == 0:
            continue
        assigned_points = assigned_points | bbox_mask
        bbox_coords = overall_coords[bbox_mask]
        bbox_coords_list.append(bbox_coords)
        part_layouts.append(slice(start_idx, start_idx + bbox_coords.shape[0]))
        start_idx += bbox_coords.shape[0]
        all_coords_wnoise.append(torch.from_numpy(bbox_coords))

    unassigned_mask = ~assigned_points
    unassigned_coords = overall_coords[unassigned_mask]
    if np.sum(unassigned_mask) > 0 and len(bbox_scene) > 0:
        nearest_bbox_indices = []
        for point in unassigned_coords:
            min_dist = float("inf")
            nearest_idx = -1
            for bbox_idx, bbox in enumerate(bbox_scene):
                points = change_pcd_range(bbox, from_rg=(-0.5, 0.5), to_rg=(0.5 / bins, 1 - 0.5 / bins))
                bbox_min = np.floor(points[0] * bins).astype(np.int32)
                bbox_max = np.ceil(points[1] * bins).astype(np.int32)
                dx = min(abs(point[0] - bbox_min[0]), abs(point[0] - bbox_max[0]))
                dy = min(abs(point[1] - bbox_min[1]), abs(point[1] - bbox_max[1]))
                dz = min(abs(point[2] - bbox_min[2]), abs(point[2] - bbox_max[2]))
                dist = min(dx, dy, dz)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = bbox_idx
            nearest_bbox_indices.append(nearest_idx)

        for bbox_idx in range(len(bbox_scene)):
            point_indices = np.array([i for i, idx in enumerate(nearest_bbox_indices) if idx == bbox_idx])
            if len(point_indices) == 0:
                continue
            additional_coords = unassigned_coords[point_indices]
            if bbox_idx < len(bbox_coords_list):
                combined_coords = np.vstack([bbox_coords_list[bbox_idx], additional_coords])
                old_slice = part_layouts[bbox_idx + 1]
                part_layouts[bbox_idx + 1] = slice(old_slice.start, old_slice.start + combined_coords.shape[0])
                additional_count = additional_coords.shape[0]
                for part_idx in range(bbox_idx + 2, len(part_layouts)):
                    old_part_slice = part_layouts[part_idx]
                    part_layouts[part_idx] = slice(
                        old_part_slice.start + additional_count,
                        old_part_slice.stop + additional_count,
                    )
                all_coords_wnoise[bbox_idx] = torch.from_numpy(combined_coords)
                start_idx += additional_count
            else:
                part_layouts.append(slice(start_idx, start_idx + additional_coords.shape[0]))
                start_idx += additional_coords.shape[0]
                all_coords_wnoise.append(torch.from_numpy(additional_coords))

    overall_coords = torch.from_numpy(overall_coords)
    all_coords_wnoise.insert(0, overall_coords)
    combined_coords = torch.cat(all_coords_wnoise, dim=0).int()
    coords = torch.cat([torch.full((combined_coords.shape[0], 1), 0, dtype=torch.int32), combined_coords], dim=-1).to(device)
    masks = ordered_mask_input.unsqueeze(0).to(device)
    return {"coords": coords, "part_layouts": part_layouts, "masks": masks}


def merge_parts(save_dir):
    surface_meshes_colored = []
    scene_list_texture = []
    tet_part_meshes = []
    render_surface_meshes = []
    tet_vertex_part_labels = []
    surface_vertex_part_labels = []
    tri_part_labels = []
    part_colors = []
    part_list = []
    for part_path in glob.glob(os.path.join(save_dir, "*.glb")):
        part_stem = os.path.splitext(os.path.basename(part_path))[0]
        if part_stem.startswith("part") and part_stem[4:].isdigit() and part_stem != "part0":
            part_list.append(part_path)
    part_list.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0][4:]))
    for index, part_surf_path in enumerate(tqdm(part_list, desc="Merging parts")):
        part_surf_mesh = trimesh.load(part_surf_path, force="mesh")
        surface_vertex_part_labels.append(np.full(part_surf_mesh.vertices.shape[0], index, dtype=np.int32))
        tri_part_labels.append(np.full(part_surf_mesh.faces.shape[0], index, dtype=np.int32))
        scene_list_texture.append(part_surf_mesh)
        random_color_uint8 = get_random_color(index, use_float=False)
        part_colors.append(random_color_uint8)
        part_surf_mesh_color = part_surf_mesh.copy()
        part_surf_mesh_color.visual = trimesh.visual.ColorVisuals(
            mesh=part_surf_mesh_color,
            vertex_colors=np.tile(random_color_uint8, (part_surf_mesh_color.vertices.shape[0], 1)),
        )
        surface_meshes_colored.append(part_surf_mesh_color)
        surface_faces = np.hstack(
            [
                np.full((part_surf_mesh.faces.shape[0], 1), 3, dtype=np.int64),
                part_surf_mesh.faces.astype(np.int64),
            ]
        ).reshape(-1)
        surface_poly = pv.PolyData(part_surf_mesh.vertices.astype(np.float64), surface_faces)
        render_surface_meshes.append(surface_poly)
        tet_mesh = pytetwild.tetrahedralize_pv(surface_poly)
        tet_vertex_part_labels.append(np.full(tet_mesh.n_points, index, dtype=np.int32))
        tet_mesh.point_data["part_id"] = tet_vertex_part_labels[-1]
        tet_mesh.cell_data["part_id"] = np.full(tet_mesh.n_cells, index, dtype=np.int32)
        tet_part_meshes.append(tet_mesh)

    scene_texture = trimesh.Scene(scene_list_texture)
    scene_texture.export(os.path.join(save_dir, "mesh_combined_textured.glb"))
    combined_surface_mesh_colored = trimesh.util.concatenate(surface_meshes_colored)
    combined_surface_mesh_colored.export(os.path.join(save_dir, "mesh_combined_colored_by_parts.glb"))
    combined_tet_mesh = pv.merge(tet_part_meshes, merge_points=False)
    pv.save_meshio(os.path.join(save_dir, "mesh_combined.mesh"), combined_tet_mesh)

    tet_vertex_part_labels = np.concatenate(tet_vertex_part_labels, axis=0)
    surface_vertex_part_labels = np.concatenate(surface_vertex_part_labels, axis=0)
    tri_part_labels = np.concatenate(tri_part_labels, axis=0)
    part_colors = np.stack(part_colors, axis=0)
    render_surface_colors = part_colors[:, :3].astype(np.float32) / 255.0
    tet_part_labels = np.asarray(combined_tet_mesh.cell_data["part_id"], dtype=np.int32).reshape(-1)
    combined_tet_mesh.point_data["part_label"] = tet_vertex_part_labels
    combined_tet_mesh.cell_data["part_label"] = tet_part_labels
    combined_tet_mesh.cell_data["part_color"] = part_colors[tet_part_labels][:, :3].astype(np.uint8)
    combined_tet_mesh.save(os.path.join(save_dir, "mesh_combined_colored.vtu"))
    np.savez(
        os.path.join(save_dir, "part_labels.npz"),
        tet_vertex_part_labels=tet_vertex_part_labels,
        tet_part_labels=tet_part_labels,
        surface_vertex_part_labels=surface_vertex_part_labels,
        tri_part_labels=tri_part_labels,
        part_colors=part_colors,
        render_surface_colors=render_surface_colors,
    )

    try:
        if not os.environ.get("DISPLAY"):
            pv.start_xvfb(wait=0.5)
        plotter = pv.Plotter(off_screen=True, notebook=False, window_size=(1024, 1024))
        if getattr(plotter, "ren_win", None) is not None:
            plotter.ren_win.SetOffScreenRendering(1)
        if hasattr(plotter, "disable_anti_aliasing"):
            plotter.disable_anti_aliasing()
        plotter.set_background("white")
        for index, render_mesh in enumerate(render_surface_meshes):
            render_color = render_surface_colors[index]
            plotter.add_mesh(
                render_mesh,
                color=tuple(render_color.tolist()),
                show_edges=False,
                lighting=False,
                smooth_shading=False,
            )

        bounds = combined_tet_mesh.bounds
        center = np.array(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ],
            dtype=np.float64,
        )
        extents = np.array(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
            dtype=np.float64,
        )
        camera_distance = max(2.5 * float(np.max(extents)), 1.0)
        view_specs = [
            ("front", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            ("back", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
            ("left", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            ("right", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            ("top", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
            ("bottom", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ]
        for view_name, direction, up in view_specs:
            position = center + camera_distance * direction
            plotter.camera_position = (
                tuple(position.tolist()),
                tuple(center.tolist()),
                tuple(up.tolist()),
            )
            plotter.reset_camera_clipping_range()
            plotter.render()
            plotter.screenshot(os.path.join(save_dir, f"mesh_combined_part_labels_{view_name}.png"))
        plotter.close()
    except Exception as exc:
        print(f"[WARN] Failed to render six-view part label visualization: {exc}")
