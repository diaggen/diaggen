import numpy as np
import trimesh
import torch


def normalize_scene(scene, rg=(-0.5, 0.5)):
    # put to [-0.5, 0.5]
    whole_center = scene.bounding_box.centroid
    scene.apply_translation(-whole_center)
    whole_scale = max(scene.bounding_box.extents)
    scene.apply_scale((rg[1]-rg[0]) / whole_scale)
    return scene

def normalize_mesh(mesh, rg=(-1,1)):
    # put to [-1, 1]
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    center = (vmin + vmax) / 2
    scale = (vmax - vmin).max()
    mesh.vertices = (mesh.vertices - center) / scale * (rg[1] - rg[0]) + (rg[0] + rg[1]) / 2

def change_mesh_range(mesh, from_rg=(-1,1), to_rg=(-1,1)):
    mesh.vertices = (mesh.vertices - (from_rg[0] + from_rg[1]) / 2) / (from_rg[1] - from_rg[0]) * (to_rg[1] - to_rg[0]) + (to_rg[0] + to_rg[1]) / 2
    return mesh

def change_pcd_range(pcd, from_rg=(-1,1), to_rg=(-1,1)):
    pcd = (pcd - (from_rg[0] + from_rg[1]) / 2) / (from_rg[1] - from_rg[0]) * (to_rg[1] - to_rg[0]) + (to_rg[0] + to_rg[1]) / 2
    return pcd

def quantize_vertices(v, bins):
    return (v * bins).astype(np.int32)

def sample_points(mesh, n):
    points, face_index = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_index]
    return points, normals

def clear_mesh(mesh):
    mesh.update_faces(mesh.nondegenerate_faces(height=1.e-8))
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=0)
    return mesh