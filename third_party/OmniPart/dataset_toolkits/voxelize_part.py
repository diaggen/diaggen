import os
import copy
import sys
import importlib
import argparse
from easydict import EasyDict as edict
import pandas as pd
from functools import partial
import numpy as np
import open3d as o3d
import utils3d
from multiprocessing import Pool
from tqdm import tqdm
import json
import glob

def _voxelize(args):
    sha256, output_dir = args
    uuid_path = os.path.join(output_dir, sha256[:2], sha256)
    pattern = os.path.join(uuid_path, "[0-9][0-9][0-9][0-9]")
    matching_dirs = glob.glob(pattern)

    for dir_path in matching_dirs:

        mesh = o3d.io.read_triangle_mesh(os.path.join(dir_path, 'mesh.stl'))
        # clamp vertices to the range [-0.5, 0.5]
        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        assert len(vertices)>0, "Error loading mesh.stl, no vertices found"
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
        vertices = (vertices + 0.5) / 64 - 0.5
        utils3d.io.write_ply(os.path.join(dir_path, f'voxel.ply'), vertices)
        # return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':

    data_path = ''
    output_dir = ''
    
    with open(data_path, 'r') as f:
        data_list = [json.loads(line.strip()) for line in f.readlines()]

    args_list = [(sha256, output_dir) for sha256 in data_list]

    with Pool(128) as p:
        results = list(tqdm(p.imap(_voxelize, args_list), total=len(args_list)))