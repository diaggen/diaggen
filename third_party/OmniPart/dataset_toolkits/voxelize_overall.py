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

import utils3d

def _voxelize(args):
    sha256, output_dir = args
    uuid_path = os.path.join(output_dir, sha256[:2], sha256)
    pattern = os.path.join(uuid_path, "[0-9][0-9][0-9][0-9]")
    matching_dirs = glob.glob(pattern)
    part_id_list = [os.path.basename(dir_path) for dir_path in matching_dirs]
    part_num = len(part_id_list)
    filter_small_part_th = 5

    part_voxel_list = []
    new_part_id_list = []

    for i, (dir_path, part_id) in enumerate(zip(matching_dirs, part_id_list)):
        part_path = os.path.join(dir_path, 'voxel.ply')
        voxel = utils3d.io.read_ply(part_path)[0]
        if len(voxel) <= filter_small_part_th:
            continue
        part_voxel_list.append(voxel)
        new_part_id_list.append(part_id)
    
    if len(part_voxel_list) == 0:
        print(f"Error: No valid parts found for {sha256}")
        return
    
    combined = list(zip(part_voxel_list, new_part_id_list))
    sorted_combined = sorted(combined, key=lambda x: x[0].min(axis=0)[2])
    part_voxel_list, new_part_id_list = zip(*sorted_combined)

    overall_voxel = np.vstack(part_voxel_list)
    overall_voxel = np.unique(overall_voxel, axis=0)

    utils3d.io.write_ply(os.path.join(uuid_path, 'overall', f'voxel.ply'), overall_voxel)
    with open(os.path.join(uuid_path, 'overall', f'part_id.txt'), 'w') as f:
        for part_id in new_part_id_list:
            f.write(f"{part_id}\n")
        
        


if __name__ == '__main__':
    data_path = ''
    output_dir = ''
    
    with open(data_path, 'r') as f:
        data_list = [json.loads(line.strip()) for line in f.readlines()]

    args_list = [(sha256, output_dir) for sha256 in data_list]

    with Pool(128) as p:
        results = list(tqdm(p.imap(_voxelize, args_list), total=len(args_list)))