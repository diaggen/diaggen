from tqdm import tqdm
from multiprocessing import Pool
import os
import numpy as np

def merge_part_latents(args):

    uuid, output_dir = args
    part_dir = os.path.join(output_dir, uuid[:2], uuid)
    valid_part_id_path = os.path.join(part_dir, 'overall', 'part_id.txt')
    with open(valid_part_id_path, 'r') as f:
        part_ids = f.read().strip().splitlines()
    all_data_coord = []
    all_data_feat = []
    all_data_offset = [0]
    overall_save_path = os.path.join(part_dir, 'overall', 'latent.npz')
    overall_latent = np.load(overall_save_path)
    all_data_coord.append(overall_latent['coords'])
    all_data_feat.append(overall_latent['feats'])
    all_data_offset.append(overall_latent['coords'].shape[0])
    for part_id in part_ids:
        part_latent_path = os.path.join(part_dir, part_id, 'latent.npz')
        part_latent = np.load(part_latent_path)
        all_data_coord.append(part_latent['coords'])
        all_data_feat.append(part_latent['feats'])
        all_data_offset.append(all_data_offset[-1] + part_latent['coords'].shape[0])
    
    all_data_coord = np.concatenate(all_data_coord, axis=0)
    all_data_feat = np.concatenate(all_data_feat, axis=0)
    all_data_offset = np.array(all_data_offset)
    save_dict = {
        'coords': all_data_coord,
        'feats': all_data_feat,
        'offsets': all_data_offset
    }
    save_path = os.path.join(part_dir, 'all_latent.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == "__main__":
 
    valid_uuid_path = ''
    output_dir = ''
    with open(valid_uuid_path, 'r') as f:
        valid_uuids = [line.strip() for line in f.readlines()]
    args_list = []
    for uuid in valid_uuids:
        args_list.append((uuid, output_dir))

    with Pool(64) as p:
        results = list(tqdm(p.imap(merge_part_latents, args_list), total=len(args_list)))