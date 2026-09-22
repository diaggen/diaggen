import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor, sparse_cat
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.dist_utils import read_file_dist
from ..utils.data_utils import load_balanced_group_indices, save_coords_as_ply
# from ..modules import sparse as sp


class SLatVisMixin:
    """
    Mixin class that adds visualization capabilities for structured latent representations.
    Handles loading of latent decoders and rendering 3D structures from latent codes.
    """
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the visualization mixin.
        
        Args:
            pretrained_slat_dec: Identifier for pretrained decoder model
            slat_dec_path: Optional path to custom decoder model
            slat_dec_ckpt: Optional checkpoint name for custom decoder
        """
        # super().__init__(*args, **kwargs)
        self.slat_dec = None  # Decoder model (loaded on demand)
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        
    def _loading_slat_dec(self):
        """
        Load the structured latent decoder model if not already loaded.
        Uses either a custom path or pretrained model based on initialization parameters.
        """
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            # Load from custom path
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
        else:
            # Load pretrained model
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        """
        Delete the decoder model to free up memory.
        """
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """
        Decode latent vectors into 3D representations.
        """
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            # Apply normalization if needed
            z = z * self.std.to(z.device) + self.mean.to(z.device)

        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i+batch_size]
            z_batch = self.remove_noise(z_batch)
            if z_batch is not None:
                reps.append(self.slat_dec(z_batch))

        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps
    
    def remove_noise(self, z_batch):
        """
        Remove noise from latent vectors by filtering out points with low confidence.
        
        Args:
            z_batch: Latent vectors to process
            
        Returns:
            Processed latent vectors with noise removed
        """
        # Create a new list for processed tensors
        processed_batch = []
        
        for i, z in enumerate(z_batch):
            coords = z.coords
            feats = z.feats
        
            if feats.shape[1] == 9:
                last_dim = feats[:, -1]
                
                sigmoid_val = torch.sigmoid(last_dim)

                to_keep = sigmoid_val >= 0.5
                
                if to_keep.any():
                    filtered_coords = coords[to_keep]
                    filtered_feats = feats[to_keep]
                    filtered_feats = filtered_feats[:, :-1]  
                    
                    processed_z = z.replace(coords=filtered_coords, feats=filtered_feats)
                    processed_batch.append(processed_z)
            else:
                processed_z = z
                
                processed_batch.append(processed_z)
        if len(processed_batch) > 0:
            return sparse_cat(processed_batch)
        else:
            return None

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        """
        Generate multi-view renderings of a 3D representation.
        
        Args:
            x_0: Input sparse tensor or dictionary containing sparse tensor
            
        Returns:
            Tensor of rendered images from multiple viewpoints part_layouts
        """
        if isinstance(x_0, SparseTensor):
            # If x_0 is already a SparseTensor, use it directly
            part_layouts = slice(None, None)
            
        else:
            part_layouts = x_0['part_layouts']
            x_0 = x_0['x_0']

        if part_layouts != slice(None, None):
            reps = []
            sparse_part = []
            for part_id, part_layout in enumerate(part_layouts):
                for part_obj_id, part_slice in enumerate(part_layout):
                    part_x_sparse_tensor = SparseTensor(
                        coords=x_0[part_id].coords[part_slice],
                        feats=x_0[part_id].feats[part_slice],
                    )
                    sparse_part.append(part_x_sparse_tensor)
            reps = self.decode_latent(sparse_cat(sparse_part).cuda())
        else:
            reps = self.decode_latent(x_0.cuda())
        
        # Build camera parameters for multiple viewpoints
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        # fix the angle
        yaws_offset = np.pi / 4
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.pi / 4 for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            # Calculate camera position based on spherical coordinates
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        # Render images for each representation
        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                # Place each view in a grid position
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        images = torch.stack(images)
            
        return images
    
    
class SLat(SLatVisMixin, StandardDatasetBase):
    """
    Structured latent dataset class.
    Handles loading and processing of structured latent representations.
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        data_root: str,
        train_mesh_list: str,
        val_mesh_list: str,
        aug_bbox: Optional[str] = None, # 0 2
        noise_mask_score: float = 4.0,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        """
        Initialize the structured latent dataset.
        """
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # Set up normalization parameters if provided
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
        
        self.data_root = data_root
        self.mesh_list = []
        self.loads = []
        with open(train_mesh_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                uuid, voxel_num = line.strip().split(' ')
                voxel_num = int(voxel_num)
                if voxel_num > self.max_num_voxels:
                    continue
                self.mesh_list.append(uuid)
                self.loads.append(voxel_num)
        # print(self.loads)
        # exit(0)
        if aug_bbox is not None:
            aug_bbox_min, aug_bbox_max = aug_bbox.split(' ')
            self.aug_bbox_range = (int(aug_bbox_min), int(aug_bbox_max))
        else:
            self.aug_bbox_range = None
        self.noise_mask_score = noise_mask_score


    def __len__(self):
        """Return the total number of instances in the dataset"""
        return len(self.mesh_list) 
    
    
    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Get an item from the dataset by index.
        
        Includes error handling - if fetching an instance fails, returns a random instance.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary containing the instance data
        """
        try:
            uuid = self.mesh_list[index]
            # print(f"uuid to find: {uuid}")
            return self.get_instance(uuid)
        except Exception as e:
            print(e)
            # On error, return a random instance instead
            return self.__getitem__(np.random.randint(0, len(self)))

        try:
            root, instance = self.instances[index]
            return self.get_whole_bbox_info(root, instance)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Failed to get instance {index} from dataset. And the length of dataset is {len(self)}")
            # On error, return a random instance instead
            return self.__getitem__(np.random.randint(0, len(self)))
            
    def coords_to_ids(self, coords, max_val=64):
        return coords[:, 0] * max_val**2 + coords[:, 1] * max_val + coords[:, 2]
        

    def get_instance(self, uuid):
        """
        Process all parts and whole model data and combine them into a single output.
        Returns a dictionary with coordinates, features, part names, and part layouts.
        """
        
        data = np.load(os.path.join(self.data_root, uuid[:2], uuid, 'all_latent.npz'))
        all_coords = data['coords'].astype(np.int32)
        all_feats = data['feats'].astype(np.float32)
        offsets = data['offsets'].astype(np.int32)
        all_coords_wnoise = []
        all_feats_wnoise = []
        part_layouts = []
        start_idx = 0
        for i in range(len(offsets) - 1):
            if i == 0:
                overall_coords = all_coords[offsets[i]:offsets[i+1]]
                overall_feats = all_feats[offsets[i]:offsets[i+1]]
                overall_ids = self.coords_to_ids(overall_coords)
                part_layouts.append(slice(start_idx, start_idx + overall_coords.shape[0]))
                start_idx += overall_coords.shape[0]
            else:
                part_coords = all_coords[offsets[i]:offsets[i+1]]
                part_feats = all_feats[offsets[i]:offsets[i+1]]
                part_bbox_min = part_coords.min(axis=0)
                part_bbox_max = part_coords.max(axis=0)
                if self.aug_bbox_range is not None:
                    aug_bbox_min = np.random.randint(self.aug_bbox_range[0], self.aug_bbox_range[1] + 1, size=3)
                    part_bbox_min = part_bbox_min - aug_bbox_min
                    part_bbox_min = np.clip(part_bbox_min, 0, 63)
                    aug_bbox_max = np.random.randint(self.aug_bbox_range[0], self.aug_bbox_range[1] + 1, size=3)
                    part_bbox_max = part_bbox_max + aug_bbox_max
                    part_bbox_max = np.clip(part_bbox_max, 0, 63)

                bbox_mask = np.all((overall_coords >= part_bbox_min) & (overall_coords <= part_bbox_max), axis=1)
                part_ids = self.coords_to_ids(part_coords)
                part_in_overall = np.isin(overall_ids, part_ids)
                noise_mask = np.logical_and(bbox_mask, np.logical_not(part_in_overall))
                noise_coords = overall_coords[noise_mask]
                noise_feats = overall_feats[noise_mask]
                noise_feats = np.concatenate((noise_feats, np.full((noise_feats.shape[0], 1), -self.noise_mask_score)), axis=1)
                part_feats = np.concatenate((part_feats, np.full((part_feats.shape[0], 1), self.noise_mask_score)), axis=1)

                part_coords = np.concatenate((part_coords, noise_coords), axis=0)
                part_feats = np.concatenate((part_feats, noise_feats), axis=0)

                part_layouts.append(slice(start_idx, start_idx + part_coords.shape[0]))
                start_idx += part_coords.shape[0]

                part_coords = torch.from_numpy(part_coords)
                part_feats = torch.from_numpy(part_feats)
                all_coords_wnoise.append(part_coords)
                all_feats_wnoise.append(part_feats)
        
        # add overall coords and feats
        overall_feats = np.concatenate((overall_feats, np.full((overall_feats.shape[0], 1), self.noise_mask_score)), axis=1)
        overall_coords = torch.from_numpy(overall_coords)
        overall_feats = torch.from_numpy(overall_feats)
        all_coords_wnoise.insert(0, overall_coords)
        all_feats_wnoise.insert(0, overall_feats)
        
        # Concatenate all coordinates and features
        combined_coords = torch.cat(all_coords_wnoise, dim=0).int()
        combined_feats = torch.cat(all_feats_wnoise, dim=0).float()
        
        # Apply normalization if needed
        if self.normalization is not None:
            combined_feats = (combined_feats - self.mean) / self.std

        return {
            'coords': combined_coords,
            'feats': combined_feats,
            'part_layouts': part_layouts,  # Includes slices for parts and whole model
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function for creating batches from individual samples.
        Handles sparse tensor construction and layout information.
        
        Args:
            batch: List of data samples to be collated
            split_size: Optional parameter to split large batches based on voxel count
            
        Returns:
            Collated batch or list of smaller batches if split_size is provided
        """
        # Determine grouping strategy based on whether split_size is specified
        if split_size is None:
            # If no split_size, use all samples in a single group
            # like [0, 1, 2, 3]
            group_idx = [list(range(len(batch)))] 
        else:
            # Create balanced groups based on voxel counts to avoid memory issues
            # This distributes samples across groups to balance computational load
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        
        packs = []
        for group in group_idx:
            # Process each group of samples
            sub_batch = [batch[i] for i in group]
            pack = {}
            pack['part_layouts'] = [b['part_layouts'] for b in sub_batch]
            
            # Initialize containers for coordinates, features and layout information
            coords = []
            feats = []
            layout = []
            start = 0
            
            # Process each sample in the current group
            for i, b in enumerate(sub_batch):
                # Add batch index as first dimension to coordinates
                coords.append(torch.cat( # concatenate the batch index and coordinates along the last dimension
                    [torch.full(
                            (b['coords'].shape[0], 1), 
                            i, # the tensor is filled with the batch index "i".
                            dtype=torch.int32), 
                        b['coords']] 
                    , dim=-1))
                feats.append(b['feats'])
                
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            
            # Concatenate all coordinates and features along the first dimension
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            
            # Create a single sparse tensor containing all samples in this group
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            
            # Set the shape of the sparse tensor
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            # Register layout information in the sparse tensor's spatial cache
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # Process any additional data fields present in the samples
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats', 'part_layouts']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            # Add the processed group to the list of packs
            packs.append(pack)
        
        # Return a single pack if no splitting was requested, otherwise return all packs
        if split_size is None:
            return packs[0]
        return packs
    
class TextConditionedSLat(TextConditionedMixin, SLat):
    """
    Text conditioned structured latent dataset.
    Extends the base SLat class with text conditioning capabilities.
    """
    pass


class ImageConditionedSLat(ImageConditionedMixin, SLat):
    """
    Image conditioned structured latent dataset.
    Extends the base SLat class with image conditioning capabilities.
    """
    pass
