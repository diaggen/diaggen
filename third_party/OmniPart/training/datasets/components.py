from typing import *  
from abc import abstractmethod 
import os  
import json  
import torch  
import numpy as np  
import pandas as pd  
from PIL import Image  
from torch.utils.data import Dataset  
import torchvision.transforms.functional as TF
import cv2
import random

class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.
    
    This class provides the foundation for dataset handling with common functionality
    like loading metadata from multiple data roots and filtering instances.

    Args:
        roots (str): Comma-separated paths to the dataset directories
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')  
        self.instances = []  
        self.metadata = pd.DataFrame()  

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Filter metadata according to implementation-specific criteria.
        
        Args:
            metadata (pd.DataFrame): The raw metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and statistics dictionary
        """
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        Get a specific instance from the dataset.
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary containing the instance data
        """
        pass
        
    def __len__(self):
        """Return the total number of instances in the dataset"""
        return len(self.instances)

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
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"Failed to get instance {index} from dataset. And the length of dataset is {len(self)}")
            # On error, return a random instance instead
            return self.__getitem__(np.random.randint(0, len(self)))
        
    # def __str__(self):
    #     """
    #     Return a string representation of the dataset, including statistics.
        
    #     Returns:
    #         str: Formatted dataset information
    #     """
    #     lines = []
    #     lines.append(self.__class__.__name__)
    #     lines.append(f'  - Total instances: {len(self)}')
    #     lines.append(f'  - Sources:')
    #     for key, stats in self._stats.items():
    #         lines.append(f'    - {key}:')
    #         for k, v in stats.items():
    #             lines.append(f'      - {k}: {v}')
    #     return '\n'.join(lines)


class TextConditionedMixin:
    """
    Mixin class for datasets conditioned on text/captions.
    
    Adds functionality for loading and accessing text captions associated with instances.
    """
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}  
        
        # Load captions for all instances from metadata
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        """
        Filter metadata to only include instances with captions.
        
        Args:
            metadata (pd.DataFrame): The metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and updated statistics
        """
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]  # Keep only instances with captions
        stats['With captions'] = len(metadata)  # Record count of instances with captions
        return metadata, stats
    
    def get_instance(self, root, instance):
        """
        Extends the base get_instance method to include a randomly chosen caption.
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary with instance data plus caption as 'cond'
        """
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])  # Select a random caption
        pack['cond'] = text  # Add the caption as a condition
        return pack
    
    
class ImageConditionedMixin:
    """
    Mixin class for datasets conditioned on images.
    
    Adds functionality for loading and preprocessing conditional images.
    """
    def __init__(self, roots, *, image_size=518, **kwargs):
        """
        Initialize the mixin with image size parameter.
        
        Args:
            roots (str): Dataset roots
            image_size (int): Size to resize images to (square)
            **kwargs: Additional arguments to pass to parent
        """
        self.image_size = image_size
        super().__init__(roots, **kwargs)
        self.img_data_root = ""
    
    def filter_metadata(self, metadata):
        """
        Filter metadata to only include instances with rendered conditional images.
        
        Args:
            metadata (pd.DataFrame): The metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and updated statistics
        """
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]  # Keep only instances with conditional renders
        stats['Cond rendered'] = len(metadata)  # Record count of instances with conditional renders
        return metadata, stats
    
    def process_image(self, image_path):
        """
        Process a conditional image by cropping to subject, resizing, and handling transparency.
        
        Args:
            image_path (str): Path to the input image file
            
        Returns:
            torch.Tensor: Processed image tensor with alpha compositing applied
        """

        image = Image.open(image_path) 
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()  
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]  
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  
        
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2  
        aug_size_ratio = 1.2  
        aug_hsize = hsize * aug_size_ratio  
        aug_center_offset = [0, 0]  
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]  
        
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), 
                   int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)  

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        alpha = image.getchannel(3)  
        image = image.convert('RGB')  
        
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  
        alpha = torch.tensor(np.array(alpha)).float() / 255.0  
        
        image = image * alpha.unsqueeze(0)  
        return image

    def load_image(self, image_path, size=(518, 518)):
        img = np.array(Image.open(image_path))
        if img.shape[-1] == 4:
            mask_img = img[..., 3] == 0
            img[mask_img] = [0, 0, 0, 255]
            img = img[..., :3]
            img_rgb = Image.fromarray(img.astype('uint8'))
        img_rgb = img_rgb.resize(size, resample=Image.Resampling.BILINEAR)
        img = TF.to_tensor(img_rgb)
        return img, img_rgb
    
    def get_instance(self, uuid, select_method='random_seg'):
        """
        Extends the base get_instance method to include a processed conditional image.
        
        Includes sophisticated image preprocessing:
        1. Loads a random view from available renders
        2. Crops the image around the non-transparent content
        3. Resizes to the target resolution
        4. Applies alpha compositing
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary with instance data plus image as 'cond'
        """
        # Call the second father class from MRO sequence to get the base instance
        pack = super().get_instance(uuid)

        # Load conditional image metadata
        # image_root = os.path.join(root, 'renders_cond', instance)
        # with open(os.path.join(image_root, 'transforms.json')) as f:
        #     metadata = json.load(f)
        
        if select_method == 'random':
            # Select a random view from available frames
            n_views = len(metadata['frames'])
            view = np.random.randint(n_views)

            image_root= ""

            metadata = metadata['frames'][view]
                # Load the image from the selected view
            image_path = os.path.join(image_root, metadata['file_path'])

            image = self.process_image(image_path)  # Process the image

            pack['cond'] = image

        elif select_method == "random_seg":

            data_dir = os.path.join(self.img_data_root, uuid[:2], uuid)
            random_index = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25])
            image_path = os.path.join(data_dir, f"color_{random_index:04d}.webp")
            mask_path = os.path.join(data_dir, f"mask_{random_index:04d}.exr")

            mask, mask_vis, ordered_mask_dino = self.load_bottom_up_mask(mask_path, size=(518, 518))
            pack['ordered_mask_dino'] = ordered_mask_dino

            if os.path.exists(image_path):
                image, image_rgb = self.load_image(image_path)
                pack['cond'] = image
            else:
                raise ValueError(f"Image not found: {image_path}")
            
        else:
            raise ValueError("Invalid select_method. Use 'random' or 'random_seg'.")

        return pack
    
    def load_bottom_up_mask(self, mask_path, size=(518, 518)):
        """
        Load mask and sort parts from bottom to top based on y-coordinate.
        
        Args:
            mask_path: Path to the mask file
            size: Target size for resizing
            
        Returns:
            Ordered mask, visualization mask, and downsampled mask for DINO
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask_dino = cv2.resize(mask, (37, 37), interpolation=cv2.INTER_NEAREST)
        mask_vis = cv2.resize(mask_dino, (518, 518), interpolation=cv2.INTER_NEAREST)
        mask_dino = np.array(mask_dino, dtype=np.int32)[..., 0]
        mask_org = np.array(mask, dtype=np.int32)[..., 0]
        
        unique_indices = np.unique(mask_org)
        unique_indices = unique_indices[unique_indices > 0]
        
        part_positions = {}
        for idx in unique_indices:
            y_coords, _ = np.where(mask_org == idx)
            if len(y_coords) > 0:
                part_positions[idx] = np.max(y_coords)
        
        sorted_parts = sorted(part_positions.items(), key=lambda x: -x[1])
        
        index_map = {}
        for new_idx, (old_idx, _) in enumerate(sorted_parts, 1):
            index_map[old_idx] = new_idx
        
        ordered_mask = np.zeros_like(mask_org)
        ordered_mask_dino = np.zeros_like(mask_dino)
        for old_idx, new_idx in index_map.items():
            ordered_mask[mask_org == old_idx] = new_idx
            ordered_mask_dino[mask_dino == old_idx] = new_idx

        mask = ordered_mask.astype(np.float32)
        mask_vis = np.array(mask_vis, dtype=np.int32)[..., 0]
        ordered_mask_dino = torch.from_numpy(ordered_mask_dino).long()
        
        return mask, mask_vis, ordered_mask_dino
