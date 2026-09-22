from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from transformers import AutoModel
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..modules.sparse.basic import SparseTensor, sparse_cat

class OmniPartImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring OmniPart image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        # Skip initialization if models is None (used in from_pretrained)
        if models is None:
            return
            
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    
    @staticmethod
    def from_pretrained(path: str) -> "OmniPartImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            
        Returns:
            OmniPartImageTo3DPipeline: Loaded pipeline instance
        """
        pipeline = super(OmniPartImageTo3DPipeline, OmniPartImageTo3DPipeline).from_pretrained(path)
        new_pipeline = OmniPartImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # Initialize samplers from saved arguments
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(
            **args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(
            **args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        
        Args:
            name (str): Name of the DINOv2 model to load
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
    
    
    def preprocess_image(self, input: Image.Image, size=(518, 518)) -> Image.Image:
        """
        Preprocess the input image for the model.
        
        Args:
            input (Image.Image): Input image
            size (tuple): Target size for resizing
            
        Returns:
            Image.Image: Preprocessed image
        """
        img = np.array(input)
        if img.shape[-1] == 4:
            # Handle alpha channel by replacing transparent pixels with black
            mask_img = img[..., 3] == 0
            img[mask_img] = [0, 0, 0, 255]
            img = img[..., :3]
            img_rgb = Image.fromarray(img.astype('uint8'))
        # Resize to target size
        img_rgb = img_rgb.resize(size, resample=Image.Resampling.BILINEAR)
        return img_rgb

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image using the conditioning model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image(s) to encode

        Returns:
            torch.Tensor: The encoded features
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            # Convert PIL images to tensors
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        # Apply normalization and run through DINOv2 model
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, List[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: Dictionary with conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)  # Negative conditioning (zero)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        save_coords: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            save_coords (bool): Whether to save coordinates internally.
            
        Returns:
            torch.Tensor: Coordinates of the sparse structure
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        
        # Merge default and custom sampler parameters
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        # Generate samples using the sampler
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent to get coordinates
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        if save_coords:
            self.save_coordinates = coords

        return coords
    
    @torch.no_grad()
    def get_coords(
        self,
        image: Union[Image.Image, List[Image.Image]],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
        save_coords: bool = False,
    ) -> dict:
        """
        Get coordinates of the sparse structure from an input image.

        Args:
            image: Input image or list of images
            num_samples: Number of samples to generate
            seed: Random seed
            sparse_structure_sampler_params: Additional parameters for the sparse structure sampler
            preprocess_image: Whether to preprocess the image
            save_coords: Whether to save coordinates internally
            
        Returns:
            torch.Tensor: Coordinates of the sparse structure
        """
        if isinstance(image, Image.Image):
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, save_coords)
            return coords
        elif isinstance(image, torch.Tensor):
            cond = self.get_cond(image.unsqueeze(0))
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, save_coords)
            return coords
        elif isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(i) for i in image]
            cond = self.get_cond(image)
            torch.manual_seed(seed)
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, save_coords)
            return coords
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
    
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        part_layouts: List[slice] = None, 
        masks: torch.Tensor = None,
        sampler_params: dict = {},
        **kwargs
    ) -> sp.SparseTensor:
        # Sample structured latent
        flow_model = self.models['slat_flow_model']

        # Create noise tensor with same coordinates as the sparse structure
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        
        # Merge default and custom sampler parameters
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        
        # Add part information if provided
        if part_layouts is not None:
            kwargs['part_layouts'] = part_layouts
        if masks is not None:
            kwargs['masks'] = masks
            
        # Generate samples
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            **kwargs
        ).samples
        
        # Normalize the features
        feat_dim = slat.feats.shape[1]
        base_std = torch.tensor(self.slat_normalization['std']).to(slat.device)
        base_mean = torch.tensor(self.slat_normalization['mean']).to(slat.device)
        
        # Handle different dimensionality cases
        if feat_dim == len(base_std):
            # Dimensions match, apply directly
            std = base_std[None, :]
            mean = base_mean[None, :]
        elif feat_dim == 8 and len(base_std) == 9:
            # Use first 8 dimensions when latent is 8-dimensional but normalization is 9-dimensional
            std = base_std[:8][None, :]
            mean = base_mean[:8][None, :]
            print(f"Warning: Normalizing {feat_dim}-dimensional features with first 8 dimensions of 9-dimensional parameters")
        else:
            # Handle general case of dimension mismatch
            std = torch.ones((1, feat_dim), device=slat.device)
            mean = torch.zeros((1, feat_dim), device=slat.device)
            
            copy_dim = min(feat_dim, len(base_std))
            std[0, :copy_dim] = base_std[:copy_dim]
            mean[0, :copy_dim] = base_mean[:copy_dim]
            print(f"Warning: Feature dimensions mismatch. Using {copy_dim} dimensions for normalization")
            
        # Apply normalization
        slat = slat * std + mean

        return slat
    
    @torch.no_grad()
    def get_slat(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor],
        coords: torch.Tensor,
        part_layouts: List[slice], 
        masks: torch.Tensor,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:

        if isinstance(image, Image.Image):
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
            torch.manual_seed(seed)
            slat = self.sample_slat(cond, coords, part_layouts, masks, slat_sampler_params)
            return self.decode_slat(self.divide_slat(slat, part_layouts), formats)
        elif isinstance(image, list):
            if preprocess_image:
                image = [self.preprocess_image(i) for i in image]
            cond = self.get_cond(image)
            torch.manual_seed(seed)
            slat = self.sample_slat(cond, coords, part_layouts, masks, slat_sampler_params)
            return self.decode_slat(self.divide_slat(slat, part_layouts), formats)
        elif isinstance(image, torch.Tensor):
            cond = self.get_cond(image.unsqueeze(0))
            torch.manual_seed(seed)
            slat = self.sample_slat(cond, coords, part_layouts, masks, slat_sampler_params)
            return self.decode_slat(self.divide_slat(slat, part_layouts), formats)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
    
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent
            formats (List[str]): The formats to decode to
            
        Returns:
            dict: Decoded outputs in requested formats
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def divide_slat(
        self,
        slat: sp.SparseTensor,
        part_layouts: List[slice],
    ) -> List[sp.SparseTensor]:
        """
        Divide the structured latent into parts.
        
        Args:
            slat (sp.SparseTensor): The structured latent
            part_layouts (List[slice]): Layout information for parts
            
        Returns:
            sp.SparseTensor: Processed and divided latent
        """ 
        sparse_part = []
        for part_id, part_layout in enumerate(part_layouts):
            for part_obj_id, part_slice in enumerate(part_layout):
                part_x_sparse_tensor = SparseTensor(
                    coords=slat[part_id].coords[part_slice],
                    feats=slat[part_id].feats[part_slice],
                )
                sparse_part.append(part_x_sparse_tensor)

        slat = sparse_cat(sparse_part)

        return self.remove_noise(slat)
    
    def remove_noise(self, z_batch):
        """
        Remove noise from latent vectors by filtering out points with low confidence.
        
        Args:
            z_batch: Latent vectors to process
            
        Returns:
            sp.SparseTensor: Processed latent with noise removed
        """
        # Create a new list for processed tensors
        processed_batch = []
        
        for i, z in enumerate(z_batch):
            coords = z.coords
            feats = z.feats

            # Only filter if features have a confidence dimension (9th dimension)
            if feats.shape[1] == 9:
                # Get the confidence values (last dimension)
                last_dim = feats[:, -1]
                sigmoid_val = torch.sigmoid(last_dim)

                # Calculate filtering statistics
                total_points = coords.shape[0]
                to_keep = sigmoid_val >= 0.3
                kept_points = to_keep.sum().item()
                discarded_points = total_points - kept_points
                discard_percentage = (discarded_points / total_points) * 100 if total_points > 0 else 0
                
                if kept_points == 0:
                    print(f"No points kept for part {i}")
                    continue
                
                print(f"Discarded {discarded_points}/{total_points} points ({discard_percentage:.2f}%)")
                
                # Filter coordinates and features
                coords = coords[to_keep]
                feats = feats[to_keep]
                feats = feats[:, :-1]  # Remove the confidence dimension
                
                # Create a filtered SparseTensor
                processed_z = z.replace(coords=coords, feats=feats)
            else:
                processed_z = z

            processed_batch.append(processed_z)
            
        return sparse_cat(processed_batch)

    
    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject
            num_images (int): The number of images to condition on
            num_steps (int): The number of steps to run the sampler for
            mode (str): Sampling strategy ('stochastic' or 'multidiffusion')
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            # Create schedule for which image to use at each step
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode == 'multidiffusion':
            from .samplers import FlowEulerSampler
            
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    # Average predictions from all conditions when within CFG interval
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    # Average predictions from all conditions when outside CFG interval
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            # Restore original inference model
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, f'_old_inference_model')