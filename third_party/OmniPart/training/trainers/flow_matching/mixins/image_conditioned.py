"""
Image Conditioned Mixin for Flow Matching Models in TRELLIS

This file defines a mixin class that adds image conditioning functionality to flow matching models.
It provides methods for encoding images using pre-trained DINOv2 vision transformer models,
which allows the models to condition their generation process on input images.
The mixin handles image preprocessing, feature extraction, and conditioning integration.
"""

from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    This class provides functionality to condition models on image inputs using a pre-trained
    vision transformer model (DINOv2 by default). It handles image preprocessing, feature extraction,
    and conditioning integration into the parent model's pipeline.
    
    Args:
        image_cond_model: The image conditioning model name to use from the DINOv2 family.
                          Default is 'dinov2_vitl14_reg'.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        # The model is initialized lazily to save resources until needed
        self.image_cond_model = None
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training by pre-downloading the required models.
        
        This static method ensures the DINOv2 model is downloaded before training starts,
        which helps prevent issues during distributed training.
        
        Args:
            image_cond_model: Name of the DINOv2 model to download.
            **kwargs: Additional arguments passed to parent class's prepare_for_training if it exists.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # Download the model in advance to avoid issues during distributed training
        # torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        
        Loads the DINOv2 model and sets up the necessary normalization transforms.
        Uses distributed utils to ensure proper initialization across processes.
        """
        with dist_utils.local_master_first():
            # Load the model only on the master process first, others will wait
            # dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
            dinov2_model = torch.hub.load('/mnt/pfs/share/pretrained_model/.cache/torch/hub/facebookresearch_dinov2_main', self.image_cond_model_name, trust_repo=True, source='local')
        
        # Move model to GPU and set to evaluation mode
        dinov2_model.eval().cuda()
        
        # Set up normalization transform with ImageNet mean and std
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Store both model and transform for later use
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image], List[torch.Tensor]]) -> torch.Tensor:
        """
        Encode the image into feature representations using the DINOv2 model.
        
        Args:
            image: Either a batch of image tensors [B, C, H, W] or a list of PIL images.
            
        Returns:
            torch.Tensor: The patch token features from DINOv2.
            
        Raises:
            ValueError: If the image input is not a supported type.
        """
        # Initialize model if not already initialized
        if self.image_cond_model is None:
            self._init_image_cond_model()

        # Handle different image input formats
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            # Process 4D tensor (batch) at once without for loop
            image = image.cuda()
            image = self.image_cond_model['transform'](image)
            features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
            patchtokens = F.layer_norm(features, features.shape[-1:])
            # print(f"shape of patchtokens: {patchtokens.shape}") # shape of patchtokens: torch.Size([4, 1374, 1024])
            return patchtokens
            
        elif isinstance(image, torch.Tensor) and image.ndim == 5:
            if all(isinstance(i, torch.Tensor) for i in image):
                # print("image is list of tensors")
                image = image.cuda()
            else:
                raise ValueError("Image list should be either list of PIL images or list of tensors")
            
        elif isinstance(image, list) and all(isinstance(i, Image.Image) for i in image):
            # print("image is list of PIL images")
            # Resize all images to 518x518 pixels using high-quality LANCZOS resampling
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            # Convert images to RGB, transform to numpy arrays, convert to float32, and normalize pixel values to range [0,1]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            # Convert numpy arrays to PyTorch tensors and rearrange dimensions from (H,W,C) to (C,H,W) format required by PyTorch
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            # Stack individual tensors into a batch and move to GPU for faster processing
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        # For non-4D tensor cases, process images one by one
        patchtoken_list = []
        # print(f"shape of image: {image.shape}") # shape of image: torch.Size([4, 3, 3, 518, 518])
        for img in image:
            # print(img.shape)
            img = self.image_cond_model['transform'](img).cuda()
            # print(f"shape of img: {img.shape}") # shape of img: torch.Size([4, 3, 518, 518])
            features = self.image_cond_model['model'](img, is_training=True)['x_prenorm']
            patchtokens = F.layer_norm(features, features.shape[-1:])
            # print(f"shape of patchtokens: {patchtokens.shape}")
            patchtoken_list.append(patchtokens)
        # Concatenate the patch tokens along the batch dimension
        # print(f"shape of patchtoken_list: {len(patchtoken_list)}")
        patchtokens = torch.concat(patchtoken_list, dim=-2)
        # print(f"shape of patchtokens: {patchtokens.shape}") # shape of patchtokens: torch.Size([4, 4122, 1024])
        # raise ValueError(f"Unsupported type of image: {type(image)}")
        return patchtokens
        # shape of patchtokens: torch.Size([3, 1374, 1024])
        # shape of patchtokens: torch.Size([3, 1374, 1024])
        # shape of patchtokens: torch.Size([3, 1374, 1024])
        # shape of patchtokens: torch.Size([3, 1374, 1024])
        # shape of patchtoken_list: 4
        # shape of patchtokens: torch.Size([3, 5496, 1024])
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data for training.
        
        Processes the input images through the DINOv2 encoder and prepares
        both positive and negative conditioning data.
        
        Args:
            cond: Input images to condition on.
            **kwargs: Additional conditioning parameters passed to parent class.
            
        Returns:
            Conditioning data processed by the parent class's get_cond method.
        """
        # print(f"shape of cond: {cond.shape}")  # shape of cond: torch.Size([2, 3, 518, 518])
        # if cond.ndim == 5:
        #     cond = cond.permute(1, 0, 2, 3, 4)
        # print(f"shape of cond after permute: {cond.shape}") # shape of cond after permute: torch.Size([2, 3, 518, 518])
        cond = self.encode_image(cond)
        # print(f"shape of cond after encode: {cond.shape}") # shape of cond after encode: torch.Size([2, 1374, 1024])
        # print(f"get cond shape: {cond.shape}") # get cond shape: torch.Size([2, 1374, 1024])
        # Create negative conditioning as zeros (used for classifier-free guidance)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        kwargs['neg_mask'] = torch.zeros_like(kwargs['ordered_mask_dino'])
        cond = super().get_cond(cond, **kwargs)
        # print(f"get cond done shape: {cond.shape}") # get cond done shape: torch.Size([2, 1374, 1024])
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        
        Similar to get_cond but specifically used during inference/generation.
        
        Args:
            cond: Input images to condition on.
            **kwargs: Additional conditioning parameters passed to parent class.
            
        Returns:
            Conditioning data processed by the parent class's get_inference_cond method.
        """
        # print(f"get inference cond: {cond}")
        # print(kwargs)
        # cond.shape torch.Size([4, 3, 3, 518, 518])
        # if cond.ndim == 5:
        #     cond = cond.permute(1, 0, 2, 3, 4)
        # cond.shape torch.Size([3, 4, 3, 518, 518])
        cond = self.encode_image(cond)
        #  torch.Size([4, 4122, 1024])
        # print(f"get inference cond2222")
        # Create negative conditioning as zeros (used for classifier-free guidance)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        kwargs['neg_mask'] = torch.zeros_like(kwargs['ordered_mask_dino'])
        # print("get inference cond done2")
        # print(f"{kwargs}")
        cond = super().get_inference_cond(cond, **kwargs)
        # print("get_inference_cond done")
        # print("shape of cond: ", cond.shape)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        
        Prepares the conditioning data for visualization, typically used for debugging
        or monitoring the model's behavior.
        
        Args:
            cond: The conditioning image data.
            **kwargs: Additional visualization parameters.
            
        Returns:
            dict: Dictionary containing the image data and its type for visualization.
        """
        if isinstance(cond, torch.Tensor):
            return {'image': {'value': cond, 'type': 'image'}}
        else:
            raise ValueError(f"Unsupported type of cond: {type(cond)}")
