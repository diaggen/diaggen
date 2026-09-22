"""
Classifier-Free Guidance Mixin for Flow Matching Models

This file implements a mixin class that adds classifier-free guidance capabilities to flow matching models.
Classifier-free guidance is a technique that combines conditional and unconditional generation to improve
sample quality and control. This mixin provides functionality for training with randomly dropped conditions
and for sampling with guidance during inference.
"""

import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        """
        Initialize the ClassifierFreeGuidanceMixin.
        
        Args:
            p_uncond: Probability of using the unconditional branch during training.
                      Higher values mean more unconditional training.
            *args, **kwargs: Arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for training with classifier-free guidance.
        Randomly replaces conditioning with negative conditioning based on p_uncond.
        
        Args:
            cond: The positive conditioning data.
            neg_cond: The negative conditioning data (usually null/empty conditioning).
            **kwargs: Additional arguments.
            
        Returns:
            Modified conditioning data with some entries replaced by negative conditioning.
        """
        # print(f"=== get_cond called ===")
        # print(f"p_uncond: {self.p_uncond}") # p_uncond: 0.1
        # print(f"cond type: {type(cond)}")
        # print(f"cond shape: {cond.shape if isinstance(cond, torch.Tensor) else 'N/A'}") # cond shape: torch.Size([2, 1374, 1024])
        # print(f"neg_cond type: {type(neg_cond)}")
        
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # print(f"p_uncond > 0, applying classifier-free guidance training")
            
            # Helper function to determine batch size from conditioning data
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    # print(f"  cond is tensor with shape {cond.shape}") # ond is tensor with shape torch.Size([2, 1374, 1024])
                    return cond.shape[0]
                elif isinstance(cond, list):
                    # print(f"  cond is list with length {len(cond)}")
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            # Get the batch size from the first available conditioning element
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            # print(f"Getting batch size from ref_cond of type: {type(ref_cond)}")
            B = get_batch_size(ref_cond)
            # print(f"Determined batch size: {B}") # Determined batch size: 2
            
            # Helper function to selectively replace conditioning with negative conditioning
            def select(cond, neg_cond, mask):
                # print(f"  select called with cond type: {type(cond)}, neg_cond type: {type(neg_cond)}")
                if isinstance(cond, torch.Tensor):
                    # For tensor data, use torch.where for efficient conditional replacement
                    # print(f"  Processing tensor with shape {cond.shape}")
                    mask_tensor = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    # print(f"  Mask shape: {mask_tensor.shape}, replacing {sum(mask)}/{len(mask)} samples")
                    return torch.where(mask_tensor, neg_cond, cond)
                elif isinstance(cond, list):
                    # For list data, use list comprehension to selectively replace elements
                    # print(f"  Processing list with length {len(cond)}, replacing {sum(mask)}/{len(mask)} samples")
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            # Generate random mask for which samples to replace with negative conditioning
            mask = list(np.random.rand(B) < self.p_uncond)
            # print(f"Generated mask: {mask}") # Generated mask: [False, True]
            # print(f"Will replace {sum(mask)}/{len(mask)} samples with negative conditioning")
            
            # Apply the conditioning replacement based on data type
            # print(f"Applying conditioning replacement...")
            if not isinstance(cond, dict):
                # print(f"Direct conditioning (non-dict) replacement")
                # import pdb; pdb.set_trace()
                cond = select(cond, neg_cond, mask)
                ordered_mask_dino = kwargs.get('ordered_mask_dino', None)
                if ordered_mask_dino is not None:
                    neg_mask = torch.zeros_like(ordered_mask_dino)
                    ordered_mask_dino = select(ordered_mask_dino, neg_mask, mask)
            else:
                # print(f"Dict conditioning with keys: {list(cond.keys())}")
                # For dictionary conditioning, apply the operation to each key-value pair
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
                # print(f"Completed dict_foreach with resulting keys: {list(cond.keys())}")
        else:
            print(f"p_uncond = {self.p_uncond}, skipping classifier-free guidance")
        
        # print(f"final cond shape: {cond.shape}") # final cond shape: torch.Size([2, 1374, 1024])

        # print(f"=== get_cond returning ===")
        return cond, ordered_mask_dino

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data prepared for inference with classifier-free guidance.
        During inference, we need both conditional and unconditional branches.
        
        Args:
            cond: The positive conditioning data.
            neg_cond: The negative/null conditioning data.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing both positive and negative conditioning for the sampler.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the specialized sampler for classifier-free guidance flow matching.
        
        Args:
            **kwargs: Additional arguments to pass to the sampler.
            
        Returns:
            An instance of FlowEulerCfgSampler configured with the model's sigma_min.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
