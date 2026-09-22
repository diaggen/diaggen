"""
Sparse Flow Matching Trainer Implementation

This file implements trainers for sparse generative models using flow matching objectives.
Flow matching is a technique for generative modeling that defines probability flows between
noise and data distributions. This implementation specifically handles sparse data structures,
which are common in 3D point clouds, graphs, and other non-dense representations.

The file contains multiple trainer classes:
- SparseFlowMatchingTrainer: Base trainer for sparse flow matching models
- SparseFlowMatchingCFGTrainer: Adds classifier-free guidance for improved generation
- TextConditionedSparseFlowMatchingCFGTrainer: Enables text conditioning for sparse generation
- ImageConditionedSparseFlowMatchingCFGTrainer: Enables image conditioning for sparse generation

These trainers handle the training loop, loss calculation, data loading, and sampling for
sparse flow matching models.
"""

from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader for training.
        
        This method initializes the data sampler and dataloader with proper configurations
        for efficient batch processing. It uses BalancedResumableSampler to ensure training
        can be resumed from checkpoints, and sets up an infinite iterator over the data.
        """
        print("original dataset size:", len(self.dataset))
        num_dataset = 128
        # Wrap your dataset in the DuplicatedDataset if it's too small
        # print(f"Dataset size: {len(self.dataset)}")
        if len(self.dataset) < num_dataset:  # Adjust this threshold as needed
            from ...utils.data_utils import DuplicatedDataset

            self.dataset = DuplicatedDataset(self.dataset, repeat=num_dataset)
            print(f"Dataset duplicated to {len(self.dataset)} samples")
        
        # print("data_sampler:")
        # Create a sampler that can be resumed from checkpoints
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        
        # Create the dataloader with optimized settings
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),  # Scale workers by GPU count
            pin_memory=True,  # Speed up CPU to GPU transfer
            drop_last=True,  # Ensure all batches are the same size
            persistent_workers=True,  # Keep workers alive between epochs for efficiency
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        
        # Create an infinite data iterator to simplify training loop
        self.data_iterator = cycle(self.dataloader)
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        This method implements the flow matching training objective for sparse data.
        It diffuses the input data according to the flow matching schedule,
        predicts the velocity field, and computes the loss between the prediction and target.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A tuple containing:
            - terms: Dictionary with the key "loss" and other loss components
            - An empty dictionary (for compatibility with other trainers)
        """
        # Generate random noise with the same sparsity pattern as input
        noise = x_0.replace(torch.randn_like(x_0.feats))
        
        # Sample random timesteps for each item in the batch
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        
        # Apply diffusion process to get x_t
        x_t = self.diffuse(x_0, t, noise=noise)
        
        # Process conditional inputs if any
        cond, ordered_mask_dino = self.get_cond(cond, **kwargs)
        kwargs['ordered_mask_dino'] = ordered_mask_dino
        # print(f"loss shape cond: {cond.shape}") # loss shape cond: torch.Size([2, 1374, 1024])
        
        # Run model to predict velocity field
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        # print(pred.shape, noise.shape, x_0.shape)
        assert pred.shape == noise.shape == x_0.shape
        
        # Calculate target vector field based on flow matching objective
        target = self.get_v(x_0, noise, t)
        
        # Compute loss terms
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # Log detailed loss statistics binned by timestep
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        # Divide time range [0,1] into 10 bins and compute per-bin statistics
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Generate samples for visualization and evaluation.
        
        This method creates samples from the model for visualization purposes. It processes
        batches of data from the dataset, generates conditional samples, and organizes
        them for logging and visualization.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size to use for generation
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing generated samples, ground truth, and conditioning information
        """
        # Create a temporary dataloader for sampling
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # No parallelization for simplicity
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # Get the sampler for generation
        sampler = self.get_sampler()
        
        # Initialize lists to store results
        sample_gt = []  # Ground truth samples
        sample = []     # Generated samples
        cond_vis = []   # Visualization of conditioning

        part_layouts_list = []
        
        # Generate samples in batches
        for i in range(0, num_samples, batch_size):
            # Get actual batch size (might be smaller for the last batch)
            batch = min(batch_size, num_samples - i)
            
            # Get a batch of data
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}

            for layout in data['part_layouts']:
                part_layouts_list.append(layout)
            
            # Create initial noise with same sparsity pattern as input
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            
            # Store ground truth and conditioning visualization
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            
            # Remove ground truth from data dictionary
            del data['x_0']
            
            # Prepare conditioning for inference
            args = self.get_inference_cond(**data)
            
            # Generate samples using the sampler
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50,  # Number of sampling steps
                cfg_strength=3.0,  # Classifier-free guidance strength
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)

        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},  # Ground truth
            'sample': {'value': sample, 'type': 'sample'},        # Generated samples
        }
        
        # Add conditioning visualization to dictionary
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        sample_dict['layout'] = {'value': part_layouts_list, 'type': 'layout'}
        
        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    This class extends SparseFlowMatchingTrainer with classifier-free guidance capabilities,
    which helps improve sample quality by learning both conditional and unconditional models.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    This class adds text conditioning capabilities to the sparse flow matching trainer,
    allowing the generation of sparse data conditioned on text prompts.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    This class adds image conditioning capabilities to the sparse flow matching trainer,
    allowing the generation of sparse data conditioned on input images.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
