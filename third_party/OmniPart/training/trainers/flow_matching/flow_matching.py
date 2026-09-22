"""
Flow Matching Trainer Implementation

This file implements trainers for diffusion models using the flow matching objective.
Flow matching is a generative modeling technique that defines a continuous path between
a noise distribution and the data distribution, and trains models to learn the velocity field 
of this path. Unlike traditional diffusion models that learn to denoise, flow matching directly
learns the vector field that transforms noise to data samples.

The file contains several trainer classes:
1. FlowMatchingTrainer: Base trainer for flow matching
2. FlowMatchingCFGTrainer: Adds classifier-free guidance support
3. TextConditionedFlowMatchingCFGTrainer: Supports text conditioning with CFG
4. ImageConditionedFlowMatchingCFGTrainer: Supports image conditioning with CFG
"""

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
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
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        # Initialize the parent class with all args and kwargs
        super().__init__(*args, **kwargs)
        # Store time schedule configuration for sampling timesteps
        self.t_schedule = t_schedule
        # Store minimum sigma value to prevent numerical instability
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        # Generate random noise if none is provided
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        # Reshape t to broadcast correctly across spatial dimensions
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        # Interpolate between clean data and noise based on timestep
        # Apply minimum sigma to ensure stability at t=1
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        
        Args:
            x_t: The noisy tensor at timestep t.
            t: The timestep values [0-1].
            noise: The noise component to remove.
            
        Returns:
            x_0: The recovered clean data.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        # Reshape t for proper broadcasting
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        # Invert the diffusion process to recover x_0
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        
        Args:
            x_0: Clean data input.
            noise: Noise input.
            t: Timestep values.
            
        Returns:
            v: Velocity vector field at time t.
        """
        # The velocity is the time derivative of the path from data to noise
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        
        Args:
            cond: Conditioning inputs.
            kwargs: Additional arguments.
            
        Returns:
            Processed conditioning data.
        """
        # print(f"debugging get_cond shape {cond.shape}")
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        
        Args:
            cond: Conditioning inputs.
            kwargs: Additional arguments for inference.
            
        Returns:
            Dictionary with conditioning data and additional arguments.
        """
        # print("debugging get_inference_cond")
        # print(cond)
        # print(kwargs)
        
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        
        Args:
            kwargs: Additional arguments for sampler.
            
        Returns:
            A flow-based sampler for generating samples.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        
        Args:
            kwargs: Arguments containing conditioning data.
            
        Returns:
            Dictionary with visualization data (empty by default).
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps based on the configured time schedule.
        
        Args:
            batch_size: Number of timesteps to sample.
            
        Returns:
            Tensor of timestep values in range [0, 1].
        """
        if self.t_schedule['name'] == 'uniform':
            # Uniform sampling between 0 and 1
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            # Logit-normal distribution for timestep sampling
            # Gives more samples near 0 and 1 than uniform
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            A dict with the key "loss" containing a tensor of shape [N],
            may also contain other keys for different terms.
        """
        # Generate random noise for diffusion
        noise = torch.randn_like(x_0)
        # Sample random timesteps for training
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        # Diffuse the clean data to timestep t
        x_t = self.diffuse(x_0, t, noise=noise)
        # print("cond shape ", cond.shape)
        # Process conditioning information
        cond = self.get_cond(cond, **kwargs)

        # print(f"FlowMatchingTrainer cond: {cond.shape}")
        
        # Get model's prediction of the velocity field
        # Multiply t by 1000 to match model's expected timestep scale
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        
        # Compute the target velocity vector field
        target = self.get_v(x_0, noise, t)
        
        # Calculate loss terms
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]  # Total loss is just MSE for basic flow matching

        # Calculate per-time-bin losses for analysis
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        # Divide time range [0,1] into 10 bins and log loss per bin
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
        Generate samples for visualization during training.
        
        Args:
            num_samples: Total number of samples to generate.
            batch_size: Batch size for generation.
            verbose: Whether to print progress information.
            
        Returns:
            Dictionary with generated samples and ground truth data.
        """
        # Create a dataloader to get ground truth samples
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # Get the sampler for inference
        sampler = self.get_sampler()
        sample_gt = []  # Ground truth samples
        sample = []     # Generated samples
        cond_vis = []   # Visualization of conditioning
        
        # Generate samples in batches
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            # print("data is instance")

            #   Move data to GPU and slice to current batch size
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}

            # print("data is instance2")
            # Generate random noise for sampling start point
            noise = torch.randn_like(data['x_0'])
            # Store ground truth data
            sample_gt.append(data['x_0'])
            # Prepare conditioning visualization
            cond_vis.append(self.vis_cond(**data))

            # print("data is instance3")

            # Remove ground truth from conditioning data
            del data['x_0']

            # print("data['cond'] shape ", data['cond'].shape) # torch.Size([4, 3, 3, 518, 518])
            # print("data is instance4.5")
            # for k, v in data.items():
            #     print(f"data {k} value shape {len(v)}")
            #     for i in v:
            #         print(f"{i.shape}")
            # data cond value shape 4
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # torch.Size([3, 3, 518, 518])
            # print(**data)
            # Get conditioning for inference
            args = self.get_inference_cond(**data)
            # args['cond']
            # print("args['cond'] shape ", args['cond'].shape) # torch.Size([4, 4122, 1024])
            # print("data is instance4")
            # Run the sampler to generate samples
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50,           # Number of sampling steps
                cfg_strength=3.0,   # Classifier-free guidance strength
                verbose=verbose,    # Whether to display progress
            )
            # print("data is instance5")
            # Store generated samples
            sample.append(res.samples)

        # Concatenate batches of samples
        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)

        # print("sample shape ", sample.shape)
        
        # Prepare results dictionary
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        # print("cond vis ", cond_vis)
        # raise NotImplementedError("Debugging sample_dict")
        # Add conditioning visualizations
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        # print("run snapshot done")
        # Debug function to print shapes of all values in nested dictionaries

        # Print all shapes in the sample_dict
        # print("Sample dictionary structure and shapes:")
        # sample_gt: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # sample: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # image: 
        # tensor with shape torch.Size([64, 3, 3, 518, 518])
        # image

        # Sample dictionary structure and shapes:
        # sample_gt: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # sample: 
        # tensor with shape torch.Size([64, 8, 16, 16, 16])
        # sample
        # image: 
        # image
        # tensor with shape torch.Size([64, 3, 518, 518])
        # print("Sample dictionary structure and shapes:")
        # for key, value in sample_dict.items():
        #     print(f"{key}: ")
        #     for k, v in value.items():
        #         # print(f"  {k.shape}: {v}")
        #         if k == 'type':
        #             print(v)
        #         elif k == 'value':
        #             print(f"tensor with shape {v.shape}")

        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
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


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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
