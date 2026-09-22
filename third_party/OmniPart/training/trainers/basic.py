import os
import copy
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.
    
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
    """

    def __str__(self):
        """
        Returns a string representation of the trainer with all its configurations.
        This provides a comprehensive overview of the training setup.
        """
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - FP16 mode: {self.fp16_mode}')
        return '\n'.join(lines)
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        
        This method performs the following tasks:
        1. Sets up DDP for distributed training if world_size > 1
        2. Collects model parameters for optimization
        3. Sets up mixed precision training based on fp16_mode
        4. Initializes EMA parameters if needed
        5. Sets up the optimizer and learning rate scheduler
        6. Configures elastic memory management if enabled
        7. Sets up gradient clipping if configured
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()], [])
        if self.fp16_mode == 'amp':
            # For AMP mode, we use PyTorch's automatic mixed precision
            self.master_params = self.model_params
            # self.scaler = torch.GradScaler() if self.fp16_mode == 'amp' else None
            self.scaler = torch.cuda.amp.GradScaler() if self.fp16_mode == 'amp' else None
        elif self.fp16_mode == 'inflat_all':
            # Manual FP16 mode with master params in FP32
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            self.log_scale = 20.0
        elif self.fp16_mode is None:
            # Standard FP32 training
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'FP16 mode {self.fp16_mode} is not implemented.')

        # Build EMA params - only the master process maintains EMA parameters
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        
        This is used when saving model parameters, to convert from the flat
        list of master parameters back to the state_dict format for each model.
        
        Args:
            master_params: List of parameters in flat format
            
        Returns:
            dict: Dictionary of model state dicts
        """
        if self.fp16_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        
        This is used when loading parameters, to convert from the state_dict
        format to the flat list of master parameters.
        
        Args:
            master_params: List of parameters in flat format to be updated
            state_dicts: Dictionary of model state dicts
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        
        params = []
        for name, param_name in master_params_names:
            if name in state_dicts and param_name in state_dicts[name]:
                # Parameter exists in checkpoint
                params.append(state_dicts[name][param_name])
            else:
                # Parameter missing in checkpoint, use the model's initialized parameter
                if self.is_master:
                    print(f'Warning: {name}.{param_name} not found in checkpoint, using model initialization.')
                # Find the parameter in the model
                for n, p in self.models[name].named_parameters():
                    if n == param_name:
                        params.append(p.detach().clone())
                        break
        
        if self.fp16_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        
        This method loads model parameters, optimizer state, and other training state
        from a previously saved checkpoint.
        
        Args:
            load_dir: Directory where checkpoints are saved
            step: Training step to load from
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
        
        # Load models
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt, strict=False)
            if self.fp16_mode == 'inflat_all':
                model.convert_to_fp16()
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        # Load EMA parameters (only for master process)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        # Load miscellaneous state (optimizer, step, etc.)
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.fp16_mode == 'amp':
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.fp16_mode == 'inflat_all':
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        # Synchronize processes after loading
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        # Verify that DDP is working correctly
        if self.world_size > 1:
            self.check_ddp()

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        
        This method loads model parameters from a checkpoint for finetuning,
        handling shape mismatches and partial loading.
        
        Args:
            finetune_ckpt: Dictionary mapping model names to checkpoint paths
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        # Load models
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                # Load checkpoint if available for this model
                model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=False)
                
                # Handle shape mismatches more intelligently
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} in checkpoint but not in model, skipped.')
                        continue
                    
                    if model_ckpt[k].shape != model_state_dict[k].shape:
                        ckpt_shape = model_ckpt[k].shape
                        model_shape = model_state_dict[k].shape
                        
                        # Try to adapt the parameter if possible
                        adapted = False
                        
                        # Case 1: Weights where model has more input features (like input_layer.weight)
                        if len(ckpt_shape) == 2 and len(model_shape) == 2 and ckpt_shape[0] == model_shape[0] and ckpt_shape[1] < model_shape[1]:
                            if self.is_master:
                                print(f'Adapting {k}: {ckpt_shape} -> {model_shape} (adding features)')
                            new_tensor = torch.zeros(model_shape, dtype=model_ckpt[k].dtype, device=model_ckpt[k].device)
                            new_tensor[:, :ckpt_shape[1]] = model_ckpt[k]
                            model_ckpt[k] = new_tensor
                            adapted = True
                        
                        # Case 2: Weights where model has more output features (like out_layer.weight)
                        elif len(ckpt_shape) == 2 and len(model_shape) == 2 and ckpt_shape[0] < model_shape[0] and ckpt_shape[1] == model_shape[1]:
                            if self.is_master:
                                print(f'Adapting {k}: {ckpt_shape} -> {model_shape} (adding outputs)')
                            new_tensor = torch.zeros(model_shape, dtype=model_ckpt[k].dtype, device=model_ckpt[k].device)
                            new_tensor[:ckpt_shape[0], :] = model_ckpt[k]
                            model_ckpt[k] = new_tensor
                            adapted = True
                        
                        # Case 3: Biases where model has more outputs (like out_layer.bias)
                        elif len(ckpt_shape) == 1 and len(model_shape) == 1 and ckpt_shape[0] < model_shape[0]:
                            if self.is_master:
                                print(f'Adapting {k}: {ckpt_shape} -> {model_shape} (expanding bias)')
                            new_tensor = torch.zeros(model_shape, dtype=model_ckpt[k].dtype, device=model_ckpt[k].device)
                            new_tensor[:ckpt_shape[0]] = model_ckpt[k]
                            model_ckpt[k] = new_tensor
                            adapted = True
                        
                        # If we couldn't adapt the parameter, use the model's initialization
                        if not adapted:
                            if self.is_master:
                                print(f'Warning: {k} shape mismatch, {ckpt_shape} vs {model_shape}, skipped.')
                            model_ckpt[k] = model_state_dict[k]
                
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt, strict=False)
                if self.fp16_mode == 'inflat_all':
                    model.convert_to_fp16()
            else:
                # Keep original params if no checkpoint for this model
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        # Synchronize processes after loading
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        # Verify that DDP is working correctly
        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        
        This method saves model parameters, optimizer state, and other training state
        to checkpoint files for later resumption of training.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        # Save models
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        # Save EMA parameters
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        # Save miscellaneous state (optimizer, step, etc.)
        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.fp16_mode == 'amp':
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.fp16_mode == 'inflat_all':
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        
        This method updates the EMA parameters using the current model parameters
        according to the formula: ema_param = ema_rate * ema_param + (1 - ema_rate) * master_param
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        
        This method verifies that parameters are consistent across all processes
        in distributed training, which is crucial for correct DDP operation.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def run_step(self, data_list):
        """
        Run a training step.
        
        This method performs one complete training step:
        1. Zero gradients
        2. For each micro-batch in the batch split:
           a. Compute loss
           b. Scale loss for gradient accumulation
           c. Backward pass
        3. Apply gradient clipping if configured
        4. Update parameters
        5. Update learning rate if scheduler is configured
        6. Update EMA parameters if running on master process
        7. Collect and return logs
        
        Args:
            data_list: List of data batches for gradient accumulation
            
        Returns:
            dict: Dictionary containing loss values and other metrics for logging
        """
        step_log = {'loss': {}, 'status': {}}
        # Set up context managers for mixed precision and elastic memory
        amp_context = partial(torch.autocast, device_type='cuda') if self.fp16_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    # Forward pass - compute loss
                    # print("start training loss")
                    loss, status = self.training_losses(**mb_data)
                    # print("end training loss")
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.fp16_mode == 'amp':
                    # AMP backward pass with scaling
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == 'inflat_all':
                    # Manual FP16 backward pass with scaling
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    # Standard backward pass
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
                
        ## gradient clip
        if self.grad_clip is not None:
            if self.fp16_mode == 'amp':
                # Unscale before clipping for AMP
                self.scaler.unscale_(self.optimizer)
            elif self.fp16_mode == 'inflat_all':
                # Convert and unscale gradients for manual FP16
                model_grads_to_master_grads(self.model_params, self.master_params)
                self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            
            # Apply gradient clipping
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
                
        ## step
        if self.fp16_mode == 'amp':
            # AMP optimizer step with scale adjustment
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.fp16_mode == 'inflat_all':
            # Manual FP16 step with overflow detection and scale adjustment
            prev_scale = 2 ** self.log_scale
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                self.optimizer.step()
                master_params_to_model_params(self.model_params, self.master_params)
                self.log_scale += self.fp16_scale_growth
            else:
                self.log_scale -= 1
        else:
            # Standard optimizer step with NaN detection
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
                
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
