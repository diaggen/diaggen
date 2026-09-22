from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler, DuplicatedDataset

import datetime

from tqdm import tqdm
class Trainer:
    """
    Base class for training.
    
    This abstract class defines the core training loop and utilities
    that are common across different training tasks. Specific training
    implementations should inherit from this class and implement the
    abstract methods.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs
    ):
        """
        Initialize the trainer with models, dataset, and training configuration.
        
        Args:
            models: Dictionary of models to train
            dataset: Dataset to train on
            output_dir: Directory to save outputs
            load_dir: Directory to load checkpoints from
            step: Current training step (for resuming)
            max_steps: Maximum number of training steps
            batch_size: Global batch size across all GPUs
            batch_size_per_gpu: Batch size per GPU (alternative to batch_size)
            batch_split: Number of microbatches to split each batch into
            optimizer: Dictionary of optimizer configurations
            lr_scheduler: Learning rate scheduler configuration
            elastic: Configuration for elastic training
            grad_clip: Value for gradient clipping
            ema_rate: Rate for exponential moving average
            fp16_mode: Mode for mixed precision training ('inflat_all' or 'amp')
            fp16_scale_growth: Growth rate for mixed precision scaling
            finetune_ckpt: Checkpoint path for fine-tuning
            log_param_stats: Whether to log parameter statistics
            prefetch_data: Whether to prefetch data to GPU
            i_print: Interval for printing progress
            i_log: Interval for logging metrics
            i_sample: Interval for sampling/visualization
            i_save: Interval for saving checkpoints
            i_ddpcheck: Interval for checking DDP consistency
        """
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        # Set up distributed training configuration
        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        # Calculate batch size parameters
        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        # Initialize models, optimizers, etc.
        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)

        # Load checkpoint or initialize from scratch
        self.step = 0
        if load_dir is not None and step is not None:
            # print("load from ...x")
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            # print("finetune from ...x")
            self.finetune_from(finetune_ckpt)
        # print(finetune_ckpt)
        # print("loading finished......")
        
        # Set up output directories and tensorboard writer on master process
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        # Check DDP setup for multi-GPU training
        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        """
        Get the device that the models are on.
        
        Returns:
            torch.device: The device of the first model parameter found.
        """
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and other components like optimizers, schedulers, etc.
        
        This abstract method must be implemented by subclasses to set up the
        specific models and related components needed for training.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader for training.
        
        Sets up the data sampler and dataloader with appropriate batch size,
        workers, and other configurations for efficient data loading.
        """
        print("original dataset size:", len(self.dataset))
        num_dataset = 128
        # Wrap your dataset in the DuplicatedDataset if it's too small
        print(f"Dataset size: {len(self.dataset)}")
        if len(self.dataset) < num_dataset:  # Adjust this threshold as needed
            from ..utils.data_utils import DuplicatedDataset

            self.dataset = DuplicatedDataset(self.dataset, repeat=num_dataset)
            print(f"Dataset duplicated to {len(self.dataset)} samples")
        
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )

        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        
        This method should be implemented to restore the training state
        from a saved checkpoint.
        
        Args:
            load_dir: Directory containing checkpoints
            step: Specific step to load, or 0 for latest
            
        Note: Should be called by all processes in distributed training.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        
        This method should be implemented to save the current training state
        to a checkpoint file.
        
        Note: Should be called only by the rank 0 process in distributed training.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        
        This method should be implemented to load pre-trained weights
        for fine-tuning.
        
        Args:
            finetune_ckpt: Path to checkpoint for fine-tuning
            
        Note: Should be called by all processes in distributed training.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        
        This method should be implemented to generate samples from the model
        for visualization.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose information
            **kwargs: Additional arguments
            
        Returns:
            dict: Dictionary of generated samples
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image for visualization.
        
        Args:
            sample: Data sample to visualize
            
        Returns:
            torch.Tensor or dict: Processed sample ready for visualization
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=4):
        """
        Sample images from the dataset for visualization.
        
        Creates a visualization of dataset samples and saves them to disk.
        
        Args:
            num_samples: Number of samples to visualize
        """
        # print(f"Starting snapshot_dataset with {num_samples} samples...")
        # print("Creating dataloader...")
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        # print("Dataloader created, retrieving batch...")
        data = next(iter(dataloader))
        # print("Batch retrieved, moving to device...")
        data = recursive_to_device(data, self.device)
        # print("Data moved to device, visualizing sample...")
        vis = self.visualize_sample(data)
        # print("Sample visualized, preparing to save images...")
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            # print(f"Saving image {name}...")
            max_images_per_file = 100
            
            # Get the number of images in this batch
            num_images = image.size(0)
            
            if num_images <= max_images_per_file:
                # If fewer than max_images_per_file, save them all in one file
                utils.save_image(
                    image,
                    os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                    nrow=int(np.sqrt(num_images)),
                    normalize=True,
                    value_range=self.dataset.value_range,
                )
            else:
                # If more than max_images_per_file, split into multiple files
                num_batches = (num_images + max_images_per_file - 1) // max_images_per_file
                
                for i in range(num_batches):
                    start_idx = i * max_images_per_file
                    end_idx = min((i + 1) * max_images_per_file, num_images)
                    
                    # Extract the current batch of images
                    batch_images = image[start_idx:end_idx]
                    batch_size_actual = batch_images.size(0)
                        
                    utils.save_image(
                        batch_images,
                        os.path.join(self.output_dir, 'samples', f'{name}_{i+1}.jpg'),
                        nrow=int(np.sqrt(batch_size_actual)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
            # print(f"Image {name} saved successfully")
        # print("snapshot_dataset completed")

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=50, batch_size=4, verbose=False):
        """
        Sample images from the model and save to disk.
        
        This function coordinates the generation of samples across all processes
        and gathers them on the master process for saving.
        
        Args:
            suffix: Suffix for the output directory name
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            verbose: Whether to print verbose information
            
        Note: This function should be called by all processes in distributed training.
        """
        # if self.world_size > 1:
        #     dist.barrier()
        # Print status message from the master process only
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        # Set default suffix to current step if none provided (used for organizing output files)
        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Calculate how many samples each process should generate in distributed setting
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        # # Generate samples using the model's snapshot implementation

        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                # Convert raw samples to visualizable format using the dataset's visualization method
                # vis = self.visualize_sample(samples[key]['value'])
                vis = self.visualize_sample({'x_0': samples[key]['value'], 'part_layouts': samples['layout']['value']})
                if isinstance(vis, dict):
                    # If visualization returns a dictionary, create multiple entries with different visualizations
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    # Remove the original entry since it's been replaced with specific visualizations
                    del samples[key]
                else:
                    # Otherwise, update the existing entry by replacing with visualization
                    samples[key] = {'value': vis, 'type': 'image'}

        # Remove the layout entry after processing
        if 'layout' in samples:
            del samples['layout']

        # Gather samples from all processes in distributed training setup
        if self.world_size > 1:
            for key in samples.keys():
                # if isinstance(samples[key]['value'], list):
                #     continue
                if key in ['sample', 'sample_gt', 'image']:
                    # Ensure tensor is contiguous in memory for efficient gathering operation
                    samples[key]['value'] = samples[key]['value'].contiguous()
                    # print(samples[key]['value'].shape)
                    if self.is_master:
                        # Create buffers on master process to receive data from all processes
                        all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                    else:
                        # Non-master processes don't need to allocate receive buffers
                        all_images = []
                    # Gather data from all processes to the master (rank 0)
                    dist.gather(samples[key]['value'], all_images, dst=0)
                    if self.is_master:
                        # Concatenate all gathered samples and limit to requested number
                        samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]


        # Save images to disk (only on master process)
        if self.is_master:
            # Create output directory for current snapshot
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys(): # Error: The size of tensor a (3) must match the size of tensor b (518) at non-singleton dimension 2
                # print(key)
                # print(samples[key])
                if samples[key]['type'] == 'image':
                    # print(f"Saving {key} images...")
                    # print(f"shape is {samples[key]['value'].shape}") # shape is torch.Size([64, 3, 3, 518, 518])
                    # Reshape [64, 3, 3, 518, 518] -> [64, 9, 518, 518]
                    if samples[key]['value'].ndim == 5:
                        value = samples[key]['value']
                        value = value.permute(1, 0, 2, 3, 4)
                        # print(f"Reshaped tensor from {value.shape} to {samples[key]['value'].shape}")
                        for indexx in range(value.shape[0]):
                            # Save image samples using torchvision utilities
                            utils.save_image(
                                value[indexx],
                                os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{index}_{indexx}.jpg'),
                                nrow=int(np.sqrt(num_samples)),  # Arrange images in a square grid
                                normalize=True,
                                value_range=self.dataset.value_range,  # Use dataset's specified value range for normalization
                            )
                    else:
                        image = samples[key]['value']
                        # Maximum number of images to save in one file
                        max_images_per_file = 100
                        
                        # Get the number of images in this batch
                        num_images = image.size(0)
                        
                        if num_images <= max_images_per_file:
                            # If fewer than max_images_per_file, save them all in one file
                            utils.save_image(
                                image,
                                os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                                nrow=int(np.sqrt(num_images)),
                                normalize=True,
                                value_range=self.dataset.value_range,
                            )
                        else:
                            # If more than max_images_per_file, split into multiple files
                            num_batches = (num_images + max_images_per_file - 1) // max_images_per_file
                            
                            for i in range(num_batches):
                                start_idx = i * max_images_per_file
                                end_idx = min((i + 1) * max_images_per_file, num_images)
                                
                                # Extract the current batch of images
                                batch_images = image[start_idx:end_idx]
                                batch_size_actual = batch_images.size(0)
                                
                                utils.save_image(
                                    batch_images,
                                    os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}_{i+1}.jpg'),
                                    nrow=int(np.sqrt(batch_size_actual)),
                                    normalize=True,
                                    value_range=self.dataset.value_range,
                                )
                    # print(f"key is ****{key}")
                elif samples[key]['type'] == 'number':
                    # Process and save numerical samples as images with annotations
                    min_val = samples[key]['value'].min()
                    max_val = samples[key]['value'].max()
                    # Normalize values to [0, 1] range for visualization
                    images = (samples[key]['value'] - min_val) / (max_val - min_val)
                    # Create a grid of images
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,  # Already normalized above
                    )
                    # Save the image with min/max annotations for reference
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min_val}, max: {max_val}',
                    )

        # Print completion message (master process only)
        if self.is_master:
            print(' Done.')
        
        # print(1111111111111111111111)
        # if self.world_size > 1:
        #     dist.barrier() 
        #     print(222222222222222222222)

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average of model parameters.
        
        This method should be implemented to maintain EMA versions of models
        for more stable inference.
        
        Note: Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if Distributed Data Parallel (DDP) is working properly.
        
        This method should verify that parameters are synchronized
        across processes in distributed training.
        
        Note: Should be called by all processes.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses from a minibatch of data.
        
        This method should be implemented to compute all loss components
        needed for training.
        
        Args:
            **mb_data: Minibatch data
            
        Returns:
            dict: Dictionary of loss components
        """
        pass
    
    def load_data(self):
        """
        Load a batch of data from the dataloader.
        
        If prefetching is enabled, alternates between using a pre-fetched
        batch and loading the next batch in the background.
        
        Returns:
            list: List of data dictionaries, split according to batch_split
        """
        if_print = False

        if if_print:
            print(f"[{self.step}] Beginning load_data() method")

        if self.prefetch_data:
            if self._data_prefetched is None:
                if if_print:
                    print(f"[{self.step}] No prefetched data, fetching initial batch")

                data_load = next(self.data_iterator)
                if if_print:
                    print(f"[{self.step}] Prefetching next batch in background")
                self._data_prefetched = recursive_to_device(data_load, self.device, non_blocking=True)
            data = self._data_prefetched
            if if_print:
                print(f"[{self.step}] Using prefetched data and loading next batch in background")
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            if if_print:
                print(f"[{self.step}] Prefetching disabled, loading data directly")
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # Split data into multiple microbatches if needed
        if isinstance(data, dict):
            if self.batch_split == 1:
                if if_print:
                    print(f"[{self.step}] Batch split=1, using single batch")
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                if if_print:
                    print(f"[{self.step}] Splitting batch (size {batch_size}) into {self.batch_split} microbatches")
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            if if_print:
                print(f"[{self.step}] Data is already a list with {len(data)} items")
            data_list = data
        else:
            if if_print:
                print(f"[{self.step}] ERROR: Unexpected data type: {type(data)}")
            raise ValueError('Data must be a dict or a list of dicts.')
        
        if if_print:
            print(f"[{self.step}] Returning data_list with {len(data_list)} items")
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a single training step.
        
        This method should be implemented to process a batch of data,
        compute losses, and update model parameters.
        
        Args:
            data_list: List of data batches
            
        Returns:
            dict: Dictionary of metrics/losses for logging
        """
        pass
    
    def run(self):
        """
        Run the full training loop.
        
        This method handles the main training loop, including:
        - Data loading
        - Running training steps
        - Logging metrics
        - Creating snapshots
        - Saving checkpoints
        - Monitoring progress
        """
        # if self.is_master:
        #     print('\nStarting training...')
        #     self.snapshot_dataset()
        # if self.step == 0:
        #     self.snapshot(suffix='init')
        # else: # resume
        #     self.snapshot(suffix=f'resume_step{self.step:07d}')

        self.snapshot_dataset()
        # if self.step != 0:
        #     self.snapshot(suffix=f'resume_step{self.step:07d}')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0

        # Initialize progress bar (only on master process)
        pbar = None
        if self.is_master:
            pbar = tqdm(
                initial=self.step,
                total=self.max_steps,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                leave=True
            )

        while self.step < self.max_steps:

            # print("self.step", self.step)
            # print("self.i_log", self.i_log) # 500
            # print("step:", self.step)
            time_start = time.time()
            # print("load data")
            data_list = self.load_data()
            # print("run step")
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1
            
            # Update progress bar on master process
            if self.is_master and pbar is not None:
                pbar.update(1)
                
                # Update progress bar description with speed and ETA
                if self.step % self.i_print == 0:
                    speed = self.i_print / (time_elapsed - time_last_print) * 3600
                    pbar_desc = f"Training | Step: {self.step}/{self.max_steps} | " \
                                f"Speed: {speed:.2f} steps/h | " \
                                f"Elapsed: {time_elapsed/3600:.2f}h | " \
                                f"ETA: {(self.max_steps-self.step)/speed:.2f}h"
                    pbar.set_description(pbar_desc)
                    time_last_print = time_elapsed

            # Check DDP synchronization at regular intervals
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Generate and save sample images at regular intervals
            # if self.step == 0 or self.step % self.i_sample == 0:
            #     print("snapshot")
            #     self.snapshot()

            # Handle logging on master process
            if self.is_master:

                # print("self.is_master")
                log.append((self.step, {}))
                
                # print(f"log append: {log}")
                # Log timing information
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log training metrics
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scaling factor for mixed precision training
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale
                
                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    log = []

                # Save checkpoint at regular intervals
                if self.step % self.i_save == 0:
                    self.save()

        # Clean up progress bar when training is complete
        if self.is_master and pbar is not None:
            pbar.close()
            
            # Final steps after training is complete
            self.snapshot(suffix='final')
            self.writer.close()
            print('Training finished.')
            
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop for performance analysis.
        
        Uses PyTorch's profiling tools to collect performance metrics.
        
        Args:
            wait: Number of steps to wait before profiling
            warmup: Number of warmup steps for profiling
            active: Number of active profiling steps
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()