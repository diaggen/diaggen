import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from trellis import models, datasets, trainers
from trellis.utils.dist_utils import setup_dist


def find_ckpt(cfg):
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        # print(rank,local_rank, world_size)
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed random number generators for reproducibility
    setup_rng(rank)
    print("Load training dataset ...")
    # Load dataset based on configuration
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    print("Load model ...")
    # Build models dynamically based on configuration
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }


    print("Load trainer ...")
    # Initialize trainer with models, dataset and configuration
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, 
        output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)

    # Start training or profiling based on configuration
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()

if __name__ == '__main__':
    # Parse command line arguments and load configuration
    parser = argparse.ArgumentParser()
    ## Config file
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## Input/output and checkpoint options
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    ## Debug options
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## Distributed training options
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')

    # Process arguments
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    
    # Load configuration from JSON file
    config = json.load(open(opt.config, 'r'))
    
    # Combine command-line arguments and JSON config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Create output directory and save configuration (only on main node)
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config for reproducibility
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run training with optional auto-retry on errors
    if cfg.auto_retry == 0:
        # No retries - run once
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        # Run training with retries
        for rty in range(cfg.auto_retry):
            print(f'Run {rty + 1}/{cfg.auto_retry}')
            cfg = find_ckpt(cfg)
            if cfg.num_gpus > 1:
                mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
            else:
                main(0, cfg)
            
            if rty < cfg.auto_retry - 1:
                print(f'Completed run {rty + 1}/{cfg.auto_retry}')
            else:
                print(f'All {cfg.auto_retry} runs completed')