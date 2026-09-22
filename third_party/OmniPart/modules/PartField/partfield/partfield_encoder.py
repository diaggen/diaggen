import torch
import lightning.pytorch as pl
# from .dataloader import Demo_Dataset, Demo_Remesh_Dataset, Correspondence_Demo_Dataset
from torch.utils.data import DataLoader
from partfield.model.UNet.model import ResidualUNet3D
from partfield.model.triplane import TriplaneTransformer, get_grid_coord #, sample_from_planes, Voxel2Triplane
from partfield.model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import skimage
import numpy as np
import h5py
import torch.distributed as dist
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import json
import gc
import time
from plyfile import PlyData, PlyElement


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1, 
                                      out_activation="tanh", 
                                      n_neurons=64, #64
                                      n_hidden_layers=6) #6
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=self.use_2d_feat) #.cuda()
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192, 
                                out_activation="GELU", 
                                n_neurons=64, #64
                                n_hidden_layers=6) #6

    # def predict_dataloader(self):
    #     if self.cfg.remesh_demo:
    #         dataset = Demo_Remesh_Dataset(self.cfg)        
    #     elif self.cfg.correspondence_demo:
    #         dataset = Correspondence_Demo_Dataset(self.cfg)
    #     else:
    #         dataset = Demo_Dataset(self.cfg)

    #     dataloader = DataLoader(dataset, 
    #                         num_workers=self.cfg.dataset.val_num_workers,
    #                         batch_size=self.cfg.dataset.val_batch_size,
    #                         shuffle=False, 
    #                         pin_memory=True,
    #                         drop_last=False)
        
    #     return dataloader           


    @torch.no_grad()
    def encode(self, points):

        N = points.shape[0]
        # assert N == 1
        pcd = points[..., :3]

        pc_feat = self.pvcnn(pcd, pcd)

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        tensor_vertices = pcd.reshape(N, -1, 3).cuda().to(pcd.dtype)
        point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C
        # point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)
        point_feat = point_feat.reshape(N, -1, 448)

        return point_feat