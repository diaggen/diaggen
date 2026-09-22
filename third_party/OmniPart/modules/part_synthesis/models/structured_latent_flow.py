from typing import *
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin


class SparseResBlock3d(nn.Module):
    """
    3D Sparse Residual Block with time embedding conditioning.
    
    This block performs normalization, convolution operations on sparse tensors,
    and incorporates time embeddings via adaptive layer normalization.
    Supports optional up/downsampling.
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        # First normalization and convolution
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        
        # Second convolution initialized to zero for stable training
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        
        # Time embedding projection for adaptive layer norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        
        # Skip connection with linear projection if channel dimensions change
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        
        # Optional up/downsampling
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """Apply up/downsampling if configured"""
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        """
        Forward pass of the residual block.
        
        Args:
            x: Input sparse tensor
            emb: Time embedding tensor
            
        Returns:
            Processed sparse tensor
        """
        # print(f"number of points in the input: {x.coords.shape[0]}")
        # Project embedding to scale and shift factors
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        # Apply up/downsampling if needed
        x = self._updown(x)
        
        # Main processing path
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        # Apply adaptive layer norm using scale and shift from time embedding
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        
        # Residual connection
        h = h + self.skip_connection(x)

        return h
    

class SLatFlowModel(nn.Module):
    """
    Structured Latent Flow Model for 3D generative modeling.
    
    This model combines sparse convolutions with transformer blocks and 
    supports conditional generation. It uses a U-Net-like architecture with
    skip connections and has optional mixed precision support.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Validate configurations
        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        # Time step embedder
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # Shared modulation for all transformer blocks if enabled
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.part_max_size = 50

        # Positional embedding for transformer blocks
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
            self.part_pe = nn.Embedding(self.part_max_size + 1, model_channels)  # +1 for overall object
            
        self.part_pe_proj = nn.Linear(model_channels, model_channels)

        # Mask embedding
        self.dinov2_hidden_size = 1024
        self.mask_group_emb_dim = 128

        self.mask_group_emb = nn.Embedding(self.part_max_size + 1, self.mask_group_emb_dim)  # +1 for background
        self.mask_group_emb_proj = nn.Linear(self.mask_group_emb_dim, self.dinov2_hidden_size)

        # Input projection layer
        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        # Input processing blocks (downsampling path)
        self.input_blocks = nn.ModuleList([])
        # print(f"io_block_channels: {io_block_channels}")  # io_block_channels: [128]
        # print(f"model_channels: {model_channels}") # model_channels: 1024

        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                # Add regular residual blocks at current resolution
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                # Add downsampling block at the end of each resolution level
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
            
        # Core transformer blocks
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        # Output processing blocks (upsampling path)
        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                # Add upsampling block at the beginning of each resolution level
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                # Add regular residual blocks at current resolution
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
            
        # Final output projection
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)
        
        # Initialize model weights
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        # else:
        #     self.convert_to_fp32()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16 for mixed precision training.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model back to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """
        Initialize model weights with specialized initialization for different components.
        """
        # Initialize transformer layers with Xavier uniform initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP with normal distribution
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers for stable training
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers for stable training
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        # part embedding initialization
        nn.init.zeros_(self.part_pe_proj.weight)
        nn.init.zeros_(self.part_pe_proj.bias)

        # Initialize layer positional embeddings
        self.part_pe.weight.data.normal_(mean=0.0,std=0.02)

        # Initialize group embedding
        nn.init.zeros_(self.mask_group_emb_proj.weight)
        nn.init.zeros_(self.mask_group_emb_proj.bias)

        self.mask_group_emb.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> sp.SparseTensor:
        """
        Forward pass of the Structured Latent Flow model.
        
        Args:
            x: Input sparse tensor
            t: Timestep embedding inputs
            cond: Conditional input for cross-attention
            **kwargs: Additional arguments, including part_layouts if available
            
        Returns:
            Output sparse tensor
        """

        # x = x.type(self.dtype)
        # t = t.type(self.dtype)
        # cond = cond.type(self.dtype)
        input_dtype = x.dtype

        masks = kwargs['masks']  # [b, h, w]
        
        # Ensure masks are always long type regardless of source
        masks = masks.long()  # Explicitly convert to long type for embedding
        masks = rearrange(masks, 'b h w -> b (h w)')  # [b, h*w]
        masks_emb = self.mask_group_emb(masks)  # [b, h*w, 128]
        masks_emb = self.mask_group_emb_proj(masks_emb)  # [b, h*w, 1024]
        group_emb = torch.zeros((cond.shape[0], cond.shape[1], masks_emb.shape[2]), device=cond.device, dtype=cond.dtype)
        group_emb[:, :masks_emb.shape[1], :] = masks_emb
        cond = cond + group_emb
        cond = cond.type(self.dtype)

        # Store original batch IDs for later restoration
        original_batch_ids = x.coords[:, 0].clone()
        
        # Create new batch IDs to represent individual parts (instead of batches)
        new_batch_ids = torch.zeros_like(original_batch_ids)
        
        # Assign unique IDs to each part across all batches
        part_layouts = kwargs['part_layouts']
        part_id = 0
        len_before = 0
        batch_last_partid = []
        for batch_idx, part_layout in enumerate(part_layouts):
            for layout_idx, layout in enumerate(part_layout):
                adjusted_layout = slice(layout.start + len_before, layout.stop + len_before, layout.step)
                new_batch_ids[adjusted_layout] = part_id
                part_id += 1
            
            batch_last_partid.append(part_id)
            len_before += part_layout[-1].stop
        
        # Project input to model dimensions and convert to target dtype
        x = self.input_layer(x).type(self.dtype)

        x = sp.SparseTensor(
                feats = x.feats,
                coords = torch.cat([new_batch_ids.view(-1, 1), x.coords[:, 1:]], dim=1),)
        
        # Process timestep embedding and condition input
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        t_emb_updown = []
        for batch_idx, part_layout in enumerate(part_layouts):
            t_emb_updown_batch = t_emb[batch_idx:batch_idx+1].repeat(len(part_layout), 1)
            t_emb_updown.append(t_emb_updown_batch)
        t_emb_updown = torch.cat(t_emb_updown, dim=0).type(self.dtype)
        
        # Store features for skip connections
        skips = []
        
        # Downsampling path through input blocks
        for block in self.input_blocks:
            x = block(x, t_emb_updown)
            skips.append(x.feats)
        
        # Store part-wise batch IDs before transformer processing
        part_wise_batch_ids = x.coords[:, 0].clone()
        
        # Convert to batch-wise IDs for transformer blocks
        new_transformer_batch_ids = torch.zeros_like(part_wise_batch_ids)
        part_ids_in_each_object = torch.zeros_like(part_wise_batch_ids)
        start_reform = 0
        last_part_id = 0
        for part_id in batch_last_partid:
            mask = (part_wise_batch_ids >= last_part_id) & (part_wise_batch_ids < part_id)
            new_transformer_batch_ids[mask] = start_reform
            part_ids_in_each_object[mask] = part_wise_batch_ids[mask] - last_part_id
            last_part_id = part_id
            start_reform += 1
        
        # Update coordinates with batch-wise IDs for transformer processing
        h = sp.SparseTensor(
            feats = x.feats,
            coords = torch.cat([new_transformer_batch_ids.view(-1, 1), x.coords[:, 1:]], dim=1))
        
        # Add positional embeddings for transformer blocks
        if self.pe_mode == "ape":
            # Add absolute positional embeddings to spatial coordinates
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
            # Part-with PE; overall is 0
            part_pe = self.part_pe(part_ids_in_each_object)
            part_pe = self.part_pe_proj(part_pe)
            h = h + part_pe.type(self.dtype)
        
        else:
            raise NotImplementedError
            
        # Process with transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, cond)
        
        h = x.replace(feats=h.feats, coords=torch.cat([part_wise_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1))
        
        # Upsampling path with output blocks and skip connections
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb_updown)
            else:
                h = block(h, t_emb_updown)
        
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(input_dtype))
        h = sp.SparseTensor(
            feats = h.feats,
            coords = torch.cat([original_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1))
        
        return h
    

class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    Structured Latent Flow Model with elastic memory management.
    
    This class extends SLatFlowModel with memory-efficient operations,
    allowing training with limited VRAM by dynamically managing memory
    allocation for sparse tensors.
    """
    pass