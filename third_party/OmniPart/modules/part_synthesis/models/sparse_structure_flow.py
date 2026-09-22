"""
This file implements a Sparse Structure Flow model for 3D data generation or transformation.
It contains a transformer-based architecture that processes 3D volumes by:
1. Embedding timesteps for diffusion/flow-based modeling
2. Patchifying 3D inputs for efficient processing
3. Using cross-attention mechanisms to condition the generation on external features
4. Supporting various positional encoding schemes for 3D data

The model is designed for high-dimensional structure generation with conditional inputs
and follows a transformer-based architecture similar to DiT (Diffusion Transformers).
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    This is crucial for diffusion models where the model needs to know
    which noise level (timestep) it's currently operating at.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Initialize the timestep embedder.
        
        Args:
            hidden_size: Dimension of the output embeddings
            frequency_embedding_size: Dimension of the intermediate frequency embeddings
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings similar to positional encodings in transformers.
        
        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # Implementation based on OpenAI's GLIDE repository
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Embed timesteps into vectors.
        
        Args:
            t: Timesteps to embed [batch_size]
            
        Returns:
            Embedded timesteps [batch_size, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    """
    A transformer-based model for processing 3D data with conditional inputs.
    The model patchifies 3D volumes, processes them with transformer blocks,
    and then reconstructs the 3D volume at the output.
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
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        """
        Initialize the Sparse Structure Flow model.
        
        Args:
            resolution: Input resolution (assumes cubic inputs of shape [resolution, resolution, resolution])
            in_channels: Number of input channels
            model_channels: Number of model's internal channels
            cond_channels: Number of channels in conditional input
            out_channels: Number of output channels
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads (defaults to model_channels // num_head_channels)
            num_head_channels: Number of channels per attention head
            mlp_ratio: Ratio for MLP hidden dimension relative to model_channels
            patch_size: Size of patches for patchifying the input
            pe_mode: Type of positional encoding ("ape" for absolute, "rope" for rotary)
            use_fp16: Whether to use FP16 precision for most operations
            use_checkpoint: Whether to use gradient checkpointing to save memory
            share_mod: Whether to share modulation layers across blocks
            qk_rms_norm: Whether to use RMS normalization for query and key in self-attention
            qk_rms_norm_cross: Whether to use RMS normalization for query and key in cross-attention
        """
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
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # Timestep embedding network
        self.t_embedder = TimestepEmbedder(model_channels)
        
        # Optional shared modulation for all blocks
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        # Set up positional encoding
        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            # Create a grid of 3D coordinates for each patch position
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        # Input projection layer
        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        
        # Transformer blocks with cross-attention for conditioning
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        # Output projection layer
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        # Initialize model weights
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the transformer blocks of the model to float16 for improved efficiency.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the transformer blocks of the model back to float32 (e.g., for inference).
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model using carefully chosen initialization schemes.
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

        # Zero-out adaLN modulation layers to ensure stable training initially
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers to ensure initial predictions are near zero
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, resolution, resolution, resolution]
            t: Timestep tensor of shape [batch_size]
            cond: Conditional input tensor
            
        Returns:
            Output tensor of shape [batch_size, out_channels, resolution, resolution, resolution]
        """
        # Validate input shape
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        # Patchify the input volume and reshape for transformer processing
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # [B, num_patches, patch_dim]

        # Project to model dimension
        h = self.input_layer(h)
        
        # Add positional embeddings
        h = h + self.pos_emb[None]
        
        # Get timestep embeddings
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
            
        # Convert to appropriate dtype for computation
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        # print("transfer cond")
        # print("*" * 20)
        # print(cond.shape) # torch.Size([4, 4122, 1024])
        # Process through transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, cond)
        
        # print("transferred ")
            
        # Convert back to original dtype
        h = h.type(x.dtype)
        
        # Final normalization and projection
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        # Reshape and unpatchify to get final 3D output
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
