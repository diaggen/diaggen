import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention import FeedForward
from transformers.generation.logits_process import LogitsProcessor
from typing import List, Literal, Optional

from modules.bbox_gen.modules.norm import GroupNorm32, ChannelLayerNorm32


class GroupEmbedding(nn.Module):
    def __init__(self, max_group_size, hidden_size=64):
        super().__init__()

        self.group_embedding = nn.Embedding(max_group_size + 1, hidden_size)  # +1 for background
        self.group_embedding.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, masks):
        batch_size, height, width = masks.shape
        masks_flat = masks.reshape(batch_size, -1)
        embeddings = self.group_embedding(masks_flat)
        embeddings = embeddings.reshape(batch_size, height, width, -1)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings


class MultiModalProjector(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class MeshDecodeLogitsProcessor(LogitsProcessor):
    def __init__(self, bins, BOS_id, EOS_id, PAD_id, vertices_num=8):
        super().__init__()
        self.bins = bins
        self.BOS_id = BOS_id
        self.EOS_id = EOS_id
        self.PAD_id = PAD_id
        self.filter_value = -float('inf')
        self.vertices_num = vertices_num
    
    def force_token(self, scores, token_id):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, token_id] = False
        scores[mask] = self.filter_value
    
    def __call__(self, input_ids, scores):
        # # all rules:
        # # 1. first token: BOS
        current_len = input_ids.shape[-1]
        if current_len == 0:
            # force bos
            self.force_token(scores, self.BOS_id)
        elif current_len <= self.vertices_num * 3 + 1:
            scores[:, self.bins:] = self.filter_value
        else:
            scores[:, self.BOS_id] = self.filter_value
            scores[:, self.PAD_id] = self.filter_value
            
            effective_tokens = current_len - 1
            complete_boxes = effective_tokens % (self.vertices_num * 3) == 0
            # print(effective_tokens, complete_boxes)
            if not complete_boxes:
                scores[:, self.EOS_id] = self.filter_value

        return scores
    

def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    

class SparseStructureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.dtype = torch.float16
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )
        
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor):
        h = self.input_layer(x)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)
        return h