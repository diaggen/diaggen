from dataclasses import dataclass
import os
import sys
import torch
import trimesh
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from einops import rearrange

from modules.bbox_gen.models.image_encoder import DINOv2ImageEncoder
from modules.bbox_gen.config import parse_structured
from modules.bbox_gen.models.bboxopt import BBoxOPT, BBoxOPTConfig
from modules.bbox_gen.utils.bbox_tokenizer import BoundsTokenizerDiag
from modules.bbox_gen.models.bbox_gen_models import GroupEmbedding, MultiModalProjector, MeshDecodeLogitsProcessor, SparseStructureEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.dirname(os.path.dirname(current_dir))
partfield_dir = os.path.join(modules_dir, 'PartField')
if partfield_dir not in sys.path:
    sys.path.insert(0, partfield_dir)
import importlib.util
from partfield.config import default_argument_parser, setup


class BboxGen(nn.Module):

    @dataclass
    class Config:
        # encoder config
        encoder_dim_feat: int = 3
        encoder_dim: int = 64
        encoder_heads: int = 4
        encoder_token_num: int = 256
        encoder_qkv_bias: bool = False
        encoder_use_ln_post: bool = True
        encoder_use_checkpoint: bool = False
        encoder_num_embed_freqs: int = 8
        encoder_embed_include_pi: bool = False
        encoder_init_scale: float = 0.25
        encoder_random_fps: bool = True
        encoder_learnable_query: bool = False
        encoder_layers: int = 4
        group_embedding_dim: int = 64

        # decoder config
        vocab_size: int = 518
        decoder_hidden_size: int = 1536
        decoder_num_hidden_layers: int = 24
        decoder_ffn_dim: int = 6144
        decoder_heads: int = 16
        decoder_use_flash_attention: bool = True
        decoder_gradient_checkpointing: bool = True

        # data config
        bins: int = 64
        BOS_id: int = 64
        EOS_id: int = 65
        PAD_id: int = 66
        max_length: int = 2187  # bos + 50x2x3 + 1374 + 512
        voxel_token_length: int = 1886
        voxel_token_placeholder: int = -1

        # tokenizer config
        max_group_size: int = 50

        # voxel encoder
        partfield_encoder_path: str = ""

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)

        self.image_encoder = DINOv2ImageEncoder(
            model_name="facebook/dinov2-with-registers-large",
        )

        self.image_projector = MultiModalProjector(
            in_features=(1024 + self.cfg.group_embedding_dim),
            out_features=self.cfg.decoder_hidden_size,
        )

        self.group_embedding = GroupEmbedding(
            max_group_size=self.cfg.max_group_size,
            hidden_size=self.cfg.group_embedding_dim,
        )

        self.decoder_config = BBoxOPTConfig(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.decoder_hidden_size,
            num_hidden_layers=self.cfg.decoder_num_hidden_layers,
            ffn_dim=self.cfg.decoder_ffn_dim,
            max_position_embeddings=self.cfg.max_length,
            num_attention_heads=self.cfg.decoder_heads,
            pad_token_id=self.cfg.PAD_id,
            bos_token_id=self.cfg.BOS_id,
            eos_token_id=self.cfg.EOS_id,
            use_cache=True,
            init_std=0.02,
        )

        if self.cfg.decoder_use_flash_attention:
            self.decoder: BBoxOPT = AutoModelForCausalLM.from_config(
                self.decoder_config, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        else:
            self.decoder: BBoxOPT = AutoModelForCausalLM.from_config(
                self.decoder_config, 
            )
        if self.cfg.decoder_gradient_checkpointing:
            self.decoder.gradient_checkpointing_enable()

        self.logits_processor = LogitsProcessorList()

        self.logits_processor.append(MeshDecodeLogitsProcessor(
            bins=self.cfg.bins,
            BOS_id=self.cfg.BOS_id,
            EOS_id=self.cfg.EOS_id,
            PAD_id=self.cfg.PAD_id,
            vertices_num=2,
        ))
        self.tokenizer = BoundsTokenizerDiag(
            bins=self.cfg.bins,
            BOS_id=self.cfg.BOS_id,
            EOS_id=self.cfg.EOS_id,
            PAD_id=self.cfg.PAD_id,
        )

        self._load_partfield_encoder()

        self.partfield_voxel_encoder = SparseStructureEncoder(
            in_channels=451,
            channels=[448, 448, 448, 1024],
            latent_channels=448,
            num_res_blocks=1,
            num_res_blocks_middle=1,
            norm_type="layer",
        )
    
    
    def _load_partfield_encoder(self):
        # Load PartField encoder
        model_spec = importlib.util.spec_from_file_location(
                "partfield.partfield_encoder", 
                os.path.join(partfield_dir, "partfield", "partfield_encoder.py")
            )
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        Model = model_module.Model
        parser = default_argument_parser()
        args = []
        args.extend(["-c", os.path.join(partfield_dir, "configs/final/demo.yaml")])
        args.append("--opts")
        args.extend(["continue_ckpt", self.cfg.partfield_encoder_path])
        parsed_args = parser.parse_args(args)
        cfg = setup(parsed_args, freeze=False)
        self.partfield_encoder = Model(cfg)
        self.partfield_encoder.eval()
        weights = torch.load(self.cfg.partfield_encoder_path, weights_only=False)["state_dict"]
        self.partfield_encoder.load_state_dict(weights)
        for param in self.partfield_encoder.parameters():
            param.requires_grad = False
        print("PartField encoder loaded")
    
    def _prepare_lm_inputs(self, voxel_token, input_ids):
        inputs_embeds = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.cfg.decoder_hidden_size, device=input_ids.device, dtype=voxel_token.dtype)
        voxel_token_mask = (input_ids == self.cfg.voxel_token_placeholder)
        inputs_embeds[voxel_token_mask] = voxel_token.view(-1, self.cfg.decoder_hidden_size)

        inputs_embeds[~voxel_token_mask] = self.decoder.get_input_embeddings()(input_ids[~voxel_token_mask]).to(dtype=inputs_embeds.dtype)

        attention_mask = (input_ids != self.cfg.PAD_id)
        return inputs_embeds, attention_mask.long()
    
    def forward(self, batch):

        image_latents = self.image_encoder(batch['images'])
        masks = batch['masks']
        masks_emb = self.group_embedding(masks)
        masks_emb = rearrange(masks_emb, 'b c h w -> b (h w) c') # B x Q x C
        group_emb = torch.zeros((image_latents.shape[0], image_latents.shape[1], masks_emb.shape[2]), device=image_latents.device, dtype=image_latents.dtype)
        group_emb[:, :masks_emb.shape[1], :] = masks_emb
        image_latents = torch.cat([image_latents, group_emb], dim=-1) 
        image_latents = self.image_projector(image_latents)

        points = batch['points'][..., :3]
        rot_matrix = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=points.device, dtype=points.dtype)
        rot_points = torch.matmul(points, rot_matrix)
        rot_points = rot_points * (2 * 0.9)  # from (-0.5, 0.5) to (-1, 1)

        partfield_feat = self.partfield_encoder.encode(rot_points)
        feat_volume = torch.zeros((points.shape[0], 448, 64, 64, 64), device=partfield_feat.device, dtype=partfield_feat.dtype)
        whole_voxel_index = batch['whole_voxel_index']  # (b, m, 3)

        batch_size, num_points = whole_voxel_index.shape[0], whole_voxel_index.shape[1]
        batch_indices = torch.arange(batch_size, device=whole_voxel_index.device).unsqueeze(1).expand(-1, num_points)  # (b, m)
        batch_flat = batch_indices.flatten()  # (b*m,)
        x_flat = whole_voxel_index[..., 0].flatten()  # (b*m,)
        y_flat = whole_voxel_index[..., 1].flatten()  # (b*m,)
        z_flat = whole_voxel_index[..., 2].flatten()  # (b*m,)
        partfield_feat_flat = partfield_feat.reshape(-1, 448)  # (b*m, 448)
        feat_volume[batch_flat, :, x_flat, y_flat, z_flat] = partfield_feat_flat
        
        xyz_volume = torch.zeros((points.shape[0], 3, 64, 64, 64), device=points.device, dtype=points.dtype)
        xyz_volume[batch_flat, :, x_flat, y_flat, z_flat] = points.reshape(-1, 3)
        feat_volume = torch.cat([feat_volume, xyz_volume], dim=1)

        feat_volume = self.partfield_voxel_encoder(feat_volume)
        feat_volume = rearrange(feat_volume, 'b c x y z -> b (x y z) c')

        voxel_token = torch.cat([image_latents, feat_volume], dim=1) # B x N x D

        input_ids = batch['input_ids']
        inputs_embeds, attention_mask = self._prepare_lm_inputs(voxel_token, input_ids)
        output = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        return {
            "logits": output.logits,
        }

    def gen_mesh_from_bounds(self, bounds, random_color):
        bboxes = []
        for j in range(bounds.shape[0]):
            bbox = trimesh.primitives.Box(bounds=bounds[j])
            color = random_color[j]
            bbox.visual.vertex_colors = color
            bboxes.append(bbox)
        mesh = trimesh.Scene(bboxes)
        return mesh
    
    def generate(self, batch):

        image_latents = self.image_encoder(batch['images'])
        masks = batch['masks']
        masks_emb = self.group_embedding(masks)
        masks_emb = rearrange(masks_emb, 'b c h w -> b (h w) c') # B x Q x C
        group_emb = torch.zeros((image_latents.shape[0], image_latents.shape[1], masks_emb.shape[2]), device=image_latents.device, dtype=image_latents.dtype)
        group_emb[:, :masks_emb.shape[1], :] = masks_emb
        image_latents = torch.cat([image_latents, group_emb], dim=-1) 
        image_latents = self.image_projector(image_latents)

        points = batch['points'][..., :3]
        rot_matrix = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=points.device, dtype=points.dtype)
        rot_points = torch.matmul(points, rot_matrix)
        rot_points = rot_points * (2 * 0.9)  # from (-0.5, 0.5) to (-1, 1)

        partfield_feat = self.partfield_encoder.encode(rot_points)
        feat_volume = torch.zeros((points.shape[0], 448, 64, 64, 64), device=partfield_feat.device, dtype=partfield_feat.dtype)
        whole_voxel_index = batch['whole_voxel_index']  # (b, m, 3)

        batch_size, num_points = whole_voxel_index.shape[0], whole_voxel_index.shape[1]
        batch_indices = torch.arange(batch_size, device=whole_voxel_index.device).unsqueeze(1).expand(-1, num_points)  # (b, m)
        batch_flat = batch_indices.flatten()  # (b*m,)
        x_flat = whole_voxel_index[..., 0].flatten()  # (b*m,)
        y_flat = whole_voxel_index[..., 1].flatten()  # (b*m,)
        z_flat = whole_voxel_index[..., 2].flatten()  # (b*m,)
        partfield_feat_flat = partfield_feat.reshape(-1, 448)  # (b*m, 448)
        feat_volume[batch_flat, :, x_flat, y_flat, z_flat] = partfield_feat_flat

        xyz_volume = torch.zeros((points.shape[0], 3, 64, 64, 64), device=points.device, dtype=points.dtype)
        xyz_volume[batch_flat, :, x_flat, y_flat, z_flat] = points.reshape(-1, 3)
        feat_volume = torch.cat([feat_volume, xyz_volume], dim=1)

        feat_volume = self.partfield_voxel_encoder(feat_volume)
        feat_volume = rearrange(feat_volume, 'b c x y z -> b (x y z) c')

        voxel_token = torch.cat([image_latents, feat_volume], dim=1) # B x N x D

        meshes = []
        mesh_names = []
        bboxes = []

        output = self.decoder.generate(
            inputs_embeds=voxel_token,
            max_new_tokens=self.cfg.max_length - voxel_token.shape[1],
            logits_processor=self.logits_processor,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=0.5,
            use_cache=True,
        )

        for i in range(output.shape[0]):
            bounds = self.tokenizer.decode(output[i].detach().cpu().numpy(), coord_rg=(-0.5, 0.5))
            # mesh = self.gen_mesh_from_bounds(bounds, batch['random_color'][i])
            # meshes.append(mesh)
            mesh_names.append("topk=5")
            bboxes.append(bounds)

        return {
            # 'meshes': meshes,
            'mesh_names': mesh_names,
            'bboxes': bboxes,
        }


