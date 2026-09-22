import os
from pathlib import Path

import imageio
import torch

from modules.part_synthesis.appearance_evidence import (
    AppearanceSourcePart,
    write_appearance_bundle,
)
from modules.part_synthesis.utils import render_utils, postprocessing_utils
from modules.part_synthesis.representations.gaussian.gaussian_model import Gaussian


def save_parts_outputs(
    outputs,
    output_dir,
    simplify_ratio,
    save_video=True,
    save_glb=True,
    textured=True,
    *,
    seed: int,
):
    os.makedirs(output_dir, exist_ok=True)
    if not save_glb:
        raise ValueError(
            "save_glb=True is required to produce the OmniPart appearance bundle"
        )

    # num_parts = min(len(outputs['gaussian']), len(outputs['radiance_field']), len(outputs['mesh']))
    num_parts = min(len(outputs['gaussian']), len(outputs['mesh']))
    gs_list = []
    source_parts = []
    
    for i in range(num_parts):
        if i == 0:
            continue
        processed_mesh = postprocessing_utils.prepare_processed_mesh(
            outputs['mesh'][i],
            simplify=simplify_ratio,
        )
        if processed_mesh is None:
            continue
        if save_video:
            video = render_utils.render_video(outputs['gaussian'][i])['color']
            gaussian_video_path = f"{output_dir}/part{i}_gs_text.mp4"
            if os.path.exists(gaussian_video_path):
                os.remove(gaussian_video_path)
            imageio.mimsave(gaussian_video_path, video, fps=30)
            
            video = render_utils.render_video(outputs['radiance_field'][i])['color']
            rf_video_path = f"{output_dir}/part{i}_rf_text.mp4"
            if os.path.exists(rf_video_path):
                os.remove(rf_video_path)
            imageio.mimsave(rf_video_path, video, fps=30)
            
            video = render_utils.render_video(outputs['mesh'][i])['normal']
            mesh_video_path = f"{output_dir}/part{i}_mesh_text.mp4"
            if os.path.exists(mesh_video_path):
                os.remove(mesh_video_path)
            imageio.mimsave(mesh_video_path, video, fps=30)
            
        if save_glb:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][i],
                outputs['mesh'][i],
                simplify=simplify_ratio,  # Mesh simplification factor
                texture_size=1024,
                textured=textured,
                processed_mesh=processed_mesh,
            )
            glb_path = Path(output_dir) / f"part{i}.glb"
            if glb_path.exists():
                glb_path.unlink()
            glb.export(glb_path)
            part_index = len(source_parts)
            vertices, faces = processed_mesh
            source_parts.append(
                AppearanceSourcePart(
                    part_index=part_index,
                    source_output_index=i,
                    source_glb_name=glb_path.name,
                    vertices=vertices,
                    faces=faces,
                )
            )
            gs_list.append(outputs['gaussian'][i])

    if not source_parts:
        raise ValueError("no successful part GLBs are available for appearance evidence")
    merged_gaussian = merge_gaussians(gs_list)
    write_appearance_bundle(
        output_dir=output_dir,
        merged_gaussian=merged_gaussian,
        source_parts=source_parts,
        seed=seed,
    )
    merged_gaussian.save_ply(f"{output_dir}/merged_gs.ply")
    
    exploded_gs = exploded_gaussians(gs_list, explosion_scale=0.3)
    exploded_gs.save_ply(f"{output_dir}/exploded_gs.ply")


def merge_gaussians(gaussians_list):
    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(**first_gaussian.init_params, device=first_gaussian.device)
    
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []
    
    for gaussian in gaussians_list:
        if (gaussian.sh_degree != first_gaussian.sh_degree or 
            not torch.allclose(gaussian.aabb, first_gaussian.aabb)):
            raise ValueError("All Gaussian objects must have the same sh_degree and aabb parameters")
            
        if gaussian._xyz is not None:
            xyz_list.append(gaussian._xyz)
        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)
    
    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)
    
    return merged_gaussian


def exploded_gaussians(gaussians_list, explosion_scale=0.4):

    if not gaussians_list:
        raise ValueError("gaussians_list is empty")

    first_gaussian = gaussians_list[0]
    merged_gaussian = Gaussian(**first_gaussian.init_params, device=first_gaussian.device)
    
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    opacity_list = []
    
    all_centers = []
    for gaussian in gaussians_list:
        if gaussian._xyz is not None:
            center = gaussian.get_xyz.mean(dim=0)
            all_centers.append(center)
    
    if not all_centers:
        raise ValueError("No valid gaussians with xyz data found")
    
    all_centers = torch.stack(all_centers)
    global_center = all_centers.mean(dim=0)
    
    for i, gaussian in enumerate(gaussians_list):
        if (gaussian.sh_degree != first_gaussian.sh_degree or 
            not torch.allclose(gaussian.aabb, first_gaussian.aabb)):
            raise ValueError("All Gaussian objects must have the same sh_degree and aabb parameters")
        
        if i < len(all_centers):
            part_center = all_centers[i]
            direction = part_center - global_center
            direction_norm = torch.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            else:
                direction = torch.randn(3, device=gaussian.device)
                direction = direction / torch.norm(direction)
            
            offset = direction * explosion_scale
        else:
            offset = torch.zeros(3, device=gaussian.device)
            
        if gaussian._xyz is not None:
            original_xyz = gaussian.get_xyz
            exploded_xyz = original_xyz + offset
            exploded_xyz_normalized = (exploded_xyz - gaussian.aabb[None, :3]) / gaussian.aabb[None, 3:]
            xyz_list.append(exploded_xyz_normalized)
            
        if gaussian._features_dc is not None:
            features_dc_list.append(gaussian._features_dc)
        if gaussian._features_rest is not None:
            features_rest_list.append(gaussian._features_rest)
        if gaussian._scaling is not None:
            scaling_list.append(gaussian._scaling)
        if gaussian._rotation is not None:
            rotation_list.append(gaussian._rotation)
        if gaussian._opacity is not None:
            opacity_list.append(gaussian._opacity)
    
    if xyz_list:
        merged_gaussian._xyz = torch.cat(xyz_list, dim=0)
    if features_dc_list:
        merged_gaussian._features_dc = torch.cat(features_dc_list, dim=0)
    if features_rest_list:
        merged_gaussian._features_rest = torch.cat(features_rest_list, dim=0)
    else:
        merged_gaussian._features_rest = None
    if scaling_list:
        merged_gaussian._scaling = torch.cat(scaling_list, dim=0)
    if rotation_list:
        merged_gaussian._rotation = torch.cat(rotation_list, dim=0)
    if opacity_list:
        merged_gaussian._opacity = torch.cat(opacity_list, dim=0)
    
    return merged_gaussian
