import torch
import torch.nn as nn

from . import functional as F

__all__ = ['Voxelization']


def my_voxelization(features, coords, resolution):
    b, c, _ = features.shape
    result = torch.zeros(b, c + 1, resolution * resolution * resolution, device=features.device, dtype=features.dtype)
    r = resolution
    r2 = resolution * resolution
    coords = coords.long()
    indices = coords[:, 0] * r2 + coords[:, 1] * r + coords[:, 2]

    # print(r, r2, coords[:, 0].max(), coords[:, 1].max(), coords[:, 2].max())

    # print(f"Resolution: {resolution}")
    # print(f"Coords shape: {coords.shape}")
    # print(f"Coords max per dim: x={coords[:, 0].max()}, y={coords[:, 1].max()}, z={coords[:, 2].max()}")
    # print(f"Coords min per dim: x={coords[:, 0].min()}, y={coords[:, 1].min()}, z={coords[:, 2].min()}")
    # print(f"Indices shape: {indices.shape}")
    # print(f"Indices max: {indices.max()}, min: {indices.min()}")
    # print(f"Expected max index: {resolution * resolution * resolution - 1}")

    # # 检查是否有越界的索引
    # max_valid_index = resolution * resolution * resolution - 1
    # invalid_mask = (indices > max_valid_index) | (indices < 0)
    # if invalid_mask.any():
    #     print(f"Found {invalid_mask.sum()} invalid indices!")
    #     print(f"Invalid indices: {indices[invalid_mask]}")
    #     # 找到对应的坐标
    #     invalid_coords = coords[:, :, invalid_mask.any(dim=0)]
    #     print(f"Invalid coords shape: {invalid_coords.shape}")
    #     if invalid_coords.numel() > 0:
    #         print(f"Sample invalid coords: {invalid_coords[:, :, :5]}")  # 显示前5个无效坐标

    indices = indices.unsqueeze(dim=1).expand(-1, result.shape[1], -1)
    features = torch.cat([features, torch.ones(features.shape[0], 1, features.shape[2], device=features.device, dtype=features.dtype)], dim=1)
    out_feature = result.scatter_(index=indices.long(), src=features, dim=2, reduce='add')
    cnt = out_feature[:, -1:, :]
    zero_mask = (cnt == 0).to(features.dtype)
    cnt = cnt * (1 - zero_mask) + zero_mask * 1e-5
    vox_feature = out_feature[:, :-1, :] / cnt
    return vox_feature.view(b, c, resolution, resolution, resolution)

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0, scale_pvcnn=False):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.scale_pvcnn = scale_pvcnn
        assert not normalize

    def forward(self, features, coords):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            coords = coords.detach()

            if self.normalize:
                norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
            else:
                if self.scale_pvcnn:
                    norm_coords = (coords + 1) / 2.0 # [0, 1]
                    # print(norm_coords.shape, norm_coords.max(), norm_coords.min())
                else:
                    # norm_coords = (norm_coords + 1) / 2.0
                    norm_coords = (coords + 1) / 2.0
            norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
            # print(norm_coords.shape, norm_coords.max(), norm_coords.min())
            vox_coords = torch.round(norm_coords)
            # print(vox_coords.shape, vox_coords.max(), vox_coords.min())
            # print(features.shape)
        new_vox_feat = my_voxelization(features, vox_coords, self.r)
        return new_vox_feat, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
