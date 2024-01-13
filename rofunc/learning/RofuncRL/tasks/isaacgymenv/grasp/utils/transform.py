import torch
from torch import Tensor
from typing import *


def fuse_rgbd_to_pointcloud(rgbd: Tensor, k: Tensor, v: Tensor,
                            convert_rgb_to_camera_id: bool = False
                            ) -> Tuple[Tensor, Tensor]:
    """Fuses multiple depth views into a point could.

    Args:
        rgbd (tensor [B,V,H,W,4]): depth images from V different views
        k: (tensor [4, 4]): camera projection matrix
        v:  (tensor [V, 4, 4]): view matrix of V different cameras
        convert_rgb_to_camera_id (bool): Use colors to identify cameras
    Returns:
    """
    xyz, features = [], []
    num_cameras = rgbd.shape[1]
    for c in range(num_cameras):
        xyz_view, features_view = rgbd_to_pointcloud(rgbd[:, c], k, v[c], c,
                                                     convert_rgb_to_camera_id)
        xyz.append(xyz_view)
        features.append(features_view)
    return torch.stack(xyz, dim=1), torch.stack(features, dim=1)


def rgbd_to_pointcloud(rgbd: Tensor, k: Tensor, v: Tensor,
                       camera_id: int,
                       convert_rgb_to_camera_id: bool
                       ) -> Tuple[Tensor, Tensor]:
    color = rgbd[..., 0:3]
    depth = rgbd[..., -1]
    batch_size, height, width = depth.shape

    sparse_depth = depth.to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()

    if convert_rgb_to_camera_id:
        color_map = torch.Tensor(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]]).to(depth.device)
        features = color_map[camera_id].unsqueeze(0).repeat(values.shape[0], 1)
    else:
        features = color.reshape(-1, 3)

    xy_depth = torch.cat([indices.T[:, 1:].flip(1), values[..., None]], dim=-1)

    fu = 2 / k[0, 0]
    fv = 2 / k[1, 1]
    center_u = width / 2
    center_v = height / 2

    k_new = torch.Tensor([[fu, 0, 0],
                          [0, fv, 0],
                          [0, 0, 1]]).to(depth.device)

    xy_depth[:, 0] = -(xy_depth[:, 0] - center_u) / width
    xy_depth[:, 1] = (xy_depth[:, 1] - center_v) / height
    xy_depth[:, 0] *= xy_depth[:, 2]
    xy_depth[:, 1] *= xy_depth[:, 2]

    x2 = xy_depth @ k_new
    x2_hom = torch.cat([x2, torch.ones_like(x2[:, 0:1])], dim=1)

    xyz = (x2_hom @ v.inverse())[:, 0:3]

    xyz = xyz.view(batch_size, -1, 3)
    features = features.view(batch_size, -1, 3)
    return xyz, features


def subsample_valid(xyz, features, start=(-1, -1, 0.75), end=(1, 1, 1.75)):
    valid = (xyz[..., 0] >= start[0]) & (xyz[..., 0] <= end[0]) & \
            (xyz[..., 1] >= start[1]) & (xyz[..., 1] <= end[1]) & \
            (xyz[..., 2] >= start[2]) & (xyz[..., 2] <= end[2])
    xyz_valid = xyz[valid]
    features_valid = features[valid]
    return xyz_valid, features_valid


def pointcloud_to_voxelgrid(xyz: Tensor, features: Tensor,
                            voxel_size: float = 0.01,
                            start: Tuple[float, float, float] = (-0.5, -0.5, 0.5),
                            end: Tuple[float, float, float] = (0.5, 0.5, 1.5)
                            ) -> Tuple[Tensor, Tensor]:
    num_envs = xyz.shape[0]

    # I should be able to avoid iterating through envs if I adjust the subsample
    # function to pad the smaller pointclouds with dummy values

    vg, vf = [], []
    for env_id in range(num_envs):
        xyz_valid, features_valid = subsample_valid(
            xyz[env_id].view(-1, 3), features[env_id].view(-1, 3), start, end)

        xyz_valid[..., 0] -= start[0]
        xyz_valid[..., 1] -= start[1]
        xyz_valid[..., 2] -= start[2]

        # quantize
        xyz_indices = torch.round(xyz_valid / voxel_size).long()

        voxel_grid = torch.zeros([int(((end[0] - start[0]) / voxel_size) + 1),
                                  int(((end[1] - start[1]) / voxel_size) + 1),
                                  int(((end[2] - start[2]) / voxel_size) + 1)],
                                 device=xyz.device)

        voxel_features = torch.zeros(
            [int(((end[0] - start[0]) / voxel_size) + 1),
             int(((end[1] - start[1]) / voxel_size) + 1),
             int(((end[2] - start[2]) / voxel_size) + 1), 3],
            device=xyz.device)

        voxel_grid[xyz_indices[:, 0], xyz_indices[:, 1], xyz_indices[:, 2]] = 1

        voxel_features[xyz_indices[:, 0], xyz_indices[:, 1], xyz_indices[:, 2]] = features_valid

        vg.append(voxel_grid)
        vf.append(voxel_features)

    return torch.stack(vg), torch.stack(vf)
