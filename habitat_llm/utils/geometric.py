#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional

import numpy as np
import torch


# ACKNOWLEDGEMENT: Taken from home-robot repository
def unproject_masked_depth_to_xyz_coordinates(
    depth: torch.Tensor,
    pose: torch.Tensor,
    inv_intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the XYZ coordinates for a batch posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        inv_intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """

    batch_size, _, height, width = depth.shape
    if mask is None:
        mask = torch.full_like(depth, fill_value=False, dtype=torch.bool)
    flipped_mask = ~mask

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=depth.device),
        torch.arange(0, height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(batch_size, dim=0)
    xy = xy[flipped_mask.squeeze(1)]
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Associates poses and intrinsics with XYZ coordinates.
    inv_intrinsics = inv_intrinsics[:, None, None, :, :].expand(
        batch_size, height, width, 3, 3
    )[flipped_mask.squeeze(1)]
    pose = pose[:, None, None, :, :].expand(batch_size, height, width, 4, 4)[
        flipped_mask.squeeze(1)
    ]
    depth = depth[flipped_mask]

    # Applies intrinsics and extrinsics.
    xyz = xyz.to(inv_intrinsics).unsqueeze(1) @ inv_intrinsics.permute([0, 2, 1])
    xyz = xyz * depth[:, None, None]
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[
        ..., None, :3, 3
    ]
    xyz = xyz.squeeze(1)

    return xyz


def opengl_to_opencv(pose):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pose = pose @ transform
    return pose
