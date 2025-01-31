#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional

import magnum as mn
import numpy as np
import numpy.typing as npt
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

    return xyz[:, :3]


def unproject_coordinates(
    im_coordinates: npt.NDArray[np.float64],
    depth: npt.NDArray[np.float64],
    camera_matrix: mn.Matrix4,
    projection_matrix: mn.Matrix4,
    im_size: npt.NDArray[np.float64],
):
    """
    Unproject from image coordinates to 3D. Input are the coordinates in image space ([0, W], [0, H])
    and metric depth. Output are the xyz coordinates in world_space.
    :param im_coordinates: [B,2] coordinates we want to project in image space.
    :param depth: [B,1] metric depth.
    :param camera_matrix: 4x4 pose matrix
    :param projection_matrix: 4x4 projection matrix in OpenGL format.
    :param im_size: [1, 2] array containing image W, H
    """
    ndc = im_coordinates * 2.0 / im_size - 1.0
    ndc[:, 1] *= -1
    # Convert Z to normalized device coordinates
    b = depth.shape[0]
    zvec = np.zeros((b, 4))
    zvec[:, -1] = -1
    zvec[:, 2] = depth[:, 0]
    zclip = (np.array(projection_matrix) @ zvec.transpose()).transpose()
    zndc = (zclip[:, 2] / zclip[:, -1])[..., None]
    ndc = np.concatenate([ndc, zndc, np.ones((b, 1))], 1)

    # Unproject
    xyz = (
        np.array(camera_matrix.inverted() @ projection_matrix.inverted())
        @ ndc.transpose()
    ).transpose()

    xyz = (xyz / xyz[:, [-1]])[:, :3]
    return xyz


def project_to_im_coordinates(
    xyz: npt.NDArray[np.float64],
    camera_matrix: mn.Matrix4,
    projection_matrix: mn.Matrix4,
    im_size: npt.NDArray[np.float64],
):
    """
    Projects a batch of world coordinates to image coordinates. Input is XYZ with Y corresponding to height.
    output is an array [X,Y] where X is in range [0,W] and Y in range [0, H]
    :param xyz: array of size Bx3 with the world coordinates
    :param camera_matrix: 4x4 extrinsics matrix
    :param projection_matrx: 4x4 intrinsics matrix using OpenGL
    :projection_size: tuple with the camera projection_size. Typically 2, 2
    :param im_size: the target image size (width, height)
    """
    point_homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], 1)
    point_cam_coord = (np.array(camera_matrix) @ point_homog.transpose()).transpose()
    point_cam_coord = (
        np.array(projection_matrix) @ point_cam_coord.transpose()
    ).transpose()
    im_coord = point_cam_coord / point_cam_coord[:, [-2]]
    im_coord = im_coord[:, :2] / 2.0 + 0.5
    im_coord[:, 1] = 1 - im_coord[:, 1]
    im_coord = im_coord * im_size
    return im_coord


def opengl_to_opencv(pose):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pose = pose @ transform
    return pose
