#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

# Taken from: https://github.com/concept-graphs/concept-graphs/blob/main/conceptgraph/utils/general_utils.py

from typing import Union

import numpy as np
import torch


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)


def to_scalar(d: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the d to a scalar
    """
    if isinstance(d, float):
        return d

    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()

    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()

    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")
