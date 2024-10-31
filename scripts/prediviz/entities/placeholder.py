#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from omegaconf import DictConfig


class Placeholder:
    def __init__(self, config: "DictConfig") -> None:
        self.config: DictConfig = config.placeholder
        self.center_position: Optional[Tuple[float, float]] = None

    @property
    def width(self) -> float:
        return self.config.width

    @property
    def height(self) -> float:
        return self.config.height

    def plot(
        self, ax: Optional[plt.Axes] = None, origin: Tuple[float, float] = (0, 0)
    ) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        object_rect = FancyBboxPatch(
            (origin[0], origin[1]),
            self.width,
            self.height,
            edgecolor="white",
            facecolor="black",
            linewidth=0,
            linestyle="-",
            boxstyle=f"Round, pad=0, rounding_size={self.config.rounding_size}",
            alpha=1.0,
        )

        ax.add_patch(object_rect)

        self.center_position = (
            origin[0] + self.width / 2,
            origin[1] + self.height / 2,
        )  # Update center position

        if created_fig:
            return fig, ax
        else:
            return ax
