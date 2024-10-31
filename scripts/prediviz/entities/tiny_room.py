#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from omegaconf import DictConfig

from .utils import wrap_text


class TinyRoom:
    def __init__(self, config: "DictConfig", room_id: str) -> None:
        self.config: DictConfig = config.room
        self.room_id: str = room_id
        self.center_position: Optional[Tuple[float, float]] = None

    @property
    def width(self):
        return self.config.width

    @property
    def height(self):
        return self.config.height

    def plot_text_label(self, ax: plt.Axes) -> None:
        if self.center_position is None:
            raise ValueError("Center position is not set for the room")

        self.text_position = (
            self.center_position[0],
            self.center_position[1] + self.config.text_margin,
        )
        # Assuming 100 is max chars for any room name
        wrapped_text = wrap_text(self.room_id, 100, split_on_period=True)
        ax.annotate(
            wrapped_text,
            xy=self.text_position,
            ha="center",
            va="center",
            fontsize=self.config.text_size,
            zorder=float("inf"),
        )

    def plot(
        self, ax: plt.Axes = None, origin: Tuple[float, float] = (0, 0)
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        self.room_rect = FancyBboxPatch(
            origin,
            self.width,
            self.height,
            edgecolor="white",
            facecolor="#3E4C60",
            linewidth=1,
            linestyle="-",
            alpha=1.0,
        )
        ax.add_patch(self.room_rect)

        self.center_position = (origin[0] + self.width / 2, origin[1] + self.height / 2)

        self.plot_text_label(ax)

        if created_fig:
            return fig, ax
        else:
            return ax
