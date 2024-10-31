#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source


from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from omegaconf import DictConfig

from .constants import OBJECT_STATE_ICONS_PATH, ROOM_COLOR
from .instance_color_map import InstanceColorMap
from .utils import load_and_resize_icon, wrap_text


def get_object_color(object_id: str) -> str:
    if InstanceColorMap.has_color(object_id):
        color = InstanceColorMap.get_color(object_id)
    else:
        raise NotImplementedError
    return color


class Object:
    def __init__(self, config: "DictConfig", object_id: str) -> None:
        self.object_id: str = object_id
        self.config: DictConfig = config.object
        self.center_position: Optional[Tuple[float, float]] = None
        self.is_on_floor: bool = False

        # Object states
        self.states: dict = {}
        self.previous_states: dict = {}
        self.set_icons()

    @property
    def width(self) -> float:
        return self.config.width

    @property
    def height(self) -> float:
        return self.config.height

    def plot_on_floor_line(self, ax: plt.Axes) -> None:
        assert (
            self.center_position is not None
        ), f"Center position is empty for object: {self.object_id}"
        assert (
            self.text_position is not None
        ), f"Text position is empty for object: {self.object_id}"
        # Calculate the height after the text position based on number of lines in the text
        line_start = (
            self.center_position[0]
            - self.config.on_floor_line_length_ratio * self.config.width,
            self.text_position[1] - self.config.extra_space_between_objects / 2,
        )
        line_end = (
            self.center_position[0]
            + self.config.on_floor_line_length_ratio * self.config.width,
            self.text_position[1] - self.config.extra_space_between_objects / 2,
        )
        line = ConnectionPatch(
            xyA=line_start,
            xyB=line_end,
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            color="white",
            linewidth=self.config.on_floor_linewidth,
        )
        ax.add_artist(line)

    def set_icons(self) -> None:
        self.clean_icon: np.ndarray = load_and_resize_icon(
            f"{OBJECT_STATE_ICONS_PATH}/object_clean.png", self.config.height
        )
        self.clean_plus_on_icon: np.ndarray = load_and_resize_icon(
            f"{OBJECT_STATE_ICONS_PATH}/object_clean_plus_on.png", self.config.height
        )
        self.powered_on_icon: np.ndarray = load_and_resize_icon(
            f"{OBJECT_STATE_ICONS_PATH}/object_on.png", self.config.height
        )
        self.refresh_icon: np.ndarray = load_and_resize_icon(
            f"{OBJECT_STATE_ICONS_PATH}/refresh.png", self.config.height
        )

    def change_rectangle_color(self, color: str) -> None:
        self.object_rect.set_facecolor(color)
        InstanceColorMap.set_color(self.object_id, color)
        plt.gcf().canvas.draw()

    def plot_text_label(self, ax: plt.Axes) -> None:
        self.text_position = (
            self.center_position[0],
            self.center_position[1] + self.config.text_margin,
        )

        wrapped_text = wrap_text(self.object_id, self.config.max_chars_per_line)
        ax.annotate(
            wrapped_text,
            xy=self.text_position,
            ha="center",
            va="center",
            fontsize=self.config.text_size,
            zorder=float("inf"),
        )

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, ...]:
        # Remove the hash symbol if present
        hex_color = hex_color.lstrip("#")

        # Convert the hex color to RGB components
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return rgb

    def plot_state_attributes(self, ax: plt.Axes, origin: Tuple[float, float]) -> None:
        if self.previous_states != self.states:
            # show refresh icon inside the object rect
            ax.imshow(
                self.refresh_icon,
                extent=[
                    origin[0] + 0.2 * self.config.width,
                    origin[0] + 0.8 * self.config.width,
                    origin[1] + 0.2 * self.config.height,
                    origin[1] + 0.8 * self.config.height,
                ],
                zorder=float("inf"),
            )
            self.previous_states = self.states.copy()

        icon_extent = [
            origin[0] - 1 * self.config.width,
            origin[0] + 2 * self.config.width,
            origin[1] - 1 * self.config.height,
            origin[1] + 2 * self.config.height,
        ]
        state_icons = {
            ("is_clean", "is_powered_on"): self.clean_plus_on_icon,
            ("is_clean",): self.clean_icon,
            ("is_powered_on",): self.powered_on_icon,
        }

        for states, icon in state_icons.items():
            if all(self.states.get(state) for state in states):
                ax.imshow(icon, extent=icon_extent)
            break

    def plot(
        self, ax: Optional[plt.Axes] = None, origin: Tuple[float, float] = (0, 0)
    ) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        color = get_object_color(self.object_id)

        # Determine the properties of the object rectangle based on the state
        if "is_filled" in self.states and not self.states["is_filled"]:
            edgecolor, facecolor, linewidth = (
                color,
                ROOM_COLOR,
                self.config.empty_state_linewidth,
            )
        else:
            edgecolor, facecolor, linewidth = "white", color, 0

        # Create the object rectangle
        self.object_rect: FancyBboxPatch = FancyBboxPatch(
            (origin[0], origin[1]),
            self.config.width,
            self.config.height,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            linestyle="-",
            boxstyle=f"Round, pad=0, rounding_size={self.config.rounding_size}",
            alpha=1.0,
        )

        ax.add_patch(self.object_rect)

        self.center_position = (
            origin[0] + self.config.width / 2,
            origin[1] + self.config.height / 2,
        )

        self.plot_text_label(ax)

        if self.is_on_floor:
            # Draw a white line below the text label
            self.plot_on_floor_line(ax)

        color = get_object_color(self.object_id)
        icons = [
            self.clean_icon,
            self.clean_plus_on_icon,
            self.powered_on_icon,
        ]
        rgb_color = self.hex_to_rgb(color)
        for icon in icons:
            icon[icon[:, :, 3] != 0, :3] = rgb_color

        self.refresh_icon[self.refresh_icon[:, :, 3] != 0, :3] = self.hex_to_rgb(
            "#FFFFFF"
        )
        self.plot_state_attributes(ax, origin)

        if created_fig:
            return fig, ax
        else:
            return ax
