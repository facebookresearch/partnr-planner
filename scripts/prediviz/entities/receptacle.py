#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from .constants import receptacle_color_map, receptacle_properties
from .placeholder import Placeholder
from .utils import add_tint_to_rgb, resize_icon_height


def calculate_placeholder_heights(image: Image.Image) -> Tuple[float, float]:
    # This method uses the alpha values to calculate center and top height of the icon
    alpha = np.array(image)[:, :, 3]
    bottom = alpha.shape[0] + 1
    top = 0
    for idx, row in enumerate(alpha):
        middle_idx = row.shape[0] // 2
        row_sum = np.sum(row[middle_idx])
        if row_sum != 0:
            top = idx + 1
            break
    top_height = bottom - top
    center_height = top_height / 2
    return center_height, top_height


class Receptacle:
    def __init__(
        self, config: "DictConfig", receptacle_id: str, icon_path: str
    ) -> None:
        self.config: DictConfig = config.receptacle
        self.object_config: DictConfig = config.object
        self.receptacle_id: str = receptacle_id
        self.icon_path: str = icon_path

        self.center_placeholder_position: Optional[Tuple[float, float]] = None
        self.top_placeholder_position: Optional[Tuple[float, float]] = None

        self.plot_top_placeholder: bool = False
        self.plot_center_placeholder: bool = False

        self.next_top_item_position: Optional[Tuple[float, float]] = None

        self.plot_states: dict[str, bool] = {
            "is_clean": False,
            "is_filled": False,
            "is_powered_on": False,
        }
        # This initialization does not matter as we parse the states from the files
        # and set both to the same value initially
        self.previous_states: dict[str, bool] = {}
        self.states: dict[str, bool] = {}
        self.temp_mx_height: float = (
            0.0  # used to keep track of the object heights on top
        )

        self.init_size()

    @property
    def horizontal_margin(self) -> float:
        return self.config.horizontal_margin

    def init_size(self) -> None:
        icon = self.get_icon(add_tint=False)
        icon_width, icon_height = icon.size
        self.width = icon_width + 2 * self.horizontal_margin
        self.height = icon_height

    def get_icon(self, add_tint: bool = True) -> Image.Image:
        icon = Image.open(self.icon_path)
        icon = resize_icon_height(icon, self.config.target_height)
        if add_tint:
            color = receptacle_color_map["_".join(self.receptacle_id.split("_")[:-1])]
            tint_color = tuple(int(255 * i) for i in color)
            icon = add_tint_to_rgb(icon, tint_color=tint_color)
        return icon

    def set_placeholder_positions(
        self, icon: Image.Image, origin: Tuple[float, float]
    ) -> None:
        center_height, top_height = calculate_placeholder_heights(icon)
        self.center_placeholder_position = (
            origin[0] + self.width / 2,
            origin[1] + center_height,
        )
        self.center_placeholder_origin = (
            self.center_placeholder_position[0] - self.config.placeholder.width / 2,
            self.center_placeholder_position[1] - self.config.placeholder.height / 2,
        )
        self.top_placeholder_position = (
            origin[0] + self.width / 2,
            origin[1] + top_height + self.config.placeholder_margin,
        )
        self.top_placeholder_origin = (
            self.top_placeholder_position[0] - self.config.placeholder.width / 2,
            self.top_placeholder_position[1] - self.config.placeholder.height / 2,
        )

        # The bottom is needed to calculate the stacked objects in case
        # there are many initialized on the same receptacle
        self.next_top_item_position = (
            self.top_placeholder_origin[0],
            self.top_placeholder_origin[1]
            + abs(self.object_config.text_margin)
            + self.object_config.bottom_text_extra_margin,
        )

    def plot_state_attributes(self, ax: plt.Axes, origin: Tuple[float, float]) -> None:
        # NOTE: The current logic is to set whether to display a particular state or not outside of this class
        # This is because we control at the row level whether to display the state or not
        if not any(self.plot_states.values()):  # If no states are set to be plotted,
            return

        # Plot the attributes as a text
        plot_height = origin[1] - self.config.state_text_relative_height
        state_texts = {
            "is_clean": ("clean", "dirty"),
            "is_filled": ("filled", "empty"),
            "is_powered_on": ("on", "off"),
        }
        for state, (true_text, false_text) in state_texts.items():
            if self.plot_states[state] and state in self.states:
                text = true_text if self.states[state] else false_text
                color = "white" if self.states[state] else "black"
                ax.text(
                    self.center_placeholder_position[0],
                    plot_height,
                    text,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=color,
                )
                plot_height -= self.config.state_text_relative_height

    def plot_placeholders(self, ax: plt.Axes) -> None:
        assert hasattr(
            self, "next_top_item_position"
        ), f"next item position is not set for receptacle: {self.receptacle_id}"
        properties = receptacle_properties["_".join(self.receptacle_id.split("_")[:-1])]
        # TODO: See how to handle `is_same`
        if self.plot_top_placeholder and properties["is_same"]:
            self.plot_center_placeholder = True
            self.plot_top_placeholder = False
        if self.plot_top_placeholder and properties["is_on_top"]:
            self.top_placeholder = Placeholder(self.config)
            ax = self.top_placeholder.plot(ax, self.top_placeholder_origin)
            self.next_top_item_position = (
                self.next_top_item_position[0],
                self.next_top_item_position[1] + self.config.placeholder.height,
            )
        if self.plot_center_placeholder and properties["is_inside"]:
            self.center_placeholder = Placeholder(self.config)
            ax = self.center_placeholder.plot(ax, self.center_placeholder_origin)

    def plot(
        self, ax: Optional[plt.Axes] = None, origin: Tuple[float, float] = (0, 0)
    ) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        icon = self.get_icon()
        receptacle_width, receptacle_height = icon.size
        ax.imshow(
            icon,
            extent=(
                (origin[0] + self.horizontal_margin),
                (origin[0] + receptacle_width + self.horizontal_margin),
                origin[1],
                (origin[1] + receptacle_height),
            ),
        )
        self.set_placeholder_positions(icon, origin)
        self.plot_placeholders(ax)
        self.plot_state_attributes(ax, origin)

        if created_fig:
            ax.axis("off")
            return fig, ax
        else:
            return ax
