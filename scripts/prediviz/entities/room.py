#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from .constants import ROOM_COLOR
from .object import Object
from .placeholder import Placeholder
from .receptacle import Receptacle
from .utils import wrap_text


class Room:
    def __init__(
        self,
        config: "DictConfig",
        room_id: str,
        receptacles: List[Receptacle],
        objects: Optional[List[Object]] = None,
        use_full_height: bool = False,
        in_proposition: bool = False,
        object_to_recep: Optional[dict] = None,
    ) -> None:
        self.global_config: DictConfig = config
        self.config: DictConfig = config.room
        self.room_id: str = room_id

        self.receptacles: List[Receptacle] = receptacles
        self.objects: Optional[List[Object]] = objects

        self.in_proposition: bool = in_proposition

        self.plot_placeholder: bool = False

        # Initial object to receptacle mapping
        self.object_to_recep: Optional[dict] = object_to_recep

        self.num_receptacle_state_lines: int = 0

        self.use_full_height: bool = use_full_height

        if self.objects:
            self.use_full_height = True

        self.init_size()

    def init_widths(self) -> None:
        # Calculate width based on the objects and receptacles
        min_width = self.config.min_width
        if self.objects:
            object_widths = 0.0
            for o in self.objects:
                if self.object_to_recep is None:
                    object_widths += o.width
                else:
                    if o.object_id in self.object_to_recep:
                        continue
                    else:
                        object_widths += o.width

            min_width = max(min_width, object_widths * self.config.min_width_per_object)

        # Calculate total room width including margins
        minimum_room_width = max(
            min_width, sum(receptacle.width for receptacle in self.receptacles)
        )
        self.room_width = (
            minimum_room_width + self.config.left_pad + self.config.right_pad
        )
        self.width = self.room_width + 2 * self.config.horizontal_margin

    def init_heights(self) -> None:
        # Init with min receptacle height
        # Need to increase the bottom pad by the number of lines we are planning to plot for receptacle states
        extra_pad_for_receptacle_states = (
            self.config.per_receptacle_state_padding * self.num_receptacle_state_lines
        )

        self.room_height = self.config.min_height
        for receptacle in self.receptacles:
            receptacle.temp_mx_height = receptacle.height

        if self.objects:
            self.room_height *= 2
            for obj in self.objects:
                if (
                    self.object_to_recep is not None
                    and obj.object_id in self.object_to_recep
                ):
                    receptacle_id = self.object_to_recep[obj.object_id]
                    if receptacle_id == "floor":
                        continue

                    current_receptacle = self.find_receptacle_by_id(receptacle_id)
                    if current_receptacle is None:
                        raise ValueError(
                            f"Receptacle {receptacle_id} not found for object {obj.object_id}"
                        )

                    current_receptacle.temp_mx_height += (
                        abs(obj.config.text_margin)
                        + 2 * obj.config.height
                        + obj.config.extra_space_between_objects
                    )
                    # We take max of all top item positions for now
                    self.room_height = max(
                        self.room_height, current_receptacle.temp_mx_height
                    )

        self.room_height = (
            self.room_height
            + self.config.bottom_pad
            + extra_pad_for_receptacle_states
            + self.config.top_pad
        )
        self.height = self.room_height + 2 * self.config.vertical_margin

    def init_size(self) -> None:
        self.init_widths()
        self.init_heights()

    def cleanup(self) -> None:
        if self.objects:
            for obj in self.objects:
                del obj
            self.objects.clear()
        if self.receptacles:
            for recep in self.receptacles:
                del recep
            self.receptacles.clear()

    def find_object_by_id(self, object_id: str) -> Optional[Object]:
        if self.objects:
            for obj in self.objects:
                if obj.object_id == object_id:
                    return obj
        return None

    def find_receptacle_by_id(self, receptacle_id: str) -> Optional[Receptacle]:
        for receptacle in self.receptacles:
            if receptacle.receptacle_id == receptacle_id:
                return receptacle
        return None

    def plot_objects(self, ax: plt.Axes, actual_origin: Tuple[float, float]) -> None:
        if self.objects:
            # Handle non mapped objects
            # Calculate initial offset for objects considering left margin, horizontal padding, and spacing objects evenly
            total_object_width = 0.0
            num_objects = 0
            for obj in self.objects:
                if (
                    self.object_to_recep is None
                    or obj.object_id not in self.object_to_recep.keys()
                    or self.object_to_recep[obj.object_id] == "floor"
                ):
                    total_object_width += obj.width
                    num_objects += 1

            spacing = (
                (
                    self.room_width
                    - self.config.object_horizontal_margin_fraction
                    * 2
                    * self.room_width
                )
                - total_object_width
            ) / (num_objects + 1)
            offset = (
                actual_origin[0]
                + self.config.object_horizontal_margin_fraction * self.room_width
                + spacing
            )

            for obj in self.objects:
                if (
                    self.object_to_recep is None
                    or obj.object_id not in self.object_to_recep.keys()
                    or self.object_to_recep[obj.object_id] == "floor"
                ):
                    ax = obj.plot(
                        ax,
                        origin=(
                            offset,
                            actual_origin[1]
                            + self.room_height * self.config.objects_height,
                        ),
                    )
                    offset += obj.width + spacing
                elif (
                    self.object_to_recep is not None
                    and obj.object_id in self.object_to_recep
                    and self.object_to_recep[obj.object_id] != "floor"
                ):
                    receptacle_id = self.object_to_recep[obj.object_id]
                    # print(obj.object_id, self.room_id, receptacle_id, self.object_to_recep)
                    current_receptacle = self.find_receptacle_by_id(receptacle_id)
                    obj_position = current_receptacle.next_top_item_position
                    ax = obj.plot(ax, obj_position)
                    current_receptacle.next_top_item_position = (
                        obj_position[0],
                        current_receptacle.next_top_item_position[1]
                        + abs(obj.config.text_margin)
                        + 2 * obj.config.height
                        + obj.config.extra_space_between_objects,
                    )

    def plot_receptacles(
        self, ax: plt.Axes, actual_origin: Tuple[float, float]
    ) -> None:
        extra_pad_for_receptacle_states = (
            self.config.per_receptacle_state_padding * self.num_receptacle_state_lines
        )
        # Calculate initial offset considering left margin and horizontal padding
        receptacle_width = sum(recep.width for recep in self.receptacles)
        num_receptacles = len(self.receptacles)
        spacing = (
            (
                self.room_width
                - self.config.receptacle_horizontal_margin_fraction
                * 2
                * self.room_width
            )
            - receptacle_width
        ) / (num_receptacles + 1)
        offset = (
            actual_origin[0]
            + spacing
            + self.config.receptacle_horizontal_margin_fraction * self.room_width
        )
        for receptacle in self.receptacles:
            ax = receptacle.plot(
                ax,
                origin=(
                    offset,
                    actual_origin[1]
                    + self.config.bottom_pad
                    + extra_pad_for_receptacle_states,
                ),
            )
            offset += receptacle.width + spacing

    def plot_text_label(self, ax: plt.Axes, actual_origin: Tuple[float, float]) -> None:
        extra_pad_for_receptacle_states = (
            self.config.per_receptacle_state_padding * self.num_receptacle_state_lines
        )
        # Calculate text annotation position
        text_x = actual_origin[0] + self.room_width / 2
        text_y = (
            actual_origin[1]
            + (self.config.bottom_pad + extra_pad_for_receptacle_states) / 4
        )  # Offset for lower v_pad region

        wrapped_text = wrap_text(self.room_id, self.config.max_chars_per_line)

        text_y = actual_origin[1] + (
            self.config.bottom_pad + extra_pad_for_receptacle_states
        ) / 4 * 1 / (wrapped_text.count("\n") + 1)
        ax.annotate(
            wrapped_text,
            xy=(text_x, text_y),
            xytext=(text_x, text_y),
            ha="center",
            va="bottom",
            fontsize=self.config.text_size,
            zorder=float("inf"),
        )

    def plot_placeholders(
        self, ax: plt.Axes, actual_origin: Tuple[float, float]
    ) -> None:
        self.center_position = (
            actual_origin[0] + self.width / 2,
            actual_origin[1] + (self.config.placeholder_height * self.room_height),
        )
        if self.plot_placeholder:
            self.center_placeholder = Placeholder(self.config)
            center_placeholder_origin = (
                self.center_position[0] - self.config.placeholder.width / 2,
                self.center_position[1] - self.config.placeholder.height / 2,
            )
            ax = self.center_placeholder.plot(ax, center_placeholder_origin)

    def plot(
        self,
        origin: Tuple[float, float] = (0, 0),
        ax: Optional[plt.Axes] = None,
        target_width: Optional[float] = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        actual_origin = (
            origin[0] + self.config.horizontal_margin,
            origin[1] + self.config.vertical_margin,
        )

        # Re-initialize widths
        self.init_widths()

        # Add extra horizontal padding if needed to match target width
        if target_width is None:
            extra_horizontal_pad = 0
        else:
            extra_horizontal_pad = max(
                0,
                (target_width - self.room_width - 2 * self.config.horizontal_margin)
                / 2,
            )
        # Recalculate room widths and total width
        self.room_width = self.room_width + 2 * extra_horizontal_pad
        self.width = self.room_width + 2 * self.config.horizontal_margin

        self.plot_receptacles(ax, actual_origin)
        self.plot_text_label(ax, actual_origin)
        self.plot_objects(ax, actual_origin)

        # Plot the rectangle for the room
        border_width = (
            self.config.border_width
            if self.in_proposition
            and not self.config.disable_in_proposition_room_border
            else 0
        )
        rect = ax.add_patch(
            plt.Rectangle(
                (
                    actual_origin[0] + border_width,
                    actual_origin[1] + border_width,
                ),
                self.room_width - 2 * border_width,
                self.room_height - 2 * border_width,
                edgecolor="white" if border_width else None,
                linewidth=border_width,
                facecolor=ROOM_COLOR,
                alpha=self.config.box_alpha,
            )
        )
        # Set the z-order of the rectangle
        rect.set_zorder(-1)

        self.plot_placeholders(ax, actual_origin)

        if created_fig:
            ax.set_xlim(origin[0], origin[0] + self.width)
            ax.set_ylim(origin[1], origin[1] + self.height)
            ax.axis("off")
            return fig, ax
        else:
            return ax
