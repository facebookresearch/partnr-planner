#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from .constants import color_palette
from .instance_color_map import InstanceColorMap
from .legends.diff_args_legend import DiffArgsLegend
from .legends.is_next_to_legend import IsNextToLegend
from .legends.same_args_legend import SameArgsLegend
from .utils import wrap_text


class PrediViz:
    def __init__(self, config: "DictConfig", scene: Any) -> None:
        self.config = config
        self.scene = scene

    def compute_extra_height(self) -> None:
        if self.scene.instruction:
            wrapped_text = wrap_text(
                self.scene.instruction, self.scene.config.max_chars_per_line
            )
            number_of_lines = wrapped_text.count("\n") + 1
            self.extra_instruction_height = (
                self.scene.config.per_instruction_line_height * number_of_lines
            )

    def plot_instruction(
        self,
        ax: plt.Axes,
        scene_width: float,
        mx_width: float,
        height_lower: float,
        height_upper: float,
    ) -> None:
        if self.scene.instruction:
            frac = 0.5 * (scene_width / mx_width)
            wrapped_text = wrap_text(
                self.scene.instruction, self.scene.config.max_chars_per_line
            )
            ax.text(
                frac,
                (height_upper - height_lower - self.extra_instruction_height)
                / (height_upper - height_lower),
                wrapped_text,
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=ax.transAxes,
                fontsize=self.scene.config.instruction_text_size,
                zorder=float("inf"),
            )

    def get_is_next_tos(
        self, propositions: List[Dict[str, Any]], toposort: List[List[int]]
    ) -> List[List[List[Any]]]:
        is_next_tos = []
        if toposort:
            for current_level in toposort:
                current_propositions = [propositions[idx] for idx in current_level]
                current_is_next_tos = []
                for prop in current_propositions:
                    if prop["function_name"] == "is_next_to":
                        current_is_next_tos += [
                            [
                                prop["args"]["entity_handles_a_names_and_types"],
                                prop["args"]["entity_handles_b_names_and_types"],
                                prop["args"]["number"],
                            ]
                        ]
                is_next_tos.append(current_is_next_tos)
        else:
            current_is_next_tos = []
            for prop in propositions:
                if prop["function_name"] == "is_next_to":
                    current_is_next_tos += [
                        [
                            prop["args"]["entity_handles_a_names_and_types"],
                            prop["args"]["entity_handles_b_names_and_types"],
                            prop["args"]["number"],
                        ]
                    ]
            is_next_tos.append(current_is_next_tos)
        return is_next_tos

    def parse_propositions_and_set_instance_colors(
        self, propositions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        InstanceColorMap.reset_map()
        prop_idx = 0
        for prop in propositions:
            color = list(color_palette.values())[prop_idx % len(color_palette)]
            if prop["function_name"] == "is_next_to":
                prop["color"] = "white"
            else:
                prop["color"] = color
                if prop["function_name"] in [
                    "is_on_top",
                    "is_inside",
                    "is_in_room",
                    "is_on_floor",
                ]:
                    object_names = prop["args"]["object_names"]
                    for object_name in object_names:
                        if not InstanceColorMap.has_color(object_name):
                            InstanceColorMap.set_color(object_name, color)
                prop_idx += 1

        for room in self.scene.rooms:
            for o in room.objects:
                color = list(color_palette.values())[prop_idx % len(color_palette)]
                if not InstanceColorMap.has_color(o.object_id):
                    InstanceColorMap.set_color(o.object_id, color)
                    prop_idx += 1

        return propositions

    def _insert_new_legend(
        self, new_legend: Any, fig_data: List[Any], level_idx: int
    ) -> None:
        self.legends.append(new_legend)
        if level_idx not in self.fig_data_idx_to_legend_idxs:
            self.fig_data_idx_to_legend_idxs[level_idx] = []
        self.fig_data_idx_to_legend_idxs[level_idx].append(len(self.legends) - 1)
        prop_to_height_range = fig_data[level_idx][4]
        current_height_lower, current_height_upper, offset = prop_to_height_range[
            level_idx
        ]
        self.legend_bounds.append((current_height_lower, current_height_upper, offset))

    def _create_legend_data(
        self,
        propositions: List[Dict[str, Any]],
        toposort: List[List[int]],
        same_args_data: List[Dict[str, Any]],
        diff_args_data: List[Dict[str, Any]],
        cropped_receptacle_icon_mapping: Dict[str, Any],
        fig_data: List[Any],
    ) -> None:
        all_is_next_tos = self.get_is_next_tos(propositions, toposort)
        self.legends: List[Union[IsNextToLegend, SameArgsLegend, DiffArgsLegend]] = []
        self.legend_bounds: List[Tuple[float, float, float]] = []
        self.fig_data_idx_to_legend_idxs: Dict[int, List[int]] = {}
        legend_classes = {
            "is_next_to": IsNextToLegend,
            "same_args": SameArgsLegend,
            "diff_args": DiffArgsLegend,
        }

        def create_and_insert_legend(legend_type, data, level_idx):
            legend_class = legend_classes[legend_type]
            new_legend = legend_class(
                self.scene.config.is_next_to,
                data,
                cropped_receptacle_icon_mapping,
                propositions=propositions,
            )
            self._insert_new_legend(new_legend, fig_data, level_idx)

        if all_is_next_tos or same_args_data or diff_args_data:
            for level_idx, is_next_tos in enumerate(all_is_next_tos):
                if is_next_tos:
                    create_and_insert_legend("is_next_to", is_next_tos, level_idx)

            for legend_type, args_data in [
                ("same_args", same_args_data),
                ("diff_args", diff_args_data),
            ]:
                for level_idx, level in enumerate(toposort):
                    current_args_data = [
                        arg["data"]
                        for arg in args_data
                        if set(arg["proposition_indices"]).intersection(set(level))
                    ]
                    if current_args_data:
                        create_and_insert_legend(
                            legend_type, current_args_data, level_idx
                        )

    def _plot_legends(
        self,
        ax: plt.Axes,
        height_lower: float,
        height_upper: float,
        legends: List[Any],
        legend_bounds: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        range_to_num = {}
        range_to_current_height = {}
        range_to_consumed_space = {}

        # Precompute column assignments
        for legend, bound in zip(legends, legend_bounds):
            if bound not in range_to_num:
                range_to_num[bound] = 0
                range_to_consumed_space[bound] = 0
            range_to_num[bound] += 1
            range_to_current_height[bound] = 0
            range_to_consumed_space[bound] += (
                legend.height
                + self.scene.config.is_next_to.bottom_pad
                + self.scene.config.is_next_to.top_pad
            )

        # Compute necessary columns for each bound
        bounds_to_columns = {}
        for bound in range_to_consumed_space:
            (current_height_lower, current_height_upper, offset) = bound
            available_space = current_height_upper - current_height_lower
            consumed_space = range_to_consumed_space[bound]

            # Calculate how many columns are needed
            if consumed_space > available_space:
                num_columns = int(consumed_space // available_space) + 1
            else:
                num_columns = 1
            bounds_to_columns[bound] = num_columns

        # Distribute legends among the columns
        column_legend_lists: Dict[Tuple[float, float, float], List[List[Any]]] = {
            bound: [[] for _ in range(bounds_to_columns[bound])]
            for bound in bounds_to_columns
        }
        column_heights = {
            bound: [0] * bounds_to_columns[bound] for bound in bounds_to_columns
        }

        for legend, bound in zip(legends, legend_bounds):
            num_columns = bounds_to_columns[bound]
            min_height_column = min(
                range(num_columns), key=lambda col: column_heights[bound][col]
            )
            column_legend_lists[bound][min_height_column].append(legend)
            column_heights[bound][min_height_column] += (
                legend.height
                + self.scene.config.is_next_to.bottom_pad
                + self.scene.config.is_next_to.top_pad
            )

        # Plot legends
        mx_num_columns = 0
        mx_width = self.scene.width
        max_column_upper = height_upper
        min_column_lower = height_lower

        for bound in range_to_num:
            num_columns = bounds_to_columns[bound]
            mx_num_columns = max(num_columns, mx_num_columns)
            column_width = self.scene.config.is_next_to.width
            (current_height_lower, current_height_upper, offset) = bound
            available_space = current_height_upper - current_height_lower

            for col in range(num_columns):
                current_height = column_heights[bound][col]
                num_spaces = len(column_legend_lists[bound][col]) + 1
                column_spacing = max(0, (available_space - current_height) / num_spaces)
                if current_height > available_space:
                    # Center align legends vertically
                    total_legend_height = column_heights[bound][col]
                    vertical_offset = int(available_space - total_legend_height) / 2
                else:
                    vertical_offset = 0

                current_height = int(vertical_offset)
                for legend in column_legend_lists[bound][col]:
                    legend_space = (
                        legend.height
                        + self.scene.config.is_next_to.bottom_pad
                        + self.scene.config.is_next_to.top_pad
                    )
                    legend_origin = (
                        -offset - current_height - legend_space - column_spacing
                    )
                    max_column_upper = max(
                        max_column_upper,
                        legend_origin
                        + legend.height
                        + legend.top_pad
                        + legend.bottom_pad,
                    )
                    min_column_lower = min(min_column_lower, legend_origin)
                    legend_left = self.scene.width + col * (
                        column_width + legend.horizontal_margin
                    )
                    legend.plot((legend_left, legend_origin), ax)
                    mx_width = max(
                        mx_width, legend_left + legend.width + legend.horizontal_margin
                    )
                    current_height += legend_space + column_spacing
        return mx_width, max_column_upper, min_column_lower

    def plot(
        self,
        propositions: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        receptacle_icon_mapping: Dict[str, Any],
        cropped_receptacle_icon_mapping: Dict[str, Any],
        show_instruction: bool = True,
    ) -> List[Tuple[plt.Figure, plt.Axes, float, float]]:
        toposort = []
        same_args_data = []
        diff_args_data = []
        for constraint in constraints:
            if constraint["type"] == "TemporalConstraint":
                toposort = constraint["toposort"]
            elif constraint["type"] == "SameArgConstraint":
                same_args_data.append(constraint["same_args_data"])
            elif constraint["type"] == "DifferentArgConstraint":
                diff_args_data.append(constraint["diff_args_data"])
        propositions = self.parse_propositions_and_set_instance_colors(propositions)
        fig_data = self.scene.plot(
            propositions,
            constraints,
            toposort,
        )

        self._create_legend_data(
            propositions,
            toposort,
            same_args_data,
            diff_args_data,
            cropped_receptacle_icon_mapping,
            fig_data,
        )
        result_fig_data = []
        for fig_data_idx, (
            fig,
            ax,
            height_lower,
            height_upper,
            _,
        ) in enumerate(fig_data):
            if self.legends and fig_data_idx in self.fig_data_idx_to_legend_idxs:
                current_legends = [
                    self.legends[idx]
                    for idx in self.fig_data_idx_to_legend_idxs[fig_data_idx]
                ]
                current_legend_bounds = [
                    self.legend_bounds[idx]
                    for idx in self.fig_data_idx_to_legend_idxs[fig_data_idx]
                ]
                mx_width, max_column_upper, min_column_lower = self._plot_legends(
                    ax,
                    height_lower,
                    height_upper,
                    current_legends,
                    current_legend_bounds,
                )

                final_width = mx_width
                final_upper = max_column_upper
                final_lower = min_column_lower
            else:
                final_width = self.scene.width
                final_upper = height_upper
                final_lower = height_lower
            if show_instruction:
                self.compute_extra_height()
                final_upper += self.extra_instruction_height
                self.plot_instruction(
                    ax, self.scene.width, final_width, final_lower, final_upper
                )
            ax.set_xlim(0, final_width)
            ax.set_ylim(final_lower, final_upper)
            ax.axis("off")
            result_fig_data.append((fig, ax, final_upper - final_lower, final_width))
        return result_fig_data
