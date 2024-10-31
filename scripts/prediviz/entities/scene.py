#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, PathPatch
from matplotlib.path import Path
from omegaconf import DictConfig

from .constants import BACKGROUND_COLOR
from .room import Room
from .temporal_utils import update_object_recep_and_room


def sort_rooms(rooms: List[Room], instruction: Optional[str]) -> List[Room]:
    # Sorts the rooms based on "relevance"
    # Uses keyword matching with the instruction and room + receptacle + object names
    if not instruction:
        return rooms

    # Split instruction string into words and exclude "room"
    keywords = [word.lower().strip(".") for word in instruction.split()]

    relevance_scores = {}

    for room in rooms:
        score = sum(
            " ".join(room.room_id.split("_")[:-1]) in keyword for keyword in keywords
        )

        for receptacle in room.receptacles:
            score += sum(
                " ".join(receptacle.receptacle_id.split("_")[:-1]) in keyword
                for keyword in keywords
            )

        if room.objects:
            for obj in room.objects:
                score += sum(
                    " ".join(obj.object_id.split("_")[:-1]) in keyword
                    for keyword in keywords
                )

        relevance_scores[room] = score

    # Sort the rooms based on relevance score
    sorted_rooms = sorted(
        relevance_scores.keys(),
        key=lambda room: relevance_scores[room],
        reverse=True,
    )

    return sorted_rooms


def redistribute_target_width_to_rooms(
    rooms_to_plot: List[Room], target_width: float
) -> List[float]:
    total_width = sum(room.width for room in rooms_to_plot)
    redistribution_factor = target_width / total_width
    redistributed_widths = [
        room.width * redistribution_factor for room in rooms_to_plot
    ]

    return redistributed_widths


class Scene:
    def __init__(
        self,
        config: "DictConfig",
        rooms: List[Room],
        instruction: str = "",
        object_to_recep: Optional[Dict[str, str]] = None,
        object_to_room: Optional[Dict[str, str]] = None,
    ):
        self.config = config.scene
        self.instruction = instruction
        self.rooms = sort_rooms(rooms, instruction)
        self.room_id_to_room = {room.room_id: room for room in self.rooms}
        self.object_to_recep = object_to_recep
        self.object_to_room = object_to_room

    def cleanup(self):
        if self.rooms:
            for room in self.rooms:
                room.cleanup()
                del room

    def plot_object_to_receptacle_lines(
        self,
        object_names: List[str],
        receptacle_names: List[str],
        number: int,
        function_name: str,
        ax: plt.Axes,
        color: Optional[Tuple[float, float, float, float]] = None,
        height_upper: Optional[float] = None,
    ) -> None:
        for object_name in object_names:
            for room in self.rooms:
                object_obj = room.find_object_by_id(object_name)
                if object_obj:
                    for receptacle_name in receptacle_names:
                        receptacle_objs = []
                        for r_room in self.rooms:
                            receptacle_obj = r_room.find_receptacle_by_id(
                                receptacle_name
                            )
                            if receptacle_obj:
                                receptacle_objs.append(receptacle_obj)

                        for receptacle_obj in receptacle_objs:
                            if function_name == "is_inside":
                                arrow_dest_position = (
                                    receptacle_obj.center_placeholder_position
                                )
                            else:
                                arrow_dest_position = (
                                    receptacle_obj.top_placeholder_position
                                )

                            if len(object_names) > number:
                                line_style = "dotted"
                                label = str(number) if number > 1 else None
                            else:
                                line_style = "-"
                                label = None
                            self.add_arrow(
                                ax,
                                object_obj.center_position,
                                arrow_dest_position,
                                line_style,
                                height_upper=height_upper,
                                color=color,
                                label=label,
                            )

    def plot_object_to_room_lines(
        self,
        object_names: List[str],
        room_names: List[str],
        number: int,
        ax: plt.Axes,
        color: Optional[Tuple[float, float, float, float]] = None,
        height_upper: Optional[float] = None,
    ) -> None:
        source_objects = []
        target_rooms = []
        for object_name in object_names:
            for room in self.rooms:
                object_obj = room.find_object_by_id(object_name)
                if object_obj:
                    source_objects.append(object_obj)
        for room_name in room_names:
            for r_room in self.rooms:
                if r_room.room_id == room_name:
                    target_rooms.append(r_room)
        for object_obj in source_objects:
            for room_obj in target_rooms:
                if len(object_names) > number:
                    line_style = "dotted"
                    label = str(number) if number > 1 else None
                else:
                    line_style = "-"
                    label = None

                self.add_arrow(
                    ax,
                    object_obj.center_position,
                    room_obj.center_position,
                    line_style,
                    height_upper=height_upper,
                    color=color,
                    label=label,
                )

    def add_arrow(
        self,
        ax: plt.Axes,
        obj_loc: Tuple[float, float],
        room_loc: Tuple[float, float],
        line_style: str,
        height_upper: Optional[float] = None,
        color: Tuple[float, float, float, float] = (1, 1, 1, 1),
        label: Optional[str] = None,
    ) -> None:
        x0, y0 = obj_loc
        x1, y1 = room_loc
        dx, dy = x1 - x0, y1 - y0

        def get_curve_midpoint(x0, y0, ctrl_x, ctrl_y, x1, y1):
            # Given control points
            P0 = np.array([x0, y0])
            P1 = np.array([ctrl_x, ctrl_y])
            P2 = np.array([x1, y1])

            # Calculate midpoint using the formula for quadratic Bezier curve
            t = 0.5
            midpoint = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

            return midpoint

        # Calculate control points for the Bézier curve
        if abs(dy) > abs(dx):
            ctrl_x = (x0 + x1) / 2 + (-1 if np.sign(dx) == 0 else np.sign(dx)) * abs(
                dy
            ) / 2
            ctrl_y = min((y0 + y1) / 2, height_upper)
        else:
            ctrl_x = (x0 + x1) / 2  # + dy / 2
            ctrl_y = min((y0 + y1) / 2 + abs(dx) / 2, height_upper)

        # Define path for the curved arrow
        path_data = [
            (Path.MOVETO, (x0, y0)),
            (Path.CURVE3, (ctrl_x, ctrl_y)),
            (Path.CURVE3, (x1, y1)),
        ]
        codes, verts = zip(*path_data)
        path = Path(verts, codes)
        patch = PathPatch(
            path,
            linestyle=line_style,
            linewidth=self.config.arrow.linewidth,
            facecolor="none",
            edgecolor=color,
        )
        ax.add_patch(patch)

        # Calculate the derivative (tangent) at the end point of the Bézier curve
        t = 1  # At the end point
        dx_dt = 2 * (1 - t) * (ctrl_x - x0) + 2 * t * (x1 - ctrl_x)
        dy_dt = 2 * (1 - t) * (ctrl_y - y0) + 2 * t * (y1 - ctrl_y)
        arrow_dir = np.array([dx_dt, dy_dt])
        arrow_dir /= np.linalg.norm(arrow_dir)

        # Calculate the position for the arrowhead
        head_pos = np.array([x1, y1]) - arrow_dir * self.config.arrow.head_length

        # Add arrowhead
        arrow_head = FancyArrow(
            head_pos[0],
            head_pos[1],
            arrow_dir[0] * self.config.arrow.head_length,
            arrow_dir[1] * self.config.arrow.head_length,
            head_length=self.config.arrow.head_length,
            head_width=self.config.arrow.head_width,
            linewidth=self.config.arrow.linewidth,
            edgecolor=color,
            facecolor=color,
            length_includes_head=True,
            overhang=self.config.arrow.overhang,
        )
        ax.add_patch(arrow_head)

        if label:
            mid_vert = get_curve_midpoint(x0, y0, ctrl_x, ctrl_y, x1, y1)
            ax.text(
                mid_vert[0],
                mid_vert[1],
                label,
                ha="center",
                va="center",
                fontsize=self.config.arrow.label_fontsize,
                bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, linewidth=0),
            )

    def change_heights_for_receptacle_states(
        self, current_rooms_to_plot: List[Room]
    ) -> None:
        states_to_plot_for_this_row = {
            "is_clean": False,
            "is_filled": False,
            "is_powered_on": False,
        }
        for room in current_rooms_to_plot:
            for receptacle in room.receptacles:
                for state in ["is_clean", "is_filled", "is_powered_on"]:
                    if (
                        state in receptacle.states
                        and receptacle.states[state]
                        != receptacle.previous_states[state]
                    ):
                        states_to_plot_for_this_row[state] = True

        num_receptacle_state_lines = sum(states_to_plot_for_this_row.values())

        for room in current_rooms_to_plot:
            room.num_receptacle_state_lines = num_receptacle_state_lines
            room.init_heights()  # reinitialize the heights
            for receptacle in room.receptacles:
                receptacle.previous_states = receptacle.states.copy()
                for state in ["is_clean", "is_filled", "is_powered_on"]:
                    receptacle.plot_states[state] = states_to_plot_for_this_row[state]

    def plot_a_row_of_rooms(
        self,
        ax: plt.Axes,
        current_rooms_to_plot: List[Room],
        target_width: float,
        current_row_height: float,
    ) -> Tuple[float, float]:
        # Check if there are any changes in any receptacle states in this row
        self.change_heights_for_receptacle_states(current_rooms_to_plot)
        current_row_height -= max(room.height for room in current_rooms_to_plot)
        # Set max room height for all rooms
        max_room_height = max(room.room_height for room in current_rooms_to_plot)
        max_height = max(room.height for room in current_rooms_to_plot)
        for room in current_rooms_to_plot:
            room.height = max_height
            room.room_height = max_room_height
        current_row_width = 0
        room_target_widths = redistribute_target_width_to_rooms(
            current_rooms_to_plot, target_width
        )
        for room, room_target_width in zip(current_rooms_to_plot, room_target_widths):
            ax = room.plot(
                origin=(current_row_width, current_row_height),
                ax=ax,
                target_width=room_target_width,
            )
            current_row_width += room.width
        if not hasattr(self, "width"):
            self.width: float = current_row_width
        else:
            self.width = max(current_row_width, self.width)
        return current_row_width, current_row_height

    def plot_room_rows(
        self,
        mentioned_rooms: List[str],
        ax: plt.Axes,
        target_width: float,
        height_offset: float = 0,
        all_mentioned_rooms: Optional[List[str]] = None,
    ) -> Tuple[plt.Axes, float, float]:
        all_rooms = []
        if all_mentioned_rooms is None:
            all_mentioned_rooms = mentioned_rooms
        # Keep mentioned rooms first
        for room in self.rooms:
            if room.room_id in all_mentioned_rooms:
                all_rooms.append(room)
        # Then keep non-mentioned rooms
        for room in self.rooms:
            if room.room_id not in all_mentioned_rooms:
                all_rooms.append(room)

        current_rooms_to_plot = []
        current_row_width: float = 0
        current_row_height: float = 0 + height_offset
        i = 0
        while i < len(all_rooms):
            room = all_rooms[i]
            if room.width + current_row_width <= target_width:
                current_row_width += room.width
                current_rooms_to_plot.append(room)
            else:
                current_row_width, current_row_height = self.plot_a_row_of_rooms(
                    ax, current_rooms_to_plot, target_width, current_row_height
                )
                current_row_width = 0
                current_rooms_to_plot = []
                continue
            i += 1

        current_row_width, current_row_height = self.plot_a_row_of_rooms(
            ax, current_rooms_to_plot, target_width, current_row_height
        )

        height_upper = 0.0
        height_lower = current_row_height

        return ax, height_lower, height_upper

    def process_state_changes(self, propositions: List[Dict[str, Any]]) -> None:
        for proposition in propositions:
            function_name = proposition["function_name"]
            args = proposition["args"]
            if function_name in [
                "is_clean",
                "is_filled",
                "is_powered_on",
                "is_powered_off",
            ]:
                if "object_names" in args:
                    object_names = args["object_names"]
                    for object_name in object_names:
                        for room in self.rooms:
                            object_obj = room.find_object_by_id(object_name)
                            if object_obj:
                                if function_name != "is_powered_off":
                                    object_obj.states[function_name] = True
                                else:
                                    object_obj.states["is_powered_on"] = False
                else:
                    receptacle_names = args["receptacle_names"]
                    for receptacle_name in receptacle_names:
                        for room in self.rooms:
                            receptacle_obj = room.find_receptacle_by_id(receptacle_name)
                            if receptacle_obj:
                                if function_name != "is_powered_off":
                                    receptacle_obj.states[function_name] = True
                                else:
                                    receptacle_obj.states["is_powered_on"] = False

    def plot_propositions(
        self, ax: plt.Axes, propositions: List[Dict[str, Any]], height_upper: float
    ) -> None:
        for proposition in propositions:
            function_name = proposition["function_name"]
            args = proposition["args"]
            if "object_names" in args:
                # Handles is_on_top, is_inside, is_on_floor
                object_names = args["object_names"]
                number = args["number"]
                color = proposition["color"]
                if function_name in ["is_inside", "is_on_top"]:
                    receptacle_names = args["receptacle_names"]
                    self.plot_object_to_receptacle_lines(
                        object_names,
                        receptacle_names,
                        number,
                        function_name,
                        ax,
                        color,
                        height_upper,
                    )
                elif function_name == "is_in_room":
                    room_names = args["room_names"]
                    self.plot_object_to_room_lines(
                        object_names,
                        room_names,
                        number,
                        ax,
                        color,
                        height_upper,
                    )

    def plot_time_step_box(
        self, ax: plt.Axes, step_idx: int, height_upper: float
    ) -> None:
        # plot a small square on top right corner to indicate the time step
        # TODO: Remove the hardcoded values for the square and text
        ax.add_patch(
            plt.Rectangle(
                (self.width - 120, height_upper - 120),
                110,
                110,
                facecolor="white",
                edgecolor="black",
                linewidth=0,
            )
        )
        ax.text(
            self.width - 70,
            height_upper - 70,
            f"{step_idx}",
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
            color="black",
        )

    def plot_one_time_step(
        self,
        ax: plt.Axes,
        propositions: List[Dict[str, Any]],
        mentioned_rooms: List[str],
        step_idx: int = 0,
        height_offset: float = 0,
        all_mentioned_rooms: Optional[List[str]] = None,
        show_time_step_box: bool = False,
    ) -> Tuple[plt.Axes, float, float]:
        self.process_state_changes(propositions)

        ax, height_lower, height_upper = self.plot_room_rows(
            mentioned_rooms,
            ax,
            self.config.target_width,
            height_offset,
            all_mentioned_rooms,
        )

        self.plot_propositions(ax, propositions, height_offset)
        if show_time_step_box:
            self.plot_time_step_box(ax, step_idx + 1, height_upper)

        return ax, height_lower, height_upper

    def get_all_mentioned_rooms(self, propositions: List[Dict[str, Any]]) -> List[str]:
        # NOTE: All the next bits are only used to get a list of all the mentioned rooms to keep them in front!
        # We don't really care about using mentioned objects and receptacles after
        mentioned_objs = []
        mentioned_receps = []
        mentioned_rooms = []
        for prop in propositions:
            if prop["function_name"] in ["is_on_top", "is_inside"]:
                mentioned_objs += prop["args"]["object_names"]
                if prop["function_name"] == "is_on_top":
                    mentioned_receps += [
                        ("is_on_top", recep_name)
                        for recep_name in prop["args"]["receptacle_names"]
                    ]
                if prop["function_name"] == "is_inside":
                    mentioned_receps += [
                        ("is_inside", recep_name)
                        for recep_name in prop["args"]["receptacle_names"]
                    ]
            elif prop["function_name"] == "is_in_room":
                mentioned_objs += prop["args"]["object_names"]
                mentioned_rooms += prop["args"]["room_names"]
            elif prop["function_name"] == "is_on_floor":
                mentioned_objs += prop["args"]["object_names"]
            elif prop["function_name"] == "is_next_to":
                continue
            elif prop["function_name"] in [
                "is_clean",
                "is_filled",
                "is_powered_on",
                "is_powered_off",
            ]:
                if "object_names" in prop["args"]:
                    mentioned_objs += prop["args"]["object_names"]
                elif "receptacle_names" in prop["args"]:
                    mentioned_receps += [
                        (prop["function_name"], recep_name)
                        for recep_name in prop["args"]["receptacle_names"]
                    ]
                else:
                    raise NotImplementedError(
                        f"Not implemented for function with name: {prop['function_name']} and no receptacle_names or object_names."
                    )
            else:
                raise NotImplementedError(
                    f"Not implemented for function with name: {prop['function_name']}."
                )

        for room in self.rooms:
            for obj in mentioned_objs:
                found_object = room.find_object_by_id(obj)
                if found_object and room.room_id not in mentioned_rooms:
                    mentioned_rooms += [room.room_id]
            for _prop_function, recep in mentioned_receps:
                found_receptacle = room.find_receptacle_by_id(recep)
                if found_receptacle and room.room_id not in mentioned_rooms:
                    mentioned_rooms += [room.room_id]

        all_mentioned_rooms = sorted(mentioned_rooms)
        return all_mentioned_rooms

    def extract_info_before_plot(self, propositions: List[Dict[str, Any]]) -> List[str]:
        # Extract room names mentioned in propositions
        mentioned_objs = []
        on_floor_objs = []
        mentioned_receps = []
        mentioned_rooms = []
        for prop in propositions:
            if prop["function_name"] in ["is_on_top", "is_inside"]:
                mentioned_objs += prop["args"]["object_names"]
                if prop["function_name"] == "is_on_top":
                    mentioned_receps += [
                        ("is_on_top", recep_name)
                        for recep_name in prop["args"]["receptacle_names"]
                    ]
                if prop["function_name"] == "is_inside":
                    mentioned_receps += [
                        ("is_inside", recep_name)
                        for recep_name in prop["args"]["receptacle_names"]
                    ]
            elif prop["function_name"] == "is_in_room":
                mentioned_objs += prop["args"]["object_names"]
                mentioned_rooms += prop["args"]["room_names"]
            elif prop["function_name"] == "is_on_floor":
                on_floor_objs += prop["args"]["object_names"]

        for room in self.rooms:
            if room.room_id in mentioned_rooms:
                room.plot_placeholder = True
            else:
                room.plot_placeholder = False

        for room in self.rooms:
            for receptacle in room.receptacles:
                receptacle.plot_top_placeholder = False
                receptacle.plot_center_placeholder = False
            for obj in room.objects:
                obj.is_on_floor = False

        for room in self.rooms:
            for object_name in on_floor_objs:
                found_object = room.find_object_by_id(object_name)
                if found_object:
                    found_object.is_on_floor = True
                    if room.room_id not in mentioned_rooms:
                        mentioned_rooms += [room.room_id]

            for object_name in mentioned_objs:
                found_object = room.find_object_by_id(object_name)
                if found_object and room.room_id not in mentioned_rooms:
                    mentioned_rooms += [room.room_id]
            for prop_function, recep in mentioned_receps:
                found_receptacle = room.find_receptacle_by_id(recep)
                if found_receptacle:
                    if room.room_id not in mentioned_rooms:
                        mentioned_rooms += [room.room_id]
                    if prop_function == "is_on_top":
                        found_receptacle.plot_top_placeholder = True
                    elif prop_function == "is_inside":
                        found_receptacle.plot_center_placeholder = True
                    else:
                        raise NotImplementedError(
                            f"Not implemented for prop fuction {prop_function}."
                        )

        for room in self.rooms:
            if room.room_id in mentioned_rooms:
                room.in_proposition = True
            else:
                room.in_proposition = False
        return mentioned_rooms

    def update_rooms(
        self,
        current_propositions: List[Dict[str, Any]],
        evaluation_propositions: List[Dict[str, Any]],
        evaluation_constraints: List[Dict[str, Any]],
        global_to_local_idx: Dict[int, int],
    ) -> None:
        current_propositions = deepcopy(current_propositions)
        evaluation_constraints = deepcopy(evaluation_constraints)
        # Update the object_to_recep and object_to_room mappings
        new_object_to_recep, new_object_to_room = update_object_recep_and_room(
            self.object_to_recep,
            self.object_to_room,
            current_propositions,
            evaluation_propositions,
            evaluation_constraints,
            global_to_local_idx,
        )

        # Handle objects with only new receptacle mappings but no new room mappings
        for obj_name, new_recep in new_object_to_recep.items():
            # Find the current room containing the object
            for room in self.rooms:
                obj = room.find_object_by_id(obj_name)
                if obj:
                    # Remove the object from the current room
                    room.objects.remove(obj)
                    break

            # Find the room containing the new receptacle
            for room in self.rooms:
                if (
                    new_recep == "floor"
                    and room.room_id == new_object_to_room[obj_name]
                ):
                    room.objects.append(obj)
                    new_object_to_room[obj_name] = room.room_id

                recep = room.find_receptacle_by_id(new_recep)
                if recep:
                    # Add the object to the room containing the new receptacle
                    room.objects.append(obj)
                    new_object_to_room[obj_name] = room.room_id
                    break
        # Move objects across rooms and update attributes
        for obj_name, new_room_id in new_object_to_room.items():
            new_room = self.room_id_to_room.get(new_room_id)
            if new_room:
                for room in self.rooms:
                    obj = room.find_object_by_id(obj_name)
                    if obj:
                        # Remove the object from the current room
                        room.objects.remove(obj)
                        break
                # Add the object to the new room
                new_room.objects.append(obj)

        # Update mapping for all rooms
        for room in self.rooms:
            room.object_to_recep = new_object_to_recep

        # Update the mappings in the object
        self.object_to_recep = new_object_to_recep
        self.object_to_room = new_object_to_room

    def plot(
        self,
        propositions: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        toposort: Optional[List[List[int]]],
        ax: Optional[plt.Axes] = None,
    ) -> List[
        Tuple[
            plt.Figure,
            Union[Optional[plt.Axes], plt.Axes],
            float,
            float,
            Dict[int, Tuple[float, float, float]],
        ]
    ]:
        prop_to_height_range = {}
        # TODO: A lot of the code below is redundant and can be refactored
        if toposort:
            all_mentioned_rooms = self.get_all_mentioned_rooms(propositions)
            fig_data = []
            for level_idx, current_level in enumerate(toposort):
                # We create a new figure every time
                max_upper = 0.0
                min_lower = 0.0
                fig, ax = plt.subplots()
                fig.patch.set_facecolor(BACKGROUND_COLOR)
                current_propositions = [propositions[idx] for idx in current_level]
                global_to_local_idx = {
                    global_idx: local_idx
                    for local_idx, global_idx in enumerate(current_level)
                }
                mentioned_rooms = self.extract_info_before_plot(current_propositions)
                ax, height_lower, height_upper = self.plot_one_time_step(
                    ax,
                    current_propositions,
                    mentioned_rooms,
                    step_idx=level_idx,
                    height_offset=min_lower,
                    all_mentioned_rooms=all_mentioned_rooms,
                )
                prop_to_height_range[level_idx] = (
                    height_lower,
                    min_lower,
                    max_upper - min_lower,
                )
                max_upper = max(height_upper, max_upper)
                min_lower = min(height_lower, min_lower)

                # Move around objects:
                self.update_rooms(
                    current_propositions,
                    propositions,
                    constraints,
                    global_to_local_idx,
                )

                # Reset room widths
                for room in self.rooms:
                    room.init_size()
                fig_data.append((fig, ax, min_lower, max_upper, prop_to_height_range))
            self.height = max_upper - min_lower  # Does not matter
            return fig_data
        else:
            if ax is None:
                fig, ax = plt.subplots()
                fig.patch.set_facecolor(BACKGROUND_COLOR)
            else:
                fig = plt.gcf()
            mentioned_rooms = self.extract_info_before_plot(propositions)
            ax, height_lower, height_upper = self.plot_one_time_step(
                ax,
                propositions,
                mentioned_rooms,
                step_idx=0,
            )
            prop_to_height_range[0] = (height_lower, height_upper, 0)
            self.height = height_upper - height_lower
            return [(fig, ax, height_lower, height_upper, prop_to_height_range)]
