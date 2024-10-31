#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

import matplotlib.pyplot as plt
import networkx as nx

from ..constants import CROPPED_RECEPTACLE_ICONS_PATH, LEGEND_BACKGROUND_COLOR
from ..object import Object
from ..receptacle import Receptacle
from ..tiny_room import TinyRoom


class SameArgsLegend:
    def __init__(self, config, same_args, receptacle_icon_mapping, propositions):
        self.title = "same as"
        self.config = config
        # same_args is a list of lists
        # each inner list is a list of tuples
        # the first item in each tuple is supposed to be the "same" element
        # the second item in each tuple is the thing to be on the other side
        # each item has its type described in the tuple
        self.same_args = same_args
        self.propositions = propositions
        self.receptacle_icon_mapping = receptacle_icon_mapping
        self.set_graph_and_bipartite_sets()
        self.set_height()

    @property
    def width(self):
        return self.config.width

    @property
    def horizontal_margin(self):
        return self.config.horizontal_margin

    @property
    def top_pad(self):
        return self.config.top_pad

    @property
    def bottom_pad(self):
        return self.config.bottom_pad

    def set_graph_and_bipartite_sets(self):
        self.graphs = []
        self.left_sets = []
        self.right_sets = []
        self.edge_styles = []
        self.color_dicts = []
        for same_args_data in self.same_args:
            G = nx.Graph()
            edge_style = {}
            color_dict = {}
            left_elements = set()
            right_elements = []
            line_styles = []
            colors = []
            for item in same_args_data:
                left_elements = left_elements.union(set(item["common_entities"]))
                right_elements = right_elements + list(item["corresponding_entities"])
                line_styles = line_styles + [
                    item["line_style"] for i in item["corresponding_entities"]
                ]
                colors = colors + [
                    self.propositions[item["global_proposition_index"]]["color"]
                    for i in item["corresponding_entities"]
                ]
            left_elements = list(left_elements)
            right_elements = list(right_elements)

            # TODO: Think if we need a better logic here.
            # Currently just picking one of the elements here.
            # left element is the common element
            left_element = left_elements[0]
            if not G.has_node(left_element[0]):
                G.add_node(left_element[0], entity=left_element, bipartite=0)
            for item, line_style, color in zip(right_elements, line_styles, colors):
                node_label = item[0]
                if not G.has_node(node_label):
                    G.add_node(node_label, entity=item, bipartite=1)
                G.add_edge(left_element[0], node_label)
                edge_style[(left_element[0], node_label)] = line_style
                color_dict[(left_element[0], node_label)] = color
            self.edge_styles.append(edge_style)
            self.color_dicts.append(color_dict)
            left_set = set()
            right_set = set()
            # Get the two sets of nodes
            for _idx, (label, data) in enumerate(G.nodes(data=True)):
                if data["bipartite"] == 0:
                    left_set = left_set.union({label})
                else:
                    right_set = right_set.union({label})
            self.graphs.append(G)
            self.left_sets.append(left_set)
            self.right_sets.append(right_set)

    def set_height(self):
        self.left_set_length = 0
        self.left_consumed_space = 0
        for left_set, G in zip(self.left_sets, self.graphs):
            for node in left_set:
                self.left_set_length += 1
                entity_id, entity_type = G.nodes[node]["entity"]
                if entity_type == "object":
                    self.left_consumed_space += self.config.object.height
                elif entity_type == "receptacle":
                    self.left_consumed_space += self.config.receptacle.target_height
                else:
                    self.left_consumed_space += self.config.room.height

        self.right_set_length = 0
        self.right_consumed_space = 0
        for right_set, G in zip(self.right_sets, self.graphs):
            for node in right_set:
                self.right_set_length += 1
                entity_id, entity_type = G.nodes[node]["entity"]
                if entity_type == "object":
                    self.right_consumed_space += self.config.object.height
                elif entity_type == "receptacle":
                    self.right_consumed_space += self.config.receptacle.target_height
                else:
                    self.right_consumed_space += self.config.room.height
        self.height = max(
            self.left_consumed_space
            + self.left_set_length * self.config.around_entity_spacing,
            self.right_consumed_space
            + self.right_set_length * self.config.around_entity_spacing,
        )

    def plot_entity_column(
        self,
        ax,
        G,
        entity_set,
        midpoint,
        current_height,
        spacing,
    ):
        entities = {}
        for _idx, node in enumerate(entity_set):
            entity_id, entity_type = G.nodes[node]["entity"]
            if entity_type == "object":
                entity = Object(self.config, entity_id)
                origin = (
                    midpoint - self.config.object.width / 2,
                    current_height - self.config.object.height / 2,
                )
            elif entity_type == "receptacle":
                icon_path = self.receptacle_icon_mapping.get(
                    entity_id, f"{CROPPED_RECEPTACLE_ICONS_PATH}/chair@2x.png"
                )
                entity = Receptacle(self.config, entity_id, icon_path)
                origin = (
                    midpoint - entity.width / 2,
                    current_height - entity.height / 2,
                )
            else:
                entity = TinyRoom(self.config, entity_id)
                origin = (
                    midpoint - entity.width / 2,
                    current_height - entity.height / 2,
                )
            entities[f"{entity_id}"] = entity
            entity.plot(
                ax,
                origin,
            )
            current_height -= spacing
        return entities, current_height

    def plot_lines(self, G, ax, left_entities, right_entities, edge_style, color_dict):
        for edge in G.edges():
            node1, node2 = edge
            line_style = edge_style[(node1, node2)]
            color = color_dict[(node1, node2)]
            if node1 in left_entities:
                first_entity = left_entities[node1]
            elif node1 in right_entities:
                first_entity = right_entities[node1]
            else:
                raise RuntimeError(f"node1: {node1} not in left or right entities")
            if node2 in left_entities:
                second_entity = left_entities[node2]
            elif node2 in right_entities:
                second_entity = right_entities[node2]
            else:
                raise RuntimeError(f"node2: {node2} not in left or right entities")
            first_center = (
                first_entity.center_position
                if isinstance(first_entity, Object)
                else (
                    first_entity.center_placeholder_position
                    if isinstance(first_entity, Receptacle)
                    else first_entity.center_position
                )
            )
            second_center = (
                second_entity.center_position
                if isinstance(second_entity, Object)
                else (
                    second_entity.center_placeholder_position
                    if isinstance(second_entity, Receptacle)
                    else second_entity.center_position
                )
            )
            ax.plot(
                [first_center[0], second_center[0]],
                [first_center[1], second_center[1]],
                linestyle=line_style,
                linewidth=self.config.linewidth,
                color=color,
            )

            # Mark the end points of the lines with larger solid dots
            ax.scatter(
                first_center[0],
                first_center[1],
                color="white",
                s=self.config.endpoint_size,
                zorder=2,
            )
            ax.scatter(
                second_center[0],
                second_center[1],
                color=color,
                s=self.config.endpoint_size,
                zorder=2,
            )

    def plot(self, position=(0, 0), ax=None):
        # Plotting logic
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            created_fig = False

        # Plot the box
        rect = ax.add_patch(
            plt.Rectangle(
                (
                    position[0] + self.config.horizontal_margin,  # margin
                    position[1],
                ),
                self.config.width,
                self.height + self.config.top_pad + self.config.bottom_pad,
                edgecolor="white",
                linewidth=0,
                facecolor=LEGEND_BACKGROUND_COLOR,
            )
        )
        # Set the z-order of the rectangle
        rect.set_zorder(-1)

        left_spacing = self.height / self.left_set_length
        right_spacing = self.height / self.right_set_length

        # Plot the left nodes
        left_midpoint = (
            position[0] + self.config.horizontal_margin + self.config.width / 4
        )
        left_current_height = (
            position[1] + self.height + self.config.bottom_pad - left_spacing / 2
        )  # top padding
        left_entities = []
        for G, left_set in zip(self.graphs, self.left_sets):
            new_left_entities, left_current_height = self.plot_entity_column(
                ax, G, left_set, left_midpoint, left_current_height, left_spacing
            )
            left_entities.append(new_left_entities)

        # Plot the right nodes
        right_midpoint = (
            position[0] + self.config.horizontal_margin + 3 * self.config.width / 4
        )
        right_current_height = (
            position[1] + self.height + self.config.bottom_pad - right_spacing / 2
        )  # top padding
        right_entities = []
        for G, right_set in zip(self.graphs, self.right_sets):
            new_right_entities, right_current_height = self.plot_entity_column(
                ax, G, right_set, right_midpoint, right_current_height, right_spacing
            )
            right_entities.append(new_right_entities)

        for (
            G,
            current_left_entities,
            current_right_entities,
            edge_style,
            color_dict,
        ) in zip(
            self.graphs,
            left_entities,
            right_entities,
            self.edge_styles,
            self.color_dicts,
        ):
            self.plot_lines(
                G,
                ax,
                current_left_entities,
                current_right_entities,
                edge_style,
                color_dict,
            )

        # title
        ax.text(
            position[0] + self.config.horizontal_margin + self.config.width / 2,
            position[1] + self.height + self.config.bottom_pad,
            self.title,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=self.config.text_size,
            zorder=float("inf"),
        )

        if created_fig:
            return fig, ax
        else:
            return ax
