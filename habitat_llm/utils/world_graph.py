#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from habitat_llm.utils.core import cprint
from habitat_llm.world_model.world_graph import WorldGraph


def print_all_entities(world_graph: WorldGraph) -> None:
    """
    Prints all relevant WorldGraph.Entity to the console by type. Makes it easier to sandbox skill commands by providing active entities to target.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Currently available Entities:", "green")
    cprint(" Rooms: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_rooms()]}", "yellow")
    cprint(" Furniture: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_furnitures()]}", "yellow")
    cprint(" Objects: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_objects()]}", "yellow")
    cprint(" Receptacles: ", "green")
    cprint(f"  {[node.name for node in world_graph.get_all_receptacles()]}", "yellow")
    print("\n")


def print_furniture_entity_handles(world_graph: WorldGraph) -> None:
    """
    Prints a map of active Entity.Furniture names to their sim_handles to console. Makes it easier to debug by mapping planner commands to simulation objects.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Furniture Names to Handles:", "green")
    for entity in world_graph.get_all_furnitures():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        cprint(f"  {entity.name} : {sim_handle}", "yellow")
    print("\n")


def print_object_entity_handles(world_graph: WorldGraph) -> None:
    """
    Prints a map of active Entity.Object names to their sim_handles to console. Makes it easier to debug by mapping planner commands to simulation objects.

    :param world_graph: The active WorldGraph with all instantiated entities.
    """
    print("\n")
    cprint("Object Names to Handles:", "green")
    for entity in world_graph.get_all_objects():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        cprint(f"  {entity.name} : {sim_handle}", "yellow")
    print("\n")


def get_all_entity_names(world_graph: WorldGraph) -> List[str]:
    """
    Get a list of semantic names for all navigable entities.

    :param world_graph: The active WorldGraph with all instantiated entities.
    :return: The list of all names for navigable entities. For example, to quickly do nav to all testing.
    """
    rooms = [node.name for node in world_graph.get_all_rooms()]
    furniture = [node.name for node in world_graph.get_all_furnitures()]
    objs = [node.name for node in world_graph.get_all_objects()]
    recs = [node.name for node in world_graph.get_all_receptacles()]
    return rooms + furniture + objs + recs
