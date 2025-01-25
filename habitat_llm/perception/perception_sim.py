#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Implements privileged perception class PerceptionSim. This class has access
to the underlying Simulator instance running the episodes and uses ground-truth
information to generate current-state as a world-graph. Also has logic required to
simulator partial-observation condition which generates ground-truth world-graph over
only the entities discovered so far by the agents."""

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    pass

import habitat.sims.habitat_simulator.sim_utilities as sutils
import numpy as np
import pandas as pd
from habitat.core.logging import logger
from habitat.datasets.rearrange.samplers.receptacle import Receptacle as HabReceptacle
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_handle,
    get_obj_from_id,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from magnum import Vector3

from habitat_llm.perception.perception import Perception
from habitat_llm.sims.metadata_interface import MetadataInterface
from habitat_llm.utils.sim import get_faucet_points, get_receptacle_dict
from habitat_llm.world_model import (
    Floor,
    Furniture,
    House,
    Human,
    Object,
    Receptacle,
    Room,
    SpotRobot,
)
from habitat_llm.world_model.graph import Graph
from habitat_llm.world_model.world_graph import WorldGraph, flip_edge

HUMAN_SEMANTIC_ID = 100  # special semantic ID reserved for humanoids
UNKNOWN_SEMANTIC_ID = 0  # special semantic ID reserved for unknown object class


class PerceptionSim(Perception):
    """
    This class represents simulated perception stack of the agents.
    """

    # Parameterized Constructor
    def __init__(
        self, sim: RearrangeSim, metadata_dict: Dict[str, str] = None, detectors=None
    ):
        # Call base class constructor
        super().__init__(detectors)

        # Load the metadata
        self.metadata_interface: MetadataInterface = None
        if metadata_dict is not None:
            self.metadata_interface = MetadataInterface(metadata_dict)

        if not sim:
            raise ValueError("Cannot construct PerceptionSim with sim as None")
        self.sim = sim

        # Container to store mapping from parent furniture object handles to child receptacles
        self.fur_obj_handle_to_recs: Dict[str, Dict[str, List[HabReceptacle]]] = None

        # Container to map handles to names
        self.sim_handle_to_name: Dict[str, str] = {}

        # Container to map region ids to rooms
        self.region_id_to_name: Dict[str, str] = {}

        # Fetch the rigid and articulated object manager
        self.rom = sim.get_rigid_object_manager()
        self.aom = sim.get_articulated_object_manager()

        # Container to store ground truth sim graph
        self.gt_graph = WorldGraph()

        # Add root node house to the gt graph
        self.add_house_to_graph()

        # Add all rooms to the gt graph
        self.add_rooms_to_gt_graph()

        # Add all floors to the gt graph
        self.add_floors_to_gt_graph()

        # Add all receptacles to the gt graph
        self.add_furniture_and_receptacles_to_gt_graph()

        # Add objects to the graph.
        self.add_objects_to_gt_graph()

        # Add agents to the graph
        # This together with the above command completes the scene initialization.
        self.add_agents_to_gt_graph()

        # Cache of receptacle names containing objects.
        self._obj_to_rec_cache: Dict[str, str] = {}

        # Cache of object positions.
        self._obj_position_cache: Dict[str, Vector3] = {}

    @property
    def receptacles(self) -> Dict[str, HabReceptacle]:
        """
        Get the filtered HabReceptacles from RearrangeSim.
        """
        return self.sim.receptacles

    @property
    def metadata(self) -> pd.DataFrame:
        """
        The subordinate MetadataInterface's loaded metadata DataFrame.

        :return: The metadata dataframe about objects and receptacles from csv and json files.
        This data will typically include tags associated with these objects such as semantic class, product descriptions, object state affordances, etc...
        """

        if self.metadata_interface is None:
            return None
        return self.metadata_interface.metadata

    def get_furniture_property_from_metadata(self, handle: str, prop: str) -> str:
        """
        This method returns value of the requested property using metadata file.
        For example, this could be used to extract the semantic type of any object
        in HSSD. Note that the property should exist in the metadata file.

        :param handle: String sim-handle of the furniture to query
        :param prop: String identifier for the furniture property to read

        :return: String value of the furniture' property as initialized in metadata
        """
        # Declare default
        property_value = "unknown"

        # get hash from handle
        # handle_hash = handle.split(".", 1)[0] if "." in handle else handle.split("_", 1)[0]
        handle_hash = (
            handle.split(".")[0] if "." in handle else handle.rpartition("_")[0]
        )

        # Use loc to locate the row with the specific key
        object_row = self.metadata.loc[self.metadata["handle"] == handle_hash]

        # Extract the value from the object_row
        if not object_row.empty:
            # Make sure the property value is not nan or empty
            if object_row[prop].notna().any() and (object_row[prop] != "").any():
                property_value = object_row[prop].values[0]
        else:
            raise ValueError(f"Handle {handle} not found in the metadata.")

        return property_value

    def get_room_name(self, handle: str) -> str:
        """
        Get the name of the room that contains a given object based off of the simulator object regions.

        :param handle: The handle of the object.

        :return: The name of the room that contains the object or 'unknown_room' if not found.

        :raises ValueError: If the object is not in any region.
        """
        ao_link_map = sutils.get_ao_link_id_map(self.sim)
        regions = sutils.get_object_regions(
            self.sim, get_obj_from_handle(self.sim, handle), ao_link_map=ao_link_map
        )
        if len(regions) == 0:
            # raise ValueError(f"Object is not in any region: {handle}")
            room_name = "unknown_room"
        else:
            region_index, _ = regions[0]
            region_id = self.sim.semantic_scene.regions[region_index].id
            room_name = self.region_id_to_name[region_id]
        return room_name

    def get_latest_objects_to_receptacle_map(self) -> Dict[str, str]:
        """
        This method returns a dict which maps objects
        to their current receptacles in the sim

        :param sim: Simulator instance

        :return: Dict mapping an object' str object-UID to the str name of receptacle
        associated with it
        """
        objects = self.gt_graph.get_all_objects()
        for obj in objects:
            obj_name = obj.name
            obj_pos = obj.properties["translation"]
            cached_pos = self._obj_position_cache.get(obj_name, None)
            if cached_pos is None or cached_pos != obj_pos:
                obj_handle = obj.sim_handle
                rec_name = self._get_current_receptacle_name(obj_handle)
                # NOTE: rec_name could be None if the matching failed
                if rec_name is None:
                    rec_name = "unknown_room"
                self._obj_position_cache[obj_name] = obj_pos
                self._obj_to_rec_cache[obj_name] = rec_name

        return self._obj_to_rec_cache

    def add_house_to_graph(self) -> None:
        """
        This method adds the root node house to the the gt_graph.
        """
        # Create root node
        house = House("house", {"type": "root"}, "house_0")
        self.gt_graph.add_node(house)

    def add_rooms_to_gt_graph(self) -> None:
        """
        This method adds room nodes to the gt_graph.
        This is done by querying in which room does a given furniture lie.

        :param sim: Simulator instance

        """

        # Add room nodes to the graph
        region_names = {}
        if len(self.sim.semantic_scene.regions) == 0:
            raise ValueError(
                f"No regions found in the scene: {self.sim.ep_info['scene_id']}"
            )

        for region_idx, region in enumerate(self.sim.semantic_scene.regions):
            region_name = region.category.name().split("/")[0].replace(" ", "_")
            if region_name not in region_names:
                region_names[region_name] = 0
            region_names[region_name] = region_names[region_name] + 1
            room_name = f"{region_name}_{region_names[region_name]}"

            # Add a valid point on floor as room location
            point_on_floor = sutils.get_floor_point_in_region(self.sim, region_idx)

            # Create properties dict
            if point_on_floor is not None:
                point_on_floor = list(point_on_floor)
                properties = {"type": region_name, "translation": point_on_floor}
            else:
                properties = {"type": region_name}

            # Create room node
            room = Room(room_name, properties, room_name)

            # Update mapping from region id to room name
            self.region_id_to_name[region.id] = room_name

            # Add room nodes to the ground truth graph
            # The edges to furniture will be added
            # in the add_furniture_and_receptacles_to_gt_graph method
            self.gt_graph.add_node(room)

            # Connect room to the root node house
            self.gt_graph.add_edge(room, "house", "inside", flip_edge("inside"))

        # Add an unknown room for redundancy
        properties_unknown = {"type": "unknown"}
        unknown_room = Room("unknown_room", properties_unknown, "unknown_room")

        # Add unknown room nodes to the ground truth graph
        self.gt_graph.add_node(unknown_room)

        # Connect room to the root node house
        self.gt_graph.add_edge(unknown_room, "house", "inside", flip_edge("inside"))

    def add_floors_to_gt_graph(self) -> None:
        """
        This method adds floor nodes to the gt_graph.
        This is done by finding all rooms and adding floors as a child.
        """
        for room in self.gt_graph.get_all_rooms():
            properties = {"type": "floor"}
            if "translation" in room.properties:
                properties["translation"] = room.properties["translation"]
                floor = Floor(f"floor_{room.name}", properties)
                self.gt_graph.add_node(floor)
                self.gt_graph.add_edge(floor, room, "inside", flip_edge("inside"))

    def add_furniture_and_receptacles_to_gt_graph(self) -> None:
        """
        Adds all furniture and corresponding receptacles to the graph during graph initialization
        """

        # Make sure that the metadata is not None
        if self.metadata is None:
            raise ValueError("Trying to load furniture from sim, but metadata was None")

        # Get faucet locations
        faucet_points = get_faucet_points(self.sim)

        # Get dict mapping furniture sim handles to "on" and "within" sets containing lists of HabReceptacles
        self.fur_obj_handle_to_recs = get_receptacle_dict(
            self.sim, list(self.receptacles.values())
        )

        # Iterate through furniture to rec dict and populate the graph
        for furniture_sim_handle in self.fur_obj_handle_to_recs:
            fur_obj = sutils.get_obj_from_handle(self.sim, furniture_sim_handle)
            # Get furniture type using metadata
            furniture_type = self.get_furniture_property_from_metadata(
                furniture_sim_handle, "type"
            )

            # Generate name for furniture
            furniture_name = (
                f"{furniture_type}_{self.gt_graph.count_nodes_of_type(Furniture)}"
            )

            # Create properties dict
            properties = {
                "type": furniture_type,
                "is_articulated": fur_obj.is_articulated,
                "translation": fur_obj.translation,
                # An array to track non-receptacle sub-components of the furniture, i.e. faucet, power outlets,
                "components": [],
            }

            if furniture_sim_handle in faucet_points:
                properties["components"].append("faucet")

            # Create furniture instance and receptacle instance
            fur = Furniture(furniture_name, properties, furniture_sim_handle)

            # Add furniture to the graph
            self.gt_graph.add_node(fur)

            # Add name to handle mapping
            self.sim_handle_to_name[furniture_sim_handle] = furniture_name

            # Fetch room for this furniture
            room_name = self.get_room_name(furniture_sim_handle)

            # Add edge between furniture and room
            self.gt_graph.add_edge(fur, room_name, "inside", flip_edge("inside"))

            # Add receptacles of this furniture
            rec_counter = 0
            for proposition in self.fur_obj_handle_to_recs[furniture_sim_handle]:
                rec_list = self.fur_obj_handle_to_recs[furniture_sim_handle][
                    proposition
                ]
                for hab_rec in rec_list:
                    # Add receptacle to the graph
                    rec_name = f"rec_{furniture_name}_{rec_counter}"
                    rec = Receptacle(
                        rec_name, {"type": proposition}, hab_rec.unique_name
                    )
                    self.gt_graph.add_node(rec)

                    # Add rec name to handle mapping
                    self.sim_handle_to_name[hab_rec.unique_name] = rec_name

                    # Connect receptacle to the furniture under consideration
                    self.gt_graph.add_edge(rec, furniture_name, "joint", "joint")

                    # increment rec counter
                    rec_counter += 1

        # Confirm that the gt graph is not empty
        if self.gt_graph.is_empty():
            raise ValueError(
                "Attempted to load all furniture, but none were found in the scene"
            )

    def add_objects_to_gt_graph(self) -> None:
        """
        This method adds objects to the gt_graph during the graph initialization
        """

        # Make sure that sim is not None
        if not self.sim:
            raise ValueError("Trying to load objects from sim, but sim was None")

        # Make sure that the metadata is not None
        if self.metadata is None:
            raise ValueError("Trying to load objects from sim, but metadata was None")

        # Add object nodes to the graph
        for obj_handle, fur_rec_handle in self.sim.ep_info.name_to_receptacle.items():
            sim_obj = sutils.get_obj_from_handle(self.sim, obj_handle)
            # Get object type
            obj_type = self.metadata_interface.get_object_instance_category(sim_obj)

            # Get object position
            translation = list(sim_obj.translation)

            # Create properties dict
            properties = {"type": obj_type, "translation": translation, "states": {}}

            # Create object name
            obj_name = f"{obj_type}_{self.gt_graph.count_nodes_of_type(Object)}"
            self.sim_handle_to_name[obj_handle] = obj_name

            # Construct object based on the information
            obj = Object(obj_name, properties, obj_handle)

            # Add object node to the graph
            self.gt_graph.add_node(obj)

            # Connect object to the receptacle
            if "floor" in fur_rec_handle:
                room_name = self.get_room_name(obj_handle)
                floor_name = f"floor_{room_name}"
                floor_node = self.gt_graph.get_node_from_name(floor_name)
                self.gt_graph.add_edge(obj, floor_node, "on", flip_edge("on"))
            elif fur_rec_handle in self.sim_handle_to_name:
                self.gt_graph.add_edge(
                    obj, self.sim_handle_to_name[fur_rec_handle], "on", flip_edge("on")
                )
            else:
                logger.error(
                    f"Failed to find the expected relationship {fur_rec_handle}. Receptacle doesn't exist, skipping graph edge creation."
                )

            # The object is being connected based on the fur_rec_handle
            # This is done here because, the rec_handle by itself was not
            # found to be unique for multiple receptacles but fur_rec_handle
            # was unique.

        self.update_object_and_furniture_states()

    def add_agents_to_gt_graph(self) -> None:
        """
        Method to add agents to the ground truth graph during initialization.
        """
        # Make sure that sim is not None
        if not self.sim:
            raise ValueError("Trying to load agents from sim, but sim was None")

        # Add agents to the graph
        for agent_name in self.sim.agents_mgr.agent_names:
            # Get agent id from name
            try:
                agent_id = int(agent_name.split("_")[1])
            except ValueError:
                agent_id = 0

            # Get articulated agent
            if isinstance(self.sim.agents_mgr, list):
                articulated_agent = self.sim.agents_mgr[agent_id].articulated_agent
            else:
                articulated_agent = self.sim.agents_mgr._all_agent_data[
                    agent_id
                ].articulated_agent

            # Get agent position
            translation = list(articulated_agent.base_pos)

            # Create properties dict
            properties = {"translation": translation, "is_articulated": True}

            # Add Agent node to the world
            agent: Union[Human, SpotRobot]
            if agent_id == 0:
                agent = SpotRobot(agent_name, properties, agent_id)
            else:
                agent = Human(agent_name, properties, agent_id)

            self.gt_graph.add_node(agent)

            # Add agent to the conversion dict
            self.sim_handle_to_name[agent_name] = agent_name

            # Fetch room for this agent
            room_name = None
            for region in self.sim.semantic_scene.regions:
                if region.contains(agent.properties["translation"]):
                    room_name = self.region_id_to_name[region.id]
                    break

            # Add agent to unknown room if a valid room is not found
            if room_name == None:
                self.gt_graph.add_edge(
                    agent, "unknown_room", "inside", flip_edge("inside")
                )
            else:
                # Add edge between the agent and room
                self.gt_graph.add_edge(agent, room_name, "inside", flip_edge("inside"))

    def update_agent_room_associations(self) -> None:
        """
        This method will update the associations between agents and rooms.
        This is required because we need to update the graph every time
        the agents move in environment
        """

        # Add agents to the graph
        for agent_name in self.sim.agents_mgr.agent_names:
            # Get agent id from name
            try:
                agent_id = int(agent_name.split("_")[1])
            except ValueError:
                agent_id = 0

            # Get articulated agent
            if isinstance(self.sim.agents_mgr, list):
                articulated_agent = self.sim.agents_mgr[agent_id].articulated_agent
            else:
                articulated_agent = self.sim.agents_mgr._all_agent_data[
                    agent_id
                ].articulated_agent

            # Get agent position
            current_pos = list(articulated_agent.base_pos)

            # Update the translation of agent node in the graph
            agent_node = self.gt_graph.get_node_from_name(agent_name)
            agent_node.properties["translation"] = current_pos

            # Get old room of the agent
            old_rooms = self.gt_graph.get_neighbors_of_type(agent_node, Room)

            # Make sure that its only one neighbor
            if len(old_rooms) != 1:
                raise ValueError(
                    f"agent with name {agent_node.name} was found to have more or less than one Rooms connected."
                )

            # Fetch new room for this agent
            new_room = None
            for region in self.sim.semantic_scene.regions:
                if region.contains(agent_node.properties["translation"]):
                    new_room = self.region_id_to_name[region.id]
                    break

            # It was found that sometimes, agent is not found to be in any room
            # In that case we skip changing its room
            if new_room != None:
                # Delete edge between old room and agent
                self.gt_graph.remove_edge(agent_node, old_rooms[0])

                # Add edge between the agent and room
                self.gt_graph.add_edge(agent_node, new_room, "inside", "contains")

    def update_object_receptacle_associations(self) -> None:
        """
        This method will update the associations between object and receptacles.
        This is required because we need to update the graph every time an object
        is moved from one receptacle to another.
        """
        object_node_list = self.gt_graph.get_all_objects()
        # Update positions of all objects
        for obj_node in object_node_list:
            # Update object position
            # obj_node = self.gt_graph.get_node_from_name(obj_name)
            translation = list(
                self.rom.get_object_by_handle(obj_node.sim_handle).translation
            )
            obj_node.properties["translation"] = translation

        # Get latest mapping from object to rec
        # NOTE: this call should strictly come after updating object positions
        # as it relies on positions as a mechanism for reducing computation overload
        # mixing the order here may lead to relationships dropping or being updated
        # later than expected.
        obj_to_rec = self.get_latest_objects_to_receptacle_map()

        for obj_name, rec_name in obj_to_rec.items():
            # Remove all old edges of this object
            self.gt_graph.remove_all_edges(obj_name)

            # Add new edge
            self.gt_graph.add_edge(obj_name, rec_name, "on", flip_edge("on"))

    def update_object_and_furniture_states(self) -> None:
        """
        Updates object states for all objects in the ground truth graph.
        self.sim.object_state_machine must already be initialized.
        """

        all_objects = self.gt_graph.get_all_nodes_of_type(Object)
        full_state_dict = self.sim.object_state_machine.get_snapshot_dict(self.sim)

        if all_objects is not None:
            for obj in all_objects:
                for state_name, object_state_values in full_state_dict.items():
                    if obj.sim_handle in object_state_values:
                        obj.set_state({state_name: object_state_values[obj.sim_handle]})

        all_furniture = self.gt_graph.get_all_nodes_of_type(Furniture)
        if all_furniture is not None:
            for fur in all_furniture:
                for state_name, object_state_values in full_state_dict.items():
                    if fur.sim_handle in object_state_values:
                        fur.set_state({state_name: object_state_values[fur.sim_handle]})

    def get_sim_handles_in_view(
        self,
        obs: Dict[str, np.ndarray],
        agent_uids: List[str],
        save_object_masks: bool = False,
    ) -> Dict[str, Set[str]]:
        """
        This method uses the instance segmentation output to create a list of handles of all objects present in given agent's FOV

        We need different sensor naming for different modes. We follow given schema:
        - agent_uids = ["0", "1"] to access obs from both agents in multi-agent setup
        - agent_uids = ["0"] to access robot obs in single/multi-agent setup
        - agent_uids = ["1"] to access human obs in multi-agent setup

        :param obs: Observation dict mapping sensor name to sensor output
        :param agent_uids: List of all agents to consider
        :param save_object_masks: Whether to save object-masks on-disk for debugging

        :return: Dict mapping agent-ID to all sim-handles in the agent's FoV
        """
        handles = {}

        for uid in agent_uids:
            if uid == "0":
                if "articulated_agent_arm_panoptic" in obs:
                    key = "articulated_agent_arm_panoptic"
                elif f"agent_{uid}_articulated_agent_arm_panoptic" in obs:
                    key = f"agent_{uid}_articulated_agent_arm_panoptic"
                else:
                    raise ValueError(
                        f"Could not find a valid panoptic sensor for agent uid: {uid}"
                    )
            elif uid == "1":
                key = f"agent_{uid}_head_panoptic"

            if key in obs:
                unique_obj_ids = np.unique(obs[key])
                if save_object_masks:
                    raise NotImplementedError
                unique_obj_ids = [
                    idx - 100 for idx in unique_obj_ids if idx != UNKNOWN_SEMANTIC_ID
                ]
                # we deduct 100 because at loading time 100 is added to all object
                # semantic IDs, reserving the first 100 for special entities
                sim_objects = [get_obj_from_id(self.sim, idx) for idx in unique_obj_ids]
                ro_handles = [obj.handle for obj in sim_objects if obj is not None]
                handles[uid] = set(ro_handles)
            else:
                raise ValueError(f"{key} not found in obs")

        return handles

    def get_recent_subgraph(
        self, agent_uids: List[str], obs: Dict[str, np.ndarray]
    ) -> Graph:
        """
        Method to return receptacle/agent-object associated detections from the sim
        This returns objects in view including objects held by the agent.

        :param obs: Observation dict mapping sensor name to sensor output
        :param agent_uids: List of all agents to consider

        :return: Latest world-graph over all objects seen by agent including the one currently in hold
        """

        # Make sure that sim is not None
        if not self.sim:
            raise ValueError("Trying to get detections from sim, but sim was None")

        # Make sure that the agents list is not empty or None
        if not agent_uids:
            raise ValueError(
                "Trying to get detections from sim, but agent_uids was empty"
            )

        # Update ground truth graph to reflect most
        # recent associations between objects, their states and their
        # receptacles based on the sim info
        self.update_object_receptacle_associations()
        self.update_agent_room_associations()
        self.update_object_and_furniture_states()

        # Get handles of all objects and receptacles in agent's FOVs
        handles_per_agent = self.get_sim_handles_in_view(obs, agent_uids)

        # Unpack handles from all agents and and make union
        handle_set = set.union(*handles_per_agent.values())

        # Convert handles to names
        names = []
        for handle in handle_set:
            if handle in self.sim_handle_to_name:
                names.append(self.sim_handle_to_name[handle])

        # Forcefully add robot and human node names
        agent_names = [f"agent_{uid}" for uid in agent_uids]
        names.extend(agent_names)

        # add held objects to the subgraph because they may not be seen
        # by the observations
        for uid in agent_uids:
            grasp_mgr = self.sim.agents_mgr[int(uid)].grasp_mgr
            if grasp_mgr.is_grasped:
                held_obj_id = grasp_mgr.snap_idx
                held_obj = get_obj_from_id(self.sim, held_obj_id)
                name = self.sim_handle_to_name[held_obj.handle]
                names.append(name)

        # Get subgraph with for the objects in view
        subgraph = self.gt_graph.get_subgraph(names)

        return copy.deepcopy(subgraph)

    def get_recent_graph(self) -> WorldGraph:
        """
        Method to return most recent ground truth graph.

        :return: Complete graph describing the latest state of the episode
        """

        # Update ground truth graph to reflect most
        # recent associations between objects, their states and their
        # receptacles based on the sim info
        self.update_object_receptacle_associations()
        self.update_agent_room_associations()
        self.update_object_and_furniture_states()

        return copy.deepcopy(self.gt_graph)

    def get_graph_without_objects(self) -> WorldGraph:
        """
        Method to return ground truth graph without any objects nodes.
        This method is only called during initializing world graph.
        """

        # Make copy of the graph
        graph_without_objects = copy.deepcopy(self.gt_graph)

        # Delete all notes of type object
        graph_without_objects.remove_all_nodes_of_type(Object)

        return graph_without_objects

    def initialize(self, partial_obs: bool = False) -> Graph:
        """
        Method to return detections from sim for initializing the world.
        When partial observability of on, this method returns all receptacles
        in the world without the corresponding objects. When partial observability
        is off it returns the entire world

        :param partial_obs: Whether perception should operate under complete or partial
        observability assumptions

        :return: Graph describing initial layout of the scene in partial-observability
        condition or entire state of the world, i.e. including objects, under complete
        observability condition
        """
        if partial_obs:
            return self.get_graph_without_objects()
        else:
            return self.get_recent_graph()

    def _get_current_receptacle_name(self, object_handle: str) -> Optional[str]:
        """
        Get the name of the current receptacle of an object.

        This function checks if the object is on the floor. If it is, it returns the name of the floor node.
        If it's not, it defaults to the receptacle searching logic

        :param object_handle: The handle of the object.

        :return: The name of the current receptacle of the object.
        """
        rec_handle = self._get_receptacle_handle_from_object_handle(object_handle)
        if rec_handle is None:
            # no match was found geometrically, possibly an invalid state
            return rec_handle
        elif "floor" in rec_handle:
            # return the floor node
            room_name = self.get_room_name(object_handle)
            floor_node = self.gt_graph.get_node_from_name(f"floor_{room_name}")
            return floor_node.name
        elif rec_handle in self.sim_handle_to_name:
            # this is the standard return for a matched Receptacle
            return self.sim_handle_to_name[rec_handle]
        else:
            # this object does not match a Receptacle or floor, check that it corresponds to a room
            # if it does then add object to the floor of that room
            # NOTE: objects need to have a furniture associated with them and Floor inherits from Furniture
            if rec_handle in self.region_id_to_name:
                room_name = self.region_id_to_name[rec_handle]
                floor_node = self.gt_graph.get_node_from_name(f"floor_{room_name}")
                return floor_node.name
            else:
                logger.error(
                    f"The object is not mapped to registered Receptacle or Region. Geometric match is {rec_handle} not the expected Region, {room_name}."
                )
        # no reasonable matches were found, something is wrong
        return None

    def _get_receptacle_handle_from_object_handle(self, object_handle: str) -> str:
        """
        Returns the unique_name of the best HabReceptacle match for the given object.
        """

        # get the ManagedObject
        obj = sutils.get_obj_from_handle(self.sim, object_handle)

        # Check if the object is with any of the agents
        obj_id = obj.object_id
        num_agents = self.sim.num_articulated_agents
        for agent_id in range(num_agents):
            grasp_mgr = self.sim.agents_mgr[agent_id].grasp_mgr
            if grasp_mgr.is_grasped and grasp_mgr.snap_idx == obj_id:
                return f"agent_{agent_id}"

        # match the object to Receptacles
        recs, _confidence, info_string = sutils.get_obj_receptacle_and_confidence(
            self.sim, obj, self.receptacles, island_index=self.sim.largest_island_idx
        )
        if len(recs) == 0:
            logger.error(
                f"Found no Receptacle match for object '{object_handle}'. Info string = {info_string}."
            )
            return None
        # return the best match
        return recs[0]
