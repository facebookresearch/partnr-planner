#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import copy
from typing import Dict, List

import habitat.sims.habitat_simulator.sim_utilities as sutils
import numpy as np
import pandas as pd
from habitat.core.logging import logger
from habitat.datasets.rearrange.samplers.receptacle import Receptacle as HabReceptacle
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_handle,
    get_obj_from_id,
    on_floor,
)
from magnum import Vector3

from habitat_llm.perception.perception import Perception
from habitat_llm.sims.metadata_interface import MetadataInterface
from habitat_llm.utils.sim import (
    find_receptacles,
    get_faucet_points,
    get_receptacle_dict,
)
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
from habitat_llm.world_model.world_graph import WorldGraph, flip_edge

HUMAN_SEMANTIC_ID = 100  # special semantic ID reserved for humanoids
UNKNOWN_SEMANTIC_ID = 0  # special semantic ID reserved for unknown object class


class PerceptionSim(Perception):
    """
    This class represents simulated perception stack of the agents.
    """

    # Parameterized Constructor
    def __init__(self, sim=None, metadata_dict: Dict[str, str] = None, detectors=None):
        # Call base class constructor
        super().__init__(detectors)

        # Load the metadata
        self.metadata_interface: MetadataInterface = None
        if metadata_dict is not None:
            self.metadata_interface = MetadataInterface(metadata_dict)

        if not sim:
            raise ValueError("Cannot construct PerceptionSim with sim as None")
        self.sim = sim

        # Container to cache list of receptacles
        self.receptacles: List[HabReceptacle] = None

        # Container to store fur to rec mapping
        self.fur_to_rec: Dict[str, Dict[str, List[HabReceptacle]]] = None

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
        self.add_rooms_to_gt_graph(sim)

        # Add all floors to the gt graph
        self.add_floors_to_gt_graph()

        # Add all receptacles to the gt graph
        self.add_furniture_and_receptacles_to_gt_graph(sim)

        # Add objects to the graph.
        self.add_objects_to_gt_graph(sim)

        # Add agents to the graph
        # This together with the above command completes the scene initialization.
        self.add_agents_to_gt_graph(sim)

        # Cache of receptacle names containing objects.
        self._obj_to_rec_cache: Dict[str, str] = {}

        # Cache of object positions.
        self._obj_position_cache: Dict[str, Vector3] = {}

        return

    @property
    def metadata(self) -> pd.DataFrame:
        """
        The subordinate MetadataInterface's loaded metadata DataFrame.
        """

        if self.metadata_interface is None:
            return None
        return self.metadata_interface.metadata

    def get_furniture_property_from_metadata(self, handle, prop):
        """
        This method returns value of the requested property using metadata file.
        For example, this could be used to extract the semantic type of any object
        in HSSD. Not that the property should exist in the metadata file.
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

    def get_room_name(self, handle: str):
        """
        Get the name of the room that contains a given object based off of the simulator object regions.

        Args:
            handle (str): The handle of the object.

        Returns:
            str: The name of the room that contains the object.

        Raises:
            ValueError: If the object is not in any region.
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

    def get_latest_objects_to_receptacle_map(self, sim) -> Dict[str, str]:
        """
        This method returns a dict which maps objects
        to their current receptacles in the sim
        """
        objects = self.gt_graph.get_all_objects()
        for obj in objects:
            obj_name = obj.name
            obj_pos = obj.properties["translation"]
            cached_pos = self._obj_position_cache.get(obj_name, None)
            if cached_pos is None or cached_pos != obj_pos:
                obj_handle = obj.sim_handle
                rec_name = self._get_current_receptacle_name(obj_handle)
                self._obj_position_cache[obj_name] = obj_pos
                self._obj_to_rec_cache[obj_name] = rec_name

        return self._obj_to_rec_cache

    def add_house_to_graph(self):
        """
        This method adds the root node house to the the gt_graph.
        """
        # Create root node
        house = House("house", {"type": "root"}, "house_0")
        self.gt_graph.add_node(house)

        return

    def add_rooms_to_gt_graph(self, sim):
        """
        This method adds room nodes to the gt_graph.
        This is done by querying in which room does a given furniture lie.
        """
        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to load rooms from sim, but sim was None")

        # Add room nodes to the graph
        region_names = {}
        if len(sim.semantic_scene.regions) == 0:
            raise ValueError(
                f"No regions found in the scene: {sim.ep_info['scene_id']}"
            )

        for region_idx, region in enumerate(sim.semantic_scene.regions):
            region_name = region.category.name().split("/")[0].replace(" ", "_")
            if region_name not in region_names:
                region_names[region_name] = 0
            region_names[region_name] = region_names[region_name] + 1
            room_name = f"{region_name}_{region_names[region_name]}"

            # Add a valid point on floor as room location
            point_on_floor = sutils.get_floor_point_in_region(sim, region_idx)

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

        return

    def add_floors_to_gt_graph(self):
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

    def add_furniture_and_receptacles_to_gt_graph(self, sim, verbose: bool = False):
        """
        Adds all furniture and corresponding receptacles to the graph during graph initialization
        """
        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to load furniture from sim, but sim was None")

        # Make sure that the metadata is not None
        if self.metadata is None:
            raise ValueError("Trying to load furniture from sim, but metadata was None")

        # Load rigid and articulated object managers
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()

        # Get faucet locations
        faucet_points = get_faucet_points(sim)

        # Get the list of receptacles from sim
        # This list is maintained as a state of this class because
        # its computationally expensive to generate (~0.3 sec) and
        # need to be used elsewhere in the code
        self.receptacles = find_receptacles(sim, filter_receptacles=True)

        # Get the furniture to receptacle dict
        self.fur_to_rec = get_receptacle_dict(
            sim, filter_receptacles=True, cached_receptacles=self.receptacles
        )

        # Iterate through furniture to rec dict and populate the graph
        for _, furniture_sim_handle in enumerate(self.fur_to_rec.keys()):
            # Mark if the furniture is articulated or not
            is_articulated = aom.get_library_has_handle(furniture_sim_handle)

            # Get furniture type using metadata
            furniture_type = self.get_furniture_property_from_metadata(
                furniture_sim_handle, "type"
            )

            # Generate name for furniture
            furniture_name = (
                f"{furniture_type}_{self.gt_graph.count_nodes_of_type(Furniture)}"
            )

            # Get furniture translation
            om = aom if is_articulated else rom

            translation = list(
                om.get_object_by_handle(furniture_sim_handle).translation
            )

            # Create properties dict
            properties = {
                "type": furniture_type,
                "is_articulated": is_articulated,
                "translation": translation,
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
            for proposition in self.fur_to_rec[furniture_sim_handle]:
                rec_list = self.fur_to_rec[furniture_sim_handle][proposition]
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

        return

    def add_objects_to_gt_graph(self, sim):
        """
        This method adds objects to the gt_graph during the graph initialization
        """

        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to load objects from sim, but sim was None")

        # Make sure that the metadata is not None
        if self.metadata is None:
            raise ValueError("Trying to load objects from sim, but metadata was None")

        # Add object nodes to the graph
        for obj_handle, fur_rec_handle in sim.ep_info.name_to_receptacle.items():
            sim_obj = sutils.get_obj_from_handle(sim, obj_handle)
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

        self.update_object_and_furniture_states(sim)
        return

    def add_agents_to_gt_graph(self, sim):
        """
        Method to add agents to the ground truth graph during initialization.
        """
        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to load agents from sim, but sim was None")

        # Add agents to the graph
        for agent_name in sim.agents_mgr.agent_names:
            # Get agent id from name
            try:
                agent_id = int(agent_name.split("_")[1])
            except ValueError:
                agent_id = 0

            # Get articulated agent
            if isinstance(sim.agents_mgr, list):
                articulated_agent = sim.agents_mgr[agent_id].articulated_agent
            else:
                articulated_agent = sim.agents_mgr._all_agent_data[
                    agent_id
                ].articulated_agent

            # Get agent position
            translation = list(articulated_agent.base_pos)

            # Create properties dict
            properties = {"translation": translation, "is_articulated": True}

            # Add Agent node to the world
            if agent_id == 0:
                agent = SpotRobot(agent_name, properties, agent_id)
            else:
                agent = Human(agent_name, properties, agent_id)

            self.gt_graph.add_node(agent)

            # Add agent to the conversion dict
            self.sim_handle_to_name[agent_name] = agent_name

            # Fetch room for this agent
            room_name = None
            for region in sim.semantic_scene.regions:
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

        return

    def update_agent_room_associations(self, sim):
        """
        This method will update the associations between agents and rooms.
        This is required because we need to update the graph every time
        the agents move in environment
        """

        # Add agents to the graph
        for agent_name in sim.agents_mgr.agent_names:
            # Get agent id from name
            try:
                agent_id = int(agent_name.split("_")[1])
            except ValueError:
                agent_id = 0

            # Get articulated agent
            if isinstance(sim.agents_mgr, list):
                articulated_agent = sim.agents_mgr[agent_id].articulated_agent
            else:
                articulated_agent = sim.agents_mgr._all_agent_data[
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
            for region in sim.semantic_scene.regions:
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

        return

    def update_object_receptacle_associations(self, sim):
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
        obj_to_rec = self.get_latest_objects_to_receptacle_map(sim)

        for obj_name, rec_name in obj_to_rec.items():
            # Remove all old edges of this object
            self.gt_graph.remove_all_edges(obj_name)

            # Add new edge
            self.gt_graph.add_edge(obj_name, rec_name, "on", flip_edge("on"))

        return

    def update_object_and_furniture_states(self, sim):
        """
        Updates object states for all objects in the ground truth graph.
        self.sim.object_state_machine must already be initialized.
        """

        all_objects = self.gt_graph.get_all_nodes_of_type(Object)
        full_state_dict = sim.object_state_machine.get_snapshot_dict(sim)

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

    def get_sim_handles_in_view(self, obs, agent_uids, save_object_masks: bool = False):
        """
        This method uses the instance segmentation output to
        create a list of handles of all objects present in given agent's FOV

        We need different sensor naming for different modes. We follow given schema:
        - agent_uids = ["0", "1"] to access obs from both agents in multi-agent setup
        - agent_uids = ["0"] to access robot obs in single/multi-agent setup
        - agent_uids = ["1"] to access human obs in multi-agent setup
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

    def get_recent_subgraph(self, sim, agent_uids, obs):
        """
        Method to return receptacle/agent-object associated detections from the sim
        This returns objects in view including objects held by the agent.
        """

        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to get detections from sim, but sim was None")

        # Make sure that the agents list is not empty or None
        if not agent_uids:
            raise ValueError(
                "Trying to get detections from sim, but agent_uids was empty"
            )

        # Update ground truth graph to reflect most
        # recent associations between objects, their states and their
        # receptacles based on the sim info
        self.update_object_receptacle_associations(sim)
        self.update_agent_room_associations(sim)
        self.update_object_and_furniture_states(sim)

        # Get handles of all objects and receptacles in agent's FOVs
        handles = self.get_sim_handles_in_view(obs, agent_uids)

        # Unpack handles from all agents and and make union
        handles = set.union(*handles.values())

        # Convert handles to names
        names = []
        for handle in handles:
            if handle in self.sim_handle_to_name:
                names.append(self.sim_handle_to_name[handle])

        # Forcefully add robot and human node names
        agent_names = [f"agent_{uid}" for uid in agent_uids]
        names.extend(agent_names)

        # add held objects to the subgraph because they may not be seen
        # by the observations
        for uid in agent_uids:
            grasp_mgr = sim.agents_mgr[int(uid)].grasp_mgr
            if grasp_mgr.is_grasped:
                held_obj_id = grasp_mgr.snap_idx
                held_obj = get_obj_from_id(sim, held_obj_id)
                name = self.sim_handle_to_name[held_obj.handle]
                names.append(name)

        # Get subgraph with for the objects in view
        subgraph = self.gt_graph.get_subgraph(names)

        return copy.deepcopy(subgraph)

    def get_recent_graph(self, sim):
        """
        Method to return most recent ground truth graph
        """

        # Make sure that sim is not None
        if not sim:
            raise ValueError("Trying to get all detections from sim, but sim was None")

        # Update ground truth graph to reflect most
        # recent associations between objects, their states and their
        # receptacles based on the sim info
        self.update_object_receptacle_associations(sim)
        self.update_agent_room_associations(sim)
        self.update_object_and_furniture_states(sim)

        return copy.deepcopy(self.gt_graph)

    def get_graph_without_objects(self):
        """
        Method to return ground truth graph without any objects nodes.
        This method is only called during initializing world graph.
        """

        # Make copy of the graph
        graph_without_objects = copy.deepcopy(self.gt_graph)

        # Delete all notes of type object
        graph_without_objects.remove_all_nodes_of_type(Object)

        return graph_without_objects

    def initialize(self, sim, partial_obs=False):
        """
        Method to return detections from sim for initializing the world.
        When partial observability of on, this method returns all receptacles
        in the world without the corresponding objects. When partial observability
        is off it returns the entire world
        """
        if partial_obs:
            return self.get_graph_without_objects()
        else:
            return self.get_recent_graph(sim)

    def _in_bounded_plane(self, point, bounds):
        """
        Returns True if the given point is within the bounded plane, otherwise False.

        Parameters:
        point (numpy.ndarray): The 3D position of the point as a numpy array.
        bounds (list): The list of 3D coordinates of the plane bounds as numpy arrays.

        Returns:
        in_bounds (bool): True if the point is within the bounded plane, otherwise False.
        """
        v1 = bounds[1] - bounds[0]  # First edge vector
        v2 = bounds[2] - bounds[0]  # Second edge vector
        v = point - bounds[0]

        # Compute the projected coordinates onto the edge vectors
        u1 = np.dot(v1, v) / np.dot(v1, v1)
        u2 = np.dot(v2, v) / np.dot(v2, v2)

        # Check if the point is within the bounded plane
        return 0 <= u1 <= 1 and 0 <= u2 <= 1

    def _point_to_plane_distance(self, point, plane):
        """
        Returns the distance between a point and a plane.

        Parameters:
        point (numpy.ndarray): The 3D position of the point as a numpy array.
        plane (list): A list of 3D coordinates of the plane's vertices as numpy arrays.

        Returns:
        distance (float): The shortest distance between the point and the plane.
        """
        # Compute the normal of the plane
        normal = np.cross(plane[1] - plane[0], plane[2] - plane[0])

        # Compute the distance
        return abs(np.dot(normal, point - plane[0]) / np.linalg.norm(normal))

    def _get_surface_bounds(self, surface, sim):
        """
        Returns the list of transformed corners of a surface.

        Parameters:
        surface (surface object): The surface object to get the corners from.
        sim (Simulator)

        Returns:
        corners (list): A list of 3D coordinates of the surface corners as numpy arrays.
        """
        corners = [
            surface.bounds.front_bottom_left,
            surface.bounds.front_bottom_right,
            surface.bounds.front_top_left,
            surface.bounds.front_top_right,
        ]
        return [
            surface.get_global_transform(sim).transform_point(corner)
            for corner in corners
        ]

    def _min_distance_to_bounded_plane(self, point, corners):
        """
        Returns the minimum distance from a point to a bounded plane defined by its corners.

        Parameters:
        point (numpy.ndarray): The 3D position of the point as a numpy array.
        corners (list): The list of 3D coordinates of the plane corners as numpy arrays.

        Returns:
        min_distance (float): The minimum distance from the point to the bounded plane.
        """
        # Select three non-collinear points
        plane_corners = [corners[0], corners[1], corners[3]]
        distance = self._point_to_plane_distance(point, plane_corners)

        min_distance = float("inf")
        for i in range(4):
            plane = np.vstack((corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]))
            # If the point is within the plane, return the distance
            if self._in_bounded_plane(point, plane):
                min_distance = min(min_distance, distance)
                break
            # Otherwise, compute the distance to the closest edge
            min_distance = min(
                min_distance,
                np.linalg.norm(point - corners[i]),
                np.linalg.norm(point - corners[(i + 1) % 4]),
            )

        return min_distance

    def _get_surface_plane_distance(self, obj_pos, surface, sim):
        """
        Returns the distance from the surface plane to an object's position.

        Parameters:
        obj_pos (numpy.ndarray): The 3D position of the object as a numpy array.
        surface (surface object): The surface object to find the distance.
        sim (Simulator)

        Returns:
        distance (float): The shortest distance between the object and the surface plane.
        """
        corners = self._get_surface_bounds(surface, sim)
        return self._min_distance_to_bounded_plane(obj_pos, corners)

    def _get_obj_position(self, sim, rom, handle):
        """
        Get updated position of objects from the sim.
        """
        scene_pos = sim.get_scene_pos()
        obj_id = rom.get_object_by_handle(handle).object_id
        index = sim.scene_obj_ids.index(obj_id)
        return scene_pos[index]

    def _get_current_receptacle_name(self, object_handle: str):
        """
        Get the name of the current receptacle of an object.

        This function checks if the object is on the floor. If it is, it returns the name of the floor node.
        If it's not, it defaults to the receptacle searching logic

        Args:
            object_handle (str): The handle of the object.

        Returns:
            str: The name of the current receptacle of the object.
        """
        obj = get_obj_from_handle(self.sim, object_handle)
        if on_floor(self.sim, obj, island_index=self.sim._largest_indoor_island_idx):
            # return the floor node
            room_name = self.get_room_name(object_handle)
            floor_node = self.gt_graph.get_node_from_name(f"floor_{room_name}")
            return floor_node.name
        else:
            rec_handle = self._get_receptacle_handle_from_object_handle(
                self.sim, object_handle, self.receptacles
            )
            return self.sim_handle_to_name[rec_handle]

    def _get_receptacle_handle_from_object_handle(
        self, sim, object_handle, surfaces_input
    ):
        """
        Returns the surface in which the object is located.
        """
        # Get object managers
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()

        # Check if the object is with any of the agents
        obj_id = rom.get_object_id_by_handle(object_handle)
        num_agents = sim.num_articulated_agents
        for agent_id in range(num_agents):
            grasp_mgr = sim.agents_mgr[agent_id].grasp_mgr
            if grasp_mgr.is_grasped:
                agent_grabbed_obj_id = grasp_mgr.snap_idx
                if agent_grabbed_obj_id == obj_id:
                    return f"agent_{agent_id}"

        # Get object position
        obj_pos = self._get_obj_position(sim, rom, object_handle)

        # Convert into dictionary
        surfaces = {surface.unique_name: surface for surface in surfaces_input}

        eps = 0
        done = False
        while not done:
            bounded_indices = []
            for surface_name, surface in surfaces.items():
                a, b = map(
                    surface.get_global_transform(sim).transform_point,
                    (surface.bounds.min, surface.bounds.max),
                )

                a, b = np.array(a), np.array(b)

                min_coords = np.array((a - eps, b - eps)).min(axis=0)
                max_coords = np.array((a + eps, b + eps)).max(axis=0)

                is_inside = np.all(min_coords <= obj_pos) and np.all(
                    obj_pos <= max_coords
                )
                if is_inside:
                    bounded_indices.append(surface_name)

                # We now do the same but close the receptacle

                # Get the receptacle handle and articulation type from the surface
                parent_rec_handle = surface.parent_object_handle
                is_articulated = surface.parent_link is not None
                link = surface.parent_link

                # Get receptacle
                om = aom if is_articulated else rom
                rec = om.get_object_by_handle(parent_rec_handle)

                # if "Articulated" in str(type(rec)):
                if is_articulated and rec.joint_positions:
                    joint_pos_index = rec.get_link_joint_pos_offset(link)
                    pose = rec.joint_positions
                    initial_pose = rec.joint_positions.copy()
                    pose[joint_pos_index] = 0.0
                    rec.joint_positions = pose

                    a, b = map(
                        surface.get_global_transform(sim).transform_point,
                        (surface.bounds.min, surface.bounds.max),
                    )
                    a, b = np.array(a), np.array(b)
                    min_coords = np.array((a - eps, b - eps)).min(axis=0)
                    max_coords = np.array((a + eps, b + eps)).max(axis=0)
                    is_inside = np.all(min_coords <= obj_pos) and np.all(
                        obj_pos <= max_coords
                    )
                    if is_inside:
                        bounded_indices.append(surface_name)

                    # reset the joint position
                    rec.joint_positions = initial_pose

            # If found one or more surfaces
            if bounded_indices:
                done = True
            else:
                eps += 0.1  # Increment eps to expand the search area

        # If the object is inside more than one surface
        bounded_indices = list(set(bounded_indices))
        if len(bounded_indices) > 1:
            distances = [
                (
                    surface_name,
                    self._get_surface_plane_distance(
                        obj_pos, surfaces[surface_name], sim
                    ),
                )
                for surface_name in bounded_indices
            ]

            # # Now we compute distances to the closed receptacle
            for surface_name in bounded_indices:
                # Get the receptacle handle and articulation type from the surface
                parent_rec_handle = surfaces[surface_name].parent_object_handle
                link = surfaces[surface_name].parent_link
                is_articulated = link is not None

                # Get receptacle
                om = aom if is_articulated else rom
                rec = om.get_object_by_handle(parent_rec_handle)

                # if "Articulated" in str(type(rec)):
                if is_articulated and rec.joint_positions:
                    joint_pos_index = rec.get_link_joint_pos_offset(link)
                    pose = rec.joint_positions
                    initial_pose = rec.joint_positions.copy()
                    pose[joint_pos_index] = 0.0
                    rec.joint_positions = pose

                    distance = self._get_surface_plane_distance(
                        obj_pos, surfaces[surface_name], sim
                    )
                    distances.append((surface_name, distance))

                    # reset the joint position
                    rec.joint_positions = initial_pose

            # Return the closest surface with minimum plane distance
            closest_surface = min(distances, key=lambda x: x[1])[0]

            return closest_surface
        else:  # If the object is inside a single surface
            return surfaces[bounded_indices[0]].unique_name
