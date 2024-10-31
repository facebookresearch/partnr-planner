#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
#
# ---
# This module implements PARTNR logic for non-privileged world-graph (i.e. maintains state without
# accessing any sim information) and partial-observability logic for privileged world-graph (mainly
# how to change state based on last-action and last-action's result from either agent). All  modules
# specific to non-privileged graph have non-privileged in the name or in the docstring.
# ---

import logging
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from habitat_llm.perception.perception_obs import PerceptionObs

import numpy as np
import torch

from habitat_llm.utils.geometric import (
    opengl_to_opencv,
    unproject_masked_depth_to_xyz_coordinates,
)
from habitat_llm.utils.semantic_constants import EPISODE_OBJECTS
from habitat_llm.world_model import (
    Entity,
    Floor,
    Furniture,
    House,
    Human,
    Object,
    Room,
    SpotRobot,
    UncategorizedEntity,
    WorldGraph,
)
from habitat_llm.world_model.world_graph import flip_edge


class DynamicWorldGraph(WorldGraph):
    """
    This derived class collects all methods specific to world-graph created and
    maintained based on observations instead of privileged sim data.
    """

    def __init__(
        self,
        max_neighbors_for_room_assignment: int = 5,
        num_closest_entities_for_entity_matching: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT = max_neighbors_for_room_assignment
        self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING = (
            num_closest_entities_for_entity_matching
        )
        self.include_objects = False
        self._sim_objects = EPISODE_OBJECTS
        self._entity_names: List[str] = []
        self._logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self._logger.setLevel(logging.DEBUG)
        self._sim_object_to_detected_object_map: dict = {}
        self._articulated_agents: dict = {}

    def _cg_object_to_object_uid(self, cg_object: dict) -> str:
        cg_object["object_tag"] = cg_object["object_tag"].lower()
        return f"{cg_object['id']}_{cg_object['object_tag'].replace(' ', '_').replace('/', '_or_').replace('-', '_')}"

    def _is_object(self, object_category: str, sim: bool = True) -> bool:
        if sim:
            return object_category in self._sim_objects
        return True

    def set_articulated_agents(self, articulated_agent: dict):
        self._articulated_agents = articulated_agent

    def create_cg_edges(
        self,
        cg_dict_list: Optional[dict] = None,
        include_objects: bool = False,
        verbose: bool = False,
    ):
        """
        This method populates the graph from the dict output of CG. Creates a graph to store
        different entities in the world and their relations to one another
        """
        self.include_objects = include_objects
        self._raw_cg = cg_dict_list

        def to_entity_input(obj: dict):
            translation = obj["bbox_center"]
            if obj.get("fix_bbox", False):
                translation = [translation[0], translation[2], translation[1]]
            bbox_min = np.array(
                np.array(translation) - np.array(obj["bbox_extent"])
            ).tolist()
            bbox_max = np.array(
                np.array(translation) + np.array(obj["bbox_extent"])
            ).tolist()
            return {
                "name": self._cg_object_to_object_uid(obj),
                "properties": {
                    "type": obj["category_tag"],
                    "translation": translation,
                    "bbox_extent": obj["bbox_extent"],
                    "bbox_max": bbox_max,
                    "bbox_min": bbox_min,
                },
            }

        def is_valid_obj_or_furniture(obj: dict, include_objects: bool):
            # check that object is valid and not a wall or floor
            tag_is_OK: bool = (
                obj["object_tag"] != "invalid"
                and "floor" not in obj["object_tag"]
                and "wall" not in obj["object_tag"]
            )
            # check that object is an object or furniture
            is_furniture: bool = tag_is_OK and obj["category_tag"] == "furniture"
            is_object: bool = tag_is_OK and obj["category_tag"] == "object"
            return is_furniture or (is_object and include_objects)

        # Create root node
        house = House("house", {"type": "root"}, "house_0")
        self.add_node(house)
        self._entity_names.append("house")

        if cg_dict_list is None or not cg_dict_list:
            raise ValueError("Need a list of CG edges to create the graph")

        for edge_candidate in cg_dict_list:
            object1 = edge_candidate["object1"]
            object2 = edge_candidate["object2"]
            edge_relation = edge_candidate["object_relation"].lower()
            if verbose:
                self._logger.info(f"RAW CG OUTPUT\n:{edge_candidate}")
            object_nodes: List[Entity] = []
            for obj in [object1, object2]:
                obj_uid = self._cg_object_to_object_uid(obj)
                if (
                    is_valid_obj_or_furniture(obj, include_objects)
                    and obj_uid not in self._entity_names
                ):
                    obj["object_tag"] = obj["object_tag"].lower()
                    obj["category_tag"] = obj["category_tag"].lower()
                    obj["room_region"] = obj["room_region"].lower()
                    obj_entity_input_dict = to_entity_input(obj)
                    if obj["category_tag"] == "object":
                        object_nodes.append(Object(**obj_entity_input_dict))
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    elif obj["category_tag"] == "furniture":
                        object_nodes.append(Furniture(**obj_entity_input_dict))
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    elif obj["category_tag"] == "invalid":
                        object_nodes.append(
                            UncategorizedEntity(**obj_entity_input_dict)
                        )
                        self.add_node(object_nodes[-1])
                        self._entity_names.append(obj_entity_input_dict["name"])
                    if verbose:
                        self._logger.info(f"Added new entity: {object_nodes[-1].name}")
                    # make a child of room_region allocated
                    room_region = obj["room_region"].replace(" ", "_")
                    room_node = None
                    try:
                        room_node = self.get_node_from_name(room_region)
                    except ValueError as e:
                        self._logger.info(e)
                    if room_node is None and room_region != "fail":
                        room_node = Room(
                            **{"properties": {"type": room_region}, "name": room_region}
                        )
                        self.add_node(room_node)
                        self._entity_names.append(room_region)
                        self.add_edge(
                            room_node,
                            house,
                            "inside",
                            opposite_label=flip_edge("inside"),
                        )
                        room_floor = Floor(f"floor_{room_node.name}", {})
                        self.add_node(room_floor)
                        self._entity_names.append(room_floor.name)
                        self.add_edge(
                            room_floor, room_node, "inside", flip_edge("inside")
                        )
                        if verbose:
                            self._logger.info(f"Added new room: {room_node.name}")
                    assert room_node is not None
                    self.add_edge(
                        obj_entity_input_dict["name"],
                        room_node.name,
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                    if verbose:
                        self._logger.info(
                            f"Added above object to room: {room_node.name}"
                        )
                elif obj_uid in self._entity_names:
                    object_nodes.append(self.get_node_from_name(obj_uid))
                    if verbose:
                        self._logger.info(
                            f"Found existing entity: {object_nodes[-1].name}"
                        )
            # add edge between object1 and object2
            if len(object_nodes) == 2:
                if edge_relation in ["none of these", "fail"]:
                    continue

                if "next to" in edge_relation:
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "next to",
                        opposite_label=flip_edge("next to"),
                    )
                elif edge_relation == "a on b":
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "on",
                        opposite_label=flip_edge("on"),
                    )
                elif edge_relation == "b on a":
                    self.add_edge(
                        object_nodes[1],
                        object_nodes[0],
                        "on",
                        opposite_label=flip_edge("on"),
                    )
                elif edge_relation == "a in b":
                    self.add_edge(
                        object_nodes[0],
                        object_nodes[1],
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                elif edge_relation == "b in a":
                    self.add_edge(
                        object_nodes[1],
                        object_nodes[0],
                        "inside",
                        opposite_label=flip_edge("inside"),
                    )
                else:
                    raise ValueError(
                        f"Unknown edge candidate: {edge_relation}, between objects: {object1} and {object2}"
                    )
                if verbose:
                    self._logger.info(
                        f"Added edge {edge_relation} b/w {object_nodes[0].name} and {object_nodes[1].name}"
                    )
        if verbose:
            self._logger.info("Before pruning")
            self.display_hierarchy()
        self._fix_furniture_without_assigned_room()
        self._clean_up_room_and_floor_locations()
        if verbose:
            self._logger.info("After pruning")
            self.display_hierarchy()

    def _fix_furniture_without_assigned_room(self):
        """
        Makes sure each furniture is assigned to some room; default=unknown
        """
        furnitures = self.get_all_furnitures()
        all_rooms = self.get_all_nodes_of_type(Room)
        default_room = None
        for room in all_rooms:
            if "unknown" in room.name:
                default_room = room
                break
        for fur in furnitures:
            room = self.get_neighbors_of_type(fur, Room)
            if len(room) == 0:
                self.add_edge(default_room, fur, "in", flip_edge("in"))
        fur_room_count = [
            1 if len(self.get_neighbors_of_type(fur, Room)) > 0 else 0
            for fur in furnitures
        ]
        assert sum(fur_room_count) == len(fur_room_count)

    def _clean_up_room_and_floor_locations(self):
        """
        Iterates over the now-filled graph and attaches positions of known furniture
        belonging to a room to the floor of that room as translation property.
        Also prunes out rooms without any object/furniture in it.

        We use furniture in a room to set the room's location, if a room does not have
        furniture we can't get this geometric information and hence we remove such rooms.
        """
        # find rooms without any furniture in it
        prune_list = []
        rooms = self.get_all_rooms()
        for current_room in rooms:
            furniture = self.get_neighbors_of_type(current_room, Furniture)
            random.shuffle(furniture)
            # remove rooms with just a floor edge or no edges to furniture
            if furniture is None:
                prune_list.append(current_room)
            elif len(furniture) == 1 and isinstance(furniture[0], Floor):
                if isinstance(furniture[0], Floor):
                    room_floor = furniture[0]
                    prune_list.append(room_floor)
                prune_list.append(current_room)
            else:
                # if a room has furniture then choose an arbitrary one which has
                # translation and set the location of the floor and the room
                # to be same as this furniture
                room_floor = [fnode for fnode in furniture if isinstance(fnode, Floor)][
                    0
                ]
                valid_translation = None
                for fur in furniture:
                    if "translation" in fur.properties:
                        valid_translation = fur.properties["translation"]
                        break
                room_floor.properties["translation"] = valid_translation
                current_room.properties["translation"] = valid_translation

        for prune_room in prune_list:
            self.remove_node(prune_room)

    def add_agent_node_and_update_room(self, agent_node: Union[Human, SpotRobot]):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Add agent-node to the graph and assign room-label based on proximity logic
        """
        self.add_node(agent_node)
        self._entity_names.append(agent_node.name)
        room_node = self.find_room_of_entity(agent_node)
        if room_node is None:
            raise ValueError(
                f"[DynamicWorldGraph.initialize_agent_nodes] No room found for {agent_node.name}"
            )
        self.add_edge(agent_node, room_node, "in", opposite_label="contains")

    def initialize_agent_nodes(self, subgraph: WorldGraph):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Initializes the agent nodes in the graph.
        """
        human_node = subgraph.get_all_nodes_of_type(Human)
        if len(human_node) == 0:
            self._logger.debug("No human node found")
        else:
            human_node = human_node[0]
            dynamic_human_node = Human(human_node.name, {"type": "agent"})
            dynamic_human_node.properties["translation"] = human_node.properties[
                "translation"
            ].copy()
            self.add_agent_node_and_update_room(dynamic_human_node)

        agent_node = subgraph.get_all_nodes_of_type(SpotRobot)
        if len(agent_node) == 0:
            self._logger.debug("No SpotRobot node found")
        else:
            agent_node = agent_node[0]
            dynamic_agent_node = SpotRobot(agent_node.name, {"type": "agent"})
            dynamic_agent_node.properties["translation"] = agent_node.properties[
                "translation"
            ].copy()
            self.add_agent_node_and_update_room(dynamic_agent_node)

    def _set_sim_handles_for_non_privileged_graph(self, perception: "PerceptionObs"):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Sets sim-handles for each non-privileged entity based on proximity matching to sim entities
        s.t. simulator skills can use these as arguments
        """
        # find closest entity to each entity in non-privileged graph and assign as proxy sim-handle
        all_gt_entities = perception.gt_graph.get_all_nodes_of_type(Furniture)
        # only keep furniture with placeable receptacle
        all_gt_entities = [
            ent for ent in all_gt_entities if ent.sim_handle in perception.fur_to_rec
        ]
        # only keep entities that have a translation property
        all_gt_entities = [
            ent for ent in all_gt_entities if "translation" in ent.properties
        ]
        all_gt_entity_positions = np.array(
            [np.array(entity.properties["translation"]) for entity in all_gt_entities]
        )
        non_privileged_graph_furniture = self.get_all_nodes_of_type(Furniture)
        if non_privileged_graph_furniture is not None:
            for current_fur in non_privileged_graph_furniture:
                # find the closest entity to given target
                entity_distance = np.linalg.norm(
                    all_gt_entity_positions
                    - np.array(current_fur.properties["translation"]),
                    axis=1,
                )
                closest_entity_idx = np.argmin(entity_distance)
                current_fur.sim_handle = all_gt_entities[closest_entity_idx].sim_handle
                self._sim_object_to_detected_object_map[
                    all_gt_entities[closest_entity_idx].name
                ] = current_fur
                self._logger.debug(
                    f"Matched {all_gt_entities[closest_entity_idx].name} with non-sim object {current_fur.name}"
                )
        # make sure each entity has a sim-handle except House and Room
        for entity in self.graph:
            if isinstance(entity, Furniture):
                assert entity.sim_handle is not None

    def find_room_of_entity(
        self, entity_node: Union[Human, SpotRobot], verbose: bool = False
    ) -> Optional[Room]:
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        This method finds the room node that the agent is in

        Logic: Find the objects closest to the agent and assign the agent to the room
        that contains the most number of these objects
        """
        room_node = None
        closest_objects = self.get_closest_entities(
            self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT,
            object_node=entity_node,
            dist_threshold=-1.0,
        )
        room_counts: Dict[Room, int] = {}
        for obj in closest_objects:
            for room in self.get_neighbors_of_type(obj, Room):
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} --> Closest object: {obj.name} is in room: {room.name}"
                    )
                if room in room_counts:
                    room_counts[room] += 1
                else:
                    room_counts[room] = 1
        if room_counts:
            if verbose:
                self._logger.info(f"{room_counts=}")
            room_node = max(room_counts, key=room_counts.get)
        return room_node

    def move_object_from_agent_to_placement_node(
        self,
        object_node: Union[Entity, Object],
        agent_node: Union[Entity, Human, SpotRobot],
        placement_node: Union[Entity, Furniture],
        verbose: bool = True,
    ):
        """
        Utility method to move object to a placement node from a given agent. Does in-place manipulation of the world-graph
        """
        # Detach the object from the agent
        self.remove_edge(object_node, agent_node)

        # Add new edge from object to the receptacle
        # TODO: We should add edge to default receptacle instead of fur
        self.add_edge(object_node, placement_node, "on", flip_edge("on"))
        # snap the object to furniture's center in absence of actual location
        object_node.properties["translation"] = placement_node.properties["translation"]
        if verbose:
            self._logger.info(
                f"Moved {object_node.name} from {agent_node.name} to {placement_node.name}"
            )

    def _non_privileged_graph_check_if_object_is_redundant(
        self,
        new_object_node: Object,
        closest_objects: List[Union[Object, Furniture, Room]],
        merge_threshold: float = 0.25,
        verbose: bool = False,
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING; ONLY FOR SIM
        Check if this object already exists in the world-graph we do this by
        calculating the euclidean distance between the new object and all
        existing objects in the graph

        :param new_object_node: the Object node created based on observations
        :param closest_objects: list of Object nodes closest to newly found object
        :param merge_threshold: if dist(existing_object, new_object) < dist-threshold and
                type of existing_object is same as type of new_object, new-object-node is discarded as a duplicate
        :param verbose: boolean to toggle verbosity
        """
        redundant_object = False
        matching_object = None
        for wg_object in closest_objects:
            euclid_dist = np.linalg.norm(
                np.array(new_object_node.properties["translation"])
                - np.array(wg_object.properties["translation"])
            )
            if verbose:
                print(f"{euclid_dist=} b/w {new_object_node.name} and {wg_object.name}")
                print(
                    f"{new_object_node.name} at {new_object_node.properties['translation']}"
                )
                print(f"{wg_object.name} at {wg_object.properties['translation']}")
                print(
                    f"{wg_object.properties['type']=}; {new_object_node.properties['type']=}"
                )
            if (
                euclid_dist < merge_threshold
                and wg_object.properties["type"] == new_object_node.properties["type"]
            ):
                if verbose:
                    print("SAME OBJECT")
                redundant_object = True
                matching_object = wg_object
                break
        return redundant_object, matching_object

    def _non_privileged_graph_check_if_object_is_held(
        self, new_object_node: Object, verbose: bool = True
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        Check if the object is being held by an agent
        :param new_object_node: the Object node created based on observations
        :param verbose: boolean to toggle verbosity
        """
        agent_nodes = self.get_agents()
        dist_threshold = 0.25
        articulated_agent = None
        for a_node in agent_nodes:
            articulated_agent = self._articulated_agents[
                int(a_node.name.split("_")[1])
            ]  # int(['agent', '1'][1]); assumes agent names are agent_0 and agent_1
            ee_pos = np.array(articulated_agent.ee_transform().translation)
            is_close_to_agent_ee = (
                np.linalg.norm(
                    ee_pos - np.array(new_object_node.properties["translation"])
                )
                < dist_threshold
            )
            if is_close_to_agent_ee:
                if verbose:
                    self._logger.debug(
                        f"NEWLY DETECTED OBJECT, {new_object_node.name}, IS BEING HELD by {a_node.name}"
                    )
                matching_node = a_node.properties.get("last_held_object", None)
                return True, matching_node
        return False, None

    def update_held_objects(self, agent_node: Union[Human, SpotRobot]):
        """
        If any agent is holding an object, update the object's location to be the agent's

        :param agent_node: Agent node of type Human or SpotRobot holding data associated with pick/place
        """
        held_entity_node = agent_node.properties.get("last_held_object", None)
        if held_entity_node is not None:
            held_entity_node = self.get_node_from_name(held_entity_node.name)
            if held_entity_node.properties.get("time_of_update", None) is not None and (
                time.time() - held_entity_node.properties["time_of_update"] > 1.0
            ):
                held_entity_node.properties["translation"][0] = agent_node.properties[
                    "translation"
                ][0]
                held_entity_node.properties["translation"][2] = agent_node.properties[
                    "translation"
                ][2]
                held_entity_node.properties["time_of_update"] = time.time()
        return

    def update_agent_locations(self, detector_frames: Dict[int, Dict[str, Any]]):
        """
        Use camera pose to update agent locations in the world-graph
        """
        agent_nodes = self.get_agents()
        for uid, detector_frame in detector_frames.items():
            agent_node = [
                a_node for a_node in agent_nodes if a_node.name == f"agent_{uid}"
            ][0]
            agent_node.properties["translation"] = detector_frame["camera_pose"][
                :3, 3
            ].tolist()
            prev_room: Room = self.get_neighbors_of_type(agent_node, Room)[0]
            self.remove_edge(agent_node, prev_room)
            room_node: Optional[Room] = self.find_room_of_entity(agent_node)
            if room_node is None:
                all_rooms = self.get_all_rooms()
                random.shuffle(all_rooms)
                room_node = all_rooms[-1]
            self.add_edge(agent_node, room_node, "in", opposite_label="contains")
            self.update_held_objects(agent_node)

    def get_object_from_obs(
        self,
        detector_frame: dict,
        object_id: int,
        uid: int,
        verbose: bool = False,
        object_state_dict: Optional[dict] = None,
    ) -> Optional[Object]:
        """
        Given the processed observation, extract the object's centroid and convert to a
        node
        NOTE: We use Sim information to populate locations for all objects detected by
        Human. Needs to be refactored post bug-fix in KinematicHumanoid class
        @zephirefaith @xavipuig
        """
        obj_id_to_category_mapping = detector_frame["object_category_mapping"]
        obj_id_to_handle_mapping = detector_frame["object_handle_mapping"]
        object_mask = detector_frame["object_masks"][object_id]
        object_handle = obj_id_to_handle_mapping[object_id]
        # NOTE: can add another area based check here to ignore very small objects from far away
        if not np.any(object_mask):
            return None
        if verbose:
            print(
                f"Found object: {obj_id_to_category_mapping[object_id]} with id: {object_id}, from agent: {uid}"
            )
        # TODO: remove after testing RGB-depth alignment from KinematicHumanoid class
        if uid == 1:
            # RGB+depth from human class was misaligned. Using location information sent from human as is
            # This will be updated to use RGB+depth like for agent_0 once that misalignment fix has been tested
            if "object_locations" in detector_frame:
                object_centroid = detector_frame["object_locations"][object_id]
            else:
                raise KeyError(
                    "[DynamicWorldGraph.get_object_from_obs] No object_locations found in detector_frame for human-detected objects"
                )
        else:
            depth_numpy = detector_frame["depth"]
            H, W, C = depth_numpy.shape
            pose = opengl_to_opencv(detector_frame["camera_pose"])
            depth_tensor = torch.from_numpy(depth_numpy.reshape(1, C, H, W))
            pose_tensor = torch.from_numpy(pose.reshape(1, 4, 4))
            inv_intrinsics_tensor = torch.from_numpy(
                np.linalg.inv(detector_frame["camera_intrinsics"]).reshape(1, 3, 3)
            )
            mask_tensor = torch.from_numpy(object_mask.reshape(1, C, H, W))
            mask_tensor = ~mask_tensor.bool()
            object_xyz = unproject_masked_depth_to_xyz_coordinates(
                depth_tensor,
                pose_tensor,
                inv_intrinsics_tensor,
                mask_tensor,
            )
            object_centroid = object_xyz.mean(dim=0).numpy().tolist()
        if verbose:
            print(f"{object_centroid=}")

        # add this object to the graph
        new_object_node = Object(
            f"{len(self._entity_names)+1}_{obj_id_to_category_mapping[object_id]}",
            {
                "type": obj_id_to_category_mapping[object_id],
                "translation": object_centroid,
                "camera_pose_of_view": detector_frame["camera_pose"],
            },
        )
        # store sim handle for this object; this information is only used to pass
        # to our skills when needed for kinematics simulation. Not used for any privileged perception tasks
        new_object_node.sim_handle = object_handle
        if object_state_dict is not None:
            for state_name, object_state_values in object_state_dict.items():
                if object_handle in object_state_values:
                    new_object_node.set_state(
                        {state_name: object_state_values[object_handle]}
                    )

        return new_object_node

    def _is_point_within_bbox(self, point, bbox_min, bbox_max):
        # Check if point is within all dimensions of the bounding box
        return np.all((point >= bbox_min) & (point <= bbox_max))

    def _is_point_on_bbox(self, point, bbox_min, bbox_max):
        # reshuffle coordinates to respect XYZ convention
        point = np.array(point)[[0, 2, 1]]
        bbox_min = np.array(bbox_min)[[0, 2, 1]]
        bbox_max = np.array(bbox_max)[[0, 2, 1]]
        # Check if point is within the XY extents and has a higher Z coordinate
        within_xy = np.all((point[:2] >= bbox_min[:2]) & (point[:2] <= bbox_max[:2]))
        higher_z = point[2] > bbox_max[2]
        return within_xy and higher_z

    def _cg_check_for_relation(self, object_node):
        """
        Uses geometric heuristics to check for containment or support relation b/w provided object and closest furniture
        """
        closest_furniture = self.get_closest_entities(
            5,
            object_node=object_node,
            include_furniture=True,
            include_objects=False,
            include_rooms=False,
        )
        for fur in closest_furniture:
            if not isinstance(fur, Floor):
                is_within = self._is_point_within_bbox(
                    object_node.properties["translation"],
                    fur.properties["bbox_min"],
                    fur.properties["bbox_max"],
                )
                if is_within:
                    return fur, "in"
                is_on = self._is_point_on_bbox(
                    object_node.properties["translation"],
                    fur.properties["bbox_min"],
                    fur.properties["bbox_max"],
                )
                if is_on:
                    return fur, "on"
        return None, None

    def update_non_privileged_graph_with_detected_objects(
        self,
        frame_desc: Dict[int, Dict[str, Any]],
        object_state_dict: dict = None,
        verbose: bool = False,
    ):
        """
        ONLY FOR NON-PRIVILEGED GRAPH SETTING
        This method updates the graph based on the processed observations
        """
        # finally update the agent locations based on camera pose
        self.update_agent_locations(frame_desc)

        # create masked point-clouds per object and then extract centroid
        # as a proxy for object's location
        # NOTE: using bboxes may also include non-object points to contribute
        # to the object's position...we can fix this with nano-SAM or using
        # analytical approaches to prune object PCD
        for uid, detector_frame in frame_desc.items():
            if detector_frame["object_category_mapping"]:
                obj_id_to_category_mapping = detector_frame["object_category_mapping"]
                detector_frame["object_handle_mapping"]  # for sensing states
                for object_id in detector_frame["object_masks"]:
                    if not self._is_object(obj_id_to_category_mapping[object_id]):
                        continue
                    new_object_node = self.get_object_from_obs(
                        detector_frame,
                        object_id,
                        uid,
                        verbose,
                        object_state_dict=object_state_dict,
                    )
                    if new_object_node is None:
                        continue
                    new_object_node.properties["time_of_update"] = time.time()

                    # add an edge to the closest room to this object
                    # get top N closest objects (N defined by self.max_neighbors_for_room_matching)
                    closest_objects = self.get_closest_entities(
                        self.MAX_NEIGHBORS_FOR_ROOM_ASSIGNMENT,
                        object_node=new_object_node,
                        include_furniture=False,
                        include_rooms=False,
                    )
                    merge_threshold = 0.25  # default threshold
                    if uid == 1:
                        merge_threshold = 0.5  # increase threshold for human as object is held higher up
                    (
                        redundant_object,
                        matching_object,
                    ) = self._non_privileged_graph_check_if_object_is_redundant(
                        new_object_node,
                        closest_objects,
                        merge_threshold=merge_threshold,
                        verbose=verbose,
                    )
                    (
                        held_object,
                        held_object_node,
                    ) = self._non_privileged_graph_check_if_object_is_held(
                        new_object_node, verbose=True
                    )
                    # only add this object is it is not being held by an agent
                    # or if it is not already in the world-graph
                    skip_adding_object = redundant_object | held_object

                    if skip_adding_object:
                        # update the matching object's translation and states
                        if matching_object is None and held_object_node is not None:
                            matching_object = held_object_node
                        if matching_object is not None:
                            matching_object.properties[
                                "translation"
                            ] = new_object_node.properties["translation"]
                            if "states" in new_object_node.properties:
                                matching_object.properties[
                                    "states"
                                ] = new_object_node.properties["states"]
                            # add current time to the object's properties
                            matching_object.properties[
                                "time_of_update"
                            ] = new_object_node.properties["time_of_update"]
                            # verify translation update worked
                            matching_node_from_wg = self.get_node_from_name(
                                matching_object.name
                            )
                            # TODO: logic for checking surface-placement over another furniture
                            assert (
                                matching_node_from_wg.properties["translation"]
                                == new_object_node.properties["translation"]
                            )
                        continue

                    self.add_node(new_object_node)
                    self._entity_names.append(new_object_node.name)
                    self._logger.info(f"Added new object to CG: {new_object_node}")
                    reference_furniture, relation = self._cg_check_for_relation(
                        new_object_node
                    )
                    if reference_furniture is not None and relation is not None:
                        self.add_edge(
                            reference_furniture,
                            new_object_node,
                            relation,
                            flip_edge(relation),
                        )
                    else:
                        # if not redundant and not belonging to a furniture
                        # then find the room this object should belong to
                        # find most common room among these objects
                        # TODO: get closest objects but only consider those visible to agent
                        room_counts: Dict[Union[Object, Furniture], int] = {}
                        for obj in closest_objects:
                            for room in self.get_neighbors_of_type(obj, Room):
                                if verbose:
                                    self._logger.info(
                                        f"Adding {new_object_node.name} --> Closest object: {obj.name} is in room: {room.name}"
                                    )
                                if room in room_counts:
                                    room_counts[room] += 1
                                else:
                                    room_counts[room] = 1
                                # only use the first Room neighbor, i.e. closest room node
                                break
                        if room_counts:
                            closest_room = max(room_counts, key=room_counts.get)
                            self.add_edge(
                                new_object_node,
                                closest_room,
                                "in",
                                opposite_label="contains",
                            )

    def update_by_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        action_response: str,
        verbose: bool = False,
    ):
        """
        Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
        Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
        based on agent_{agent_uid}'s actions.

        Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
        """
        if "success" in action_response.lower():
            self._logger.debug(
                f"{agent_uid=}: {high_level_action=}, {action_response=}"
            )
            agent_node = self.get_node_from_name(f"agent_{agent_uid}")
            if (
                "place" in high_level_action[0].lower()
                or "rearrange" in high_level_action[0].lower()
            ):
                # update object's new place to be the furniture
                if "place" in high_level_action[0].lower():
                    high_level_actions = high_level_action[1].split(",")
                    # remove the proposition
                    # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                    object_node = self.get_node_from_name(high_level_actions[0].strip())
                    # TODO: Add floor support
                    placement_node = self.get_node_from_name(
                        high_level_actions[2].strip()
                    )
                elif "rearrange" in high_level_action[0].lower():
                    # Split the comma separated pair into object name and receptacle name
                    try:
                        # Handle the case for rearrange proposition usage for place skills
                        high_level_actions = high_level_action[1].split(",")
                        # remove the proposition
                        # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                        high_level_actions = [
                            high_level_actions[0],
                            high_level_actions[2],
                        ]
                        object_node, placement_node = [
                            self.get_node_from_name(value.strip())
                            for value in high_level_actions
                        ]
                    except Exception as e:
                        self._logger.info(f"Issue when split comma: {e}")
                else:
                    raise ValueError(
                        f"Cannot update world graph with action {high_level_action}"
                    )

                # TODO: replace following with the right inside/on relation
                # based on 2nd string argument to Pick when implemented
                # TODO: Temp hack do not add something in placement_node if it is None
                if placement_node is not None:
                    self.move_object_from_agent_to_placement_node(
                        object_node, agent_node, placement_node
                    )
                    if verbose:
                        self._logger.info(
                            f"{self.update_by_action.__name__} Moved object: {object_node.name} from {agent_node.name} to {placement_node.name}"
                        )
                else:
                    if verbose:
                        self._logger.info(
                            f"{self.update_by_action.__name__} Could not move object from agent to placement-node: {high_level_action}"
                        )
            elif (
                "pour" in high_level_action[0].lower()
                or "fill" in high_level_action[0].lower()
            ):
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                entity_node.set_state({"is_filled": True})
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} is now filled, {entity_node.properties}"
                    )
            elif "power" in high_level_action[0].lower():
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                if "on" in high_level_action[0].lower():
                    entity_node.set_state({"is_powered_on": True})
                    if verbose:
                        self._logger.info(
                            f"{entity_node.name} is now powered on, {entity_node.properties}"
                        )
                elif "off" in high_level_action[0].lower():
                    entity_node.set_state({"is_powered_on": False})
                    if verbose:
                        self._logger.info(
                            f"{entity_node.name} is now powered off, {entity_node.properties}"
                        )
                else:
                    raise ValueError(
                        "Expected 'on' or 'off' in power action, got: ",
                        high_level_action[0],
                    )
            elif "clean" in high_level_action[0].lower():
                entity_name = high_level_action[1]
                entity_node = self.get_node_from_name(entity_name)
                entity_node.set_state({"is_clean": True})
                if verbose:
                    self._logger.info(
                        f"{entity_node.name} is now clean, {entity_node.properties}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action,
                    )
        return

    def update_non_privileged_graph_by_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        action_response: str,
        verbose: bool = False,
        drop_placed_object_flag: bool = True,
    ):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Deterministically updates the world-graph based on last-action taken by agent_{agent_uid} based on the result of that action.
        Only updates the graph if the action was successful. Applicable only when one wants to change agent_{agent_uid}'s graph
        based on agent_{agent_uid}'s actions. If drop_placed_object_flag is True then whenever an object is placed it is simply deleter from the graph instead of being read to the receptacle.
        This method is different from update_by_action as it expects non-privileged entities as input and not GT sim entities.

        Please look at update_by_other_agent_action or update_non_privileged_graph_by_other_agent_action for updating self graph based on another agent's actions.
        """
        if (
            isinstance(action_response, str)
            and isinstance(high_level_action[0], str)
            and "success" in action_response.lower()
        ):
            self._logger.debug(
                f"{agent_uid=}: {high_level_action=}, {action_response=}"
            )
            agent_node = self.get_node_from_name(f"agent_{agent_uid}")
            if (
                "place" in high_level_action[0].lower()
                or "rearrange" in high_level_action[0].lower()
            ):
                placement_node = None
                object_node = None
                # Split the comma separated pair into object name and receptacle name
                try:
                    # Handle the case for rearrange proposition usage for place skills
                    high_level_args = high_level_action[1].split(",")
                    # remove the proposition
                    # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                    high_level_args = [
                        high_level_args[0],
                        high_level_args[2],
                    ]
                    object_node, placement_node = [
                        self.get_node_from_name(value.strip())
                        for value in high_level_args
                    ]
                except Exception as e:
                    self._logger.info(f"Issue when split comma: {e}")

                if object_node is not None:
                    if drop_placed_object_flag:
                        self.remove_node(object_node)
                        self._entity_names.remove(object_node.name)
                        if "last_held_object" in agent_node.properties:
                            del agent_node.properties["last_held_object"]
                        self._logger.debug("Object deleted once robot placed it")
                    elif placement_node is not None:
                        self.move_object_from_agent_to_placement_node(
                            object_node, agent_node, placement_node
                        )
                        if verbose:
                            self._logger.info(
                                f"Moved object: {object_node.name} from {agent_node.name} to {placement_node.name}"
                            )
                else:
                    if verbose:
                        self._logger.info(
                            f"Could not move object from agent to placement-node: {high_level_action}"
                        )
            elif "pick" in high_level_action[0].lower():
                object_name = high_level_action[1]
                try:
                    obj_node = self.get_node_from_name(object_name)
                    # remove all current edges from this node
                    obj_neighbors = self.get_neighbors(obj_node).copy()
                    edges_to_remove = []
                    for neighbor in obj_neighbors:
                        edges_to_remove.append((obj_node, neighbor))
                    for edge in edges_to_remove:
                        self.remove_edge(*edge)
                    # add edge b/w obj and the agent
                    self.add_edge(
                        obj_node, agent_node, "on", opposite_label=flip_edge("on")
                    )
                    agent_node.properties["last_held_object"] = obj_node
                    self._logger.debug(
                        f"[{self.update_non_privileged_graph_by_action.__name__}] {agent_node.name} PICKED OBJECT {obj_node.name}"
                    )
                except KeyError as e:
                    self._logger.info(
                        f"Could not find matching receptacle in agent\nException: {e}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action,
                    )
        return

    def _cg_find_self_entity_match_to_human_entity(
        self,
        human_entity_name: str,
        human_agent_node: Human,
        is_furniture: bool = False,
    ) -> Optional[Union[Object, Furniture, Room]]:
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Reusable function for finding matching objects to human-held object or furniture to human-held furniture
        """
        human_entity_type = None
        match = re.match(
            r"^(.*)_\d+$",
            human_entity_name,
        )
        if match:
            human_entity_type = match.group(1)
        # now find the object of above type closest to last known human location
        dist_threshold = 0.0
        include_objects = True
        include_furniture = True
        if is_furniture:
            include_objects = False
        else:
            dist_threshold = 2.25  # actuation distance is 2.0 for oracle-skills; adding 0.25 for noise handling
            include_furniture = False
        closest_objects = self.get_closest_entities(
            self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING,
            object_node=human_agent_node,
            include_objects=include_objects,
            include_furniture=include_furniture,
            include_rooms=False,
            dist_threshold=dist_threshold,
        )
        if not closest_objects and dist_threshold > 0.0 and is_furniture:
            closest_objects = self.get_closest_entities(
                self.NUM_CLOSEST_ENTITIES_FOR_ENTITY_MATCHING,
                object_node=human_agent_node,
                include_objects=include_objects,
                include_furniture=include_furniture,
                include_rooms=False,
                dist_threshold=-1.0,
            )
        most_likely_matching_object = [
            ent
            for ent in closest_objects
            if ent.properties["type"] == human_entity_type
        ]
        if len(most_likely_matching_object) > 0:
            self._logger.debug(
                f"Mapped node {most_likely_matching_object[0].name} in robot's WG to Human held object: {human_entity_name}; based on both proximity and type"
            )
            return most_likely_matching_object[0]
        if len(closest_objects) > 0:
            most_likely_matching_object_node = closest_objects[0]
            self._logger.debug(
                f"Mapped human's node {human_entity_name} to {most_likely_matching_object_node} based on proximity"
            )
        else:
            most_likely_matching_object_node = None
            self._logger.debug(
                f"Mapped human's node {human_entity_name} to {most_likely_matching_object_node} based on default"
            )
        return most_likely_matching_object_node

    def update_non_privileged_graph_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        verbose: bool = False,
        drop_placed_object_flag: bool = True,
    ):
        """
        ONLY FOR USE WITH NON-PRIVILEGED GRAPH

        Deterministically change self graph based on successful execution of a given action by another agent. The arguments to action
        are based on other agent's identifiers so this method implements essential logic for mapping them back to most likely match in
        self graph, e.g. what the Human agent calls 161_chest_of_drawers may be 11_chest_of_drawers for Spot.
        """
        if "success" in action_results.lower():
            if verbose:
                self._logger.debug(
                    f"{self.update_non_privileged_graph_by_other_agent_action.__name__}{other_agent_uid=}: {high_level_action_and_args=}, {action_results=}"
                )
            agent_node = self.get_human()
            if "pick" in high_level_action_and_args[0].lower():
                # find the matching node and add edge to the other agent's node
                # breakdown what human is holding to its type
                human_picked_object_name = high_level_action_and_args[1].strip()
                most_likely_held_object = (
                    self._cg_find_self_entity_match_to_human_entity(
                        human_picked_object_name, agent_node
                    )
                )
                if most_likely_held_object is not None:
                    object_prev_neighbors = self.get_neighbors(most_likely_held_object)
                    edges_to_remove = []
                    for neighbor in object_prev_neighbors:
                        edges_to_remove.append((most_likely_held_object, neighbor))
                    for edge in edges_to_remove:
                        self.remove_edge(*edge)
                    self.add_edge(
                        most_likely_held_object, agent_node, "on", flip_edge("on")
                    )
                    most_likely_held_object.properties[
                        "translation"
                    ] = agent_node.properties["translation"]
                    # also update last_held_object property
                    agent_node.properties["last_held_object"] = most_likely_held_object
                    self._logger.debug(
                        f"CG updated per human picking {most_likely_held_object.name}"
                    )
                else:
                    self._logger.info(
                        f"Could not find any matching object in robot's WG to {human_picked_object_name=}. Expect funky behavior."
                    )
            if (
                "place" in high_level_action_and_args[0].lower()
                or "rearrange" in high_level_action_and_args[0].lower()
            ):
                all_args = high_level_action_and_args[1].split(",")
                human_placement_furniture_name = all_args[2].strip()
                most_likely_held_object = agent_node.properties.get(
                    "last_held_object", None
                )
                if most_likely_held_object is not None:
                    most_likely_placement_node = (
                        self._sim_object_to_detected_object_map.get(
                            human_placement_furniture_name,
                            self._cg_find_self_entity_match_to_human_entity(
                                human_placement_furniture_name,
                                agent_node,
                                is_furniture=True,
                            ),
                        )
                    )
                    if (
                        most_likely_held_object is not None
                        and most_likely_placement_node is not None
                        and not drop_placed_object_flag
                    ):
                        self.move_object_from_agent_to_placement_node(
                            most_likely_held_object,
                            agent_node,
                            most_likely_placement_node,
                            verbose=verbose,
                        )
                        del agent_node.properties["last_held_object"]
                        self._logger.debug("CG updated per Human place")
                    elif (
                        most_likely_held_object is not None and drop_placed_object_flag
                    ):
                        self.remove_node(most_likely_held_object)
                        self._entity_names.remove(most_likely_held_object.name)
                        del agent_node.properties["last_held_object"]
                        self._logger.debug(
                            "CG updated per Human place; we just removed the object"
                        )
                else:
                    self._logger.debug(
                        "Can't update CG based on human placement; CG did not register Pick"
                    )
            elif (
                "pour" in high_level_action_and_args[0].lower()
                or "fill" in high_level_action_and_args[0].lower()
            ):
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    object_node.set_state({"is_filled": True})
                    if verbose:
                        self._logger.debug(
                            f"{object_node.name} is now filled, {object_node.properties}"
                        )
            elif "power" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    if "on" in high_level_action_and_args[0].lower():
                        object_node.set_state({"is_powered_on": True})
                        if verbose:
                            self._logger.debug(
                                f"{object_node.name} is now powered on, {object_node.properties}"
                            )
                    elif "off" in high_level_action_and_args[0].lower():
                        object_node.set_state({"is_powered_on": False})
                        if verbose:
                            self._logger.debug(
                                f"{object_node.name} is now powered off, {object_node.properties}"
                            )
                    else:
                        raise ValueError(
                            "Expected 'on' or 'off' in power action, got: ",
                            high_level_action_and_args[0],
                        )
            elif "clean" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self._cg_find_self_entity_match_to_human_entity(
                    object_name, agent_node
                )
                if object_node is not None:
                    object_node.set_state({"is_clean": True})
                    if verbose:
                        self._logger.debug(
                            f"{object_node.name} is now clean, {object_node.properties}"
                        )

    def _update_gt_graph_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        verbose: bool = False,
    ):
        """
        Uses the exact object and receptacle names given by the other agent to update the
        graph. We assume that the other agent's identifiers exactly match self identifiers.
        """
        if "success" in action_results.lower():
            self._logger.debug(f"{high_level_action_and_args=} {other_agent_uid=}")
            agent_node = self.get_node_from_name(f"agent_{other_agent_uid}")
            # parse out the object-name and the closest furniture-name
            # if the object is not already in the graph, add it
            # if the placement furniture is not already in the graph, add it as a new
            # node
            if (
                "place" in high_level_action_and_args[0].lower()
                or "rearrange" in high_level_action_and_args[0].lower()
            ):
                # <spatial_relation>, <furniture/floor to be placed>, <spatial_constraint>, <reference_object>]
                # get the object from agent properties
                high_level_action_args = high_level_action_and_args[1].split(",")
                object_node = self.get_node_from_name(high_level_action_args[0].strip())
                try:
                    placement_node = self.get_node_from_name(
                        high_level_action_args[2].strip()
                    )
                    self.move_object_from_agent_to_placement_node(
                        object_node, agent_node, placement_node
                    )
                    self._logger.debug(
                        f"From the perspective of agent_{1-int(other_agent_uid)}:\n{agent_node.name} PLACED OBJECT {object_node.name} on {placement_node.name}"
                    )
                except KeyError as e:
                    self._logger.info(
                        f"Could not find matching receptacle in agent {1-int(other_agent_uid)} graph for {high_level_action_args[2].strip()} that agent {other_agent_uid} is trying to place on.\nException: {e}"
                    )
            elif (
                "pour" in high_level_action_and_args[0].lower()
                or "fill" in high_level_action_and_args[0].lower()
            ):
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                object_node.set_state({"is_filled": True})
                if verbose:
                    self._logger.info(
                        f"{object_node.name} is now filled, {object_node.properties}"
                    )
            elif "power" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                if "on" in high_level_action_and_args[0].lower():
                    object_node.set_state({"is_powered_on": True})
                    if verbose:
                        self._logger.info(
                            f"{object_node.name} is now powered on, {object_node.properties}"
                        )
                elif "off" in high_level_action_and_args[0].lower():
                    object_node.set_state({"is_powered_on": False})
                    if verbose:
                        self._logger.info(
                            f"{object_node.name} is now powered off, {object_node.properties}"
                        )
                else:
                    raise ValueError(
                        "Expected 'on' or 'off' in power action, got: ",
                        high_level_action_and_args[0],
                    )
            elif "clean" in high_level_action_and_args[0].lower():
                object_name = high_level_action_and_args[1]
                object_node = self.get_node_from_name(object_name)
                object_node.set_state({"is_clean": True})
                if verbose:
                    self._logger.info(
                        f"{object_node.name} is now clean, {object_node.properties}"
                    )
            else:
                if verbose:
                    self._logger.info(
                        "Not updating world graph for successful action: ",
                        high_level_action_and_args,
                    )
        return

    def update_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        use_semantic_similarity: bool = False,
        verbose: bool = False,
    ):
        if use_semantic_similarity:
            raise NotImplementedError(
                "Semantic similarity based WG update is not supported. Code currently supports closed-vocab naming of object and furniture"
            )
        self._update_gt_graph_by_other_agent_action(
            other_agent_uid,
            high_level_action_and_args,
            action_results,
            verbose=verbose,
        )
