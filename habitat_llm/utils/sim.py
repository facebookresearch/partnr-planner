#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import magnum as mn
import torch
from habitat.config.default_structured_configs import AgentConfig
from habitat.datasets.rearrange.navmesh_utils import snap_point_is_occluded
from habitat.datasets.rearrange.samplers.receptacle import Receptacle as HabReceptacle
from habitat.sims.habitat_simulator.sim_utilities import (
    get_ao_default_link,
    get_global_keypoints_from_object_id,
    get_obj_from_handle,
    get_obj_size_along,
    link_is_open,
)
from habitat.tasks.rearrange.articulated_agent_manager import ArticulatedAgentManager
from habitat.tasks.rearrange.rearrange_grasp_manager import RearrangeGraspManager

from habitat_llm.agent import Agent
from habitat_llm.world_model import Furniture, Object, Receptacle

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


def init_agents(
    agent_config: Dict[int, AgentConfig], env_interface: "EnvironmentInterface"
) -> List[Agent]:
    """
    Initialize a list of Agents from a dict mapping agent indices to their AgentConfigs.

    :param agent_config: A dict mapping agent indices to their AgentConfigs.
    :param env_interface: The EnvironmentInterface instance.
    :return: A list of initialized Agents.
    """
    agents = []
    for agent_conf in agent_config.values():
        # For readability
        agent_config = agent_conf.config

        # Instantiate the agent
        agent = Agent(agent_conf.uid, agent_conf.config, env_interface)

        # Make sure that its unique by adding to the set
        agents.append(agent)
    return agents


def get_receptacle_dict(
    sim: habitat_sim.Simulator,
    filter_receptacles: bool = True,
    cached_receptacles: List[HabReceptacle] = None,
) -> Dict[str, Dict[str, List[HabReceptacle]]]:
    """
    Get a dictionary from Furniture to (Hab)Receptacles, where the key is the furniture name (ManagedObject.handle) and the
    value is a dictionary two keys, "on" and "within" with the (Hab)Receptacles for each relationship.
    "Within" objects are those which can only be accessed by opening the "default_link" of the parent ArticulatedObject.

    :param filter_receptacles: If true, apply the rec_filter_file for the scene during (Hab)Receptacle parsing. Only accessible and valid receptacles
    (as annotated in the filter file) will be available if this option is used.
    :param cached_receptacles: optionally provide an existing list of (Hab)Receptacles. If not provided, they will be re-imported from disk (warning: this is wasteful, provide whenever possible).

    :return: A dict mapping parent ManagedObject instance handles to separate "on" and "within" subsets of (Hab)Receptacles.
    """

    rec_dict: Dict[str, Dict[str, List[HabReceptacle]]] = {}
    scene_filter_filepath = hab_receptacle.get_scene_rec_filter_filepath(
        sim.metadata_mediator, sim.curr_scene_name
    )
    if cached_receptacles is None:
        all_recs = find_receptacles(sim, filter_receptacles)
    else:
        all_recs = cached_receptacles
    within_recs = hab_receptacle.get_recs_from_filter_file(
        scene_filter_filepath, filter_types=["within_set"]
    )
    for rec in all_recs:
        parent_obj = rec.parent_object_handle
        rel_name = "within" if rec.unique_name in within_recs else "on"

        if parent_obj not in rec_dict:
            rec_dict[parent_obj] = {"on": [], "within": []}
        rec_dict[parent_obj][rel_name].append(rec)

    return rec_dict


def find_receptacles(
    sim: habitat_sim.Simulator, filter_receptacles: bool = True
) -> List[HabReceptacle]:
    """
    Find the receptacles in the current scene.
    Warning: this function parses configs and loads mesh data from disk. It should be called once and the results cached whenever possible.

    :param filter_receptacles: If true, apply the rec_filter_file for the scene during Receptacle parsing. Only accessible and valid receptacles
    (as annotated in the filter file) will be returned if this option is used.

    :return: The list of imported (Hab)Receptacles.
    """

    receptacles = None
    scene_filter_filepath = hab_receptacle.get_scene_rec_filter_filepath(
        sim.metadata_mediator, sim.curr_scene_name
    )
    if filter_receptacles:
        if scene_filter_filepath is not None:
            exclude_filter_strings = hab_receptacle.get_excluded_recs_from_filter_file(
                scene_filter_filepath
            )
            # only "active" receptacles from the filter are parsed
            receptacles = hab_receptacle.find_receptacles(
                sim,
                ignore_handles=None,  # this is a list of object handles for parent objects to exclude
                exclude_filter_strings=exclude_filter_strings,  # this is a list of unique_name substrings to exclude.
            )
        else:
            print(
                f"Warning: No receptacle filter file found for scene {sim.curr_scene_name}, no filtering will be done!"
            )

    if receptacles is None:
        # all receptacles are parsed
        receptacles = hab_receptacle.find_receptacles(sim)
    return receptacles


def get_receptacle_index(rec_unique_name: str, receptacles: List[HabReceptacle]) -> int:
    """
    Returns the numerical index of a given receptacle in the provided list of receptacles.

    :param rec_unique_name: The unique_name of the (Hab)Receptacle
    :param env: The environment

    :return: The index of the receptacle in the provided list of receptacles
    """

    # Make sure that the receptacle is valid
    rec_handles = [receptacle.unique_name for receptacle in receptacles]
    if not rec_unique_name in rec_handles:
        raise ValueError(
            f"Receptacle '{rec_unique_name}' is not available in the provided list."
        )

    return rec_handles.index(rec_unique_name)


def get_ao_and_joint_idx(
    fur: Furniture, env: "EnvironmentInterface"
) -> Tuple[habitat_sim.physics.ManagedArticulatedObject, List[int]]:
    """
    This method searches for the provided Furniture's ManagedArticulatedObject
    and its available joint indices based on the furniture handle.

    :param fur: A Furniture Entity which references a ManagedArticulatedObject.
    :param env: The EnvironmentInterface with access to the WorldGraph.

    :return: The parent ManagedArticulatedObject and a list of all link ids for the object.
    """

    # Fetch furniture from the sim
    aom = env.sim.get_articulated_object_manager()
    fur_sim = aom.get_object_by_handle(fur.sim_handle)

    if fur_sim is None:
        raise ValueError(
            f"Provided Furniture sim_handle '{fur.sim_handle}' does not reference a ManagedArticulatedObject."
        )

    # Get joint indices
    joint_idx = fur_sim.get_link_ids()

    return fur_sim, joint_idx


def get_parent_ao_and_joint_idx(
    rec: Receptacle,
    env: "EnvironmentInterface",
) -> Tuple[habitat_sim.physics.ManagedArticulatedObject, List[int]]:
    """
    This method searches for the provided llm-Receptacle's ManagedArticulatedObject parent
    and its available joint indices from the llm-Receptacle metadata.

    :param rec: The Receptacle Entity.
    :param env: The EnvironmentInterface with access to the WorldGraph.

    :return: The Receptacle's parent ManagedArticulatedObject and a list of link ids to which the Receptacle is connected.
    """

    # Get parent rec handle from world model
    p_fur_handle = env.full_world_graph.find_furniture_for_receptacle(rec).sim_handle

    # Fetch parent object from the sim
    aom = env.sim.get_articulated_object_manager()
    parent_fur = aom.get_object_by_handle(p_fur_handle)

    # get the HabReceptacle from the EnvironmentInterface receptacles cache
    # TODO: @zephirefaith receptacle cache needs to be agentic as well
    hab_rec = env.perception.receptacles[
        get_receptacle_index(rec.sim_handle, env.perception.receptacles)
    ]

    # Get joint dof index
    joint_idx = [parent_fur.get_link_joint_pos_offset(hab_rec.parent_link)]

    return parent_fur, joint_idx


def is_open(
    entity: Union[Furniture, Receptacle],
    env_interface: "EnvironmentInterface",
    threshold: float = 0.1,
) -> bool:
    """
    Checks whether or not a Receptacle Entity or parent Furniture Entity has an open "default_link" allowing access to "within" Receptacles.
    NOTE: does not check whether or not a provided Receptacle is part of the "within" or "on" subset, only checks the openness of the parent.
    TODO: when we move on from "default_link" to all links this will need to change

    :param entity: The Furniture or Receptacle Entity to check
    :param env_interface: The EnvironmentInterface
    :param threshold: The joint position delta from the "closed" state required to consider a link "open"

    :return: whether or not the ManagedArticulatedObject's "default_link" is heuristically "open"
    """

    # get the parent ManagedArticulatedObject to check
    parent_object = None

    # Check if the entity is open
    if isinstance(entity, Furniture):
        parent_object = (
            env_interface.sim.get_articulated_object_manager().get_object_by_handle(
                entity.sim_handle
            )
        )
    elif isinstance(entity, Receptacle):
        parent_object, _ = get_parent_ao_and_joint_idx(entity, env_interface)
    else:
        raise ValueError(
            "checking if Entity is open, but Entity is neither a Furniture nor a Receptacle."
        )

    if parent_object is None:
        raise ValueError(
            f"Could not find a parent ManagedArticulatedObject for Entity with name '{entity.name}'."
        )

    # now we have a ManagedArticulatedObject, check if the "default_link" is open

    # query or compute the default link
    default_link = get_ao_default_link(parent_object, compute_if_not_found=True)
    if default_link is None:
        # NOTE: no link to open, so always open. This should only happen for rare edge cases where an AO does not have open-able links.
        return True

    # check if the default link is open
    return link_is_open(parent_object, default_link, threshold=threshold)


def check_if_gripper_is_full(
    env: "EnvironmentInterface",
    action: torch.Tensor,
    grasp_mgr: RearrangeGraspManager,
    target_handle: str,
) -> Tuple[torch.Tensor, Optional[str], bool]:
    """
    A function to check if the provided RearrangeGraspManager is holding an object and terminate with a message if so.

    :param env: The EnvironmentInterface with a WorldGraph.
    :param action: The current action Tensor. Used only to shape the output zero action tensor if the check results in skill termination.
    :param grasp_mgr: The RearrangeGraspManager to check.
    :param target_handle: The ManagedObject instance handle for the target object. Used to construct the termination message referencing the given object or another.

    :return: A tuple containing a new action (original or zero), a termination message (if failed), and a boolean failure flag (failed=True when an object is held)
    """

    action_zero = torch.zeros(action.shape, device=action.device)

    if grasp_mgr.is_grasped:
        rom = env.sim.get_rigid_object_manager()
        grasped_obj_handle = rom.get_object_handle_by_id(grasp_mgr.snap_idx)
        target_node = env.full_world_graph.get_node_from_sim_handle(target_handle)
        grasped_node = env.full_world_graph.get_node_from_sim_handle(grasped_obj_handle)
        if target_node.name != grasped_node.name:
            termination_message = f"Failed to pick {target_node.name}! The arm is currently grasping {grasped_node.name}. Make the agent place the grasped object first."
        else:
            termination_message = f"The arm is already grasping {grasped_node.name}. Make the agent place the {grasped_node.name} either on the floor or at its target location using the place action."
        failed = True
        return action_zero, termination_message, failed
    else:
        return action, None, False


def check_if_the_object_is_moveable(
    env: "EnvironmentInterface", action: torch.Tensor, target_handle: str
) -> Tuple[torch.Tensor, Optional[str], bool]:
    """
    A function to check if the object is pick-able by agent skills.
    NOTE: termination message indicates pick failure specifically as that skill is the only consumer.

    :param env: The EnvironmentInterface with a WorldGraph.
    :param action: The current action Tensor. Used only to shape the output zero action tensor if the check results in skill termination.
    :param target_handle: The ManagedObject instance handle for the object in question.

    :return: A tuple containing a new action (original or zero), a termination message (if failed), and a boolean failure flag (failed=True when the handle is not an Entity::Object)
    """

    action_zero = torch.zeros(action.shape, device=action.device)

    # Early exit if the object is not a movable object.
    # FIXME: this needs to change for objects to be movable in CG version
    # we won't test based on "target_handle" anymore
    entity = env.full_world_graph.get_node_from_sim_handle(target_handle)
    if not isinstance(entity, Object):
        termination_message = "Failed to pick! This is not a movable object."
        failed = True
        return action_zero, termination_message, failed
    else:
        return action, None, False


def check_if_the_object_is_inside_furniture(
    env: "EnvironmentInterface",
    action: torch.Tensor,
    target_handle: str,
    threshold_for_ao_state: float,
) -> Tuple[torch.Tensor, Optional[str], bool]:
    """
    A function to check if the object is inside furniture in order to terminate
    NOTE: termination message indicates pick failure specifically as that skill is the only consumer.

    :param env: The EnvironmentInterface with a WorldGraph.
    :param action: The current action Tensor. Used only to shape the output zero action tensor if the check results in skill termination.
    :param target_handle: The ManagedObject instance handle for the object in question.
    :param threshold_for_ao_state: The joint position delta from the "closed" state required to consider a link "open"

    :return: A tuple containing a new action (original or zero), a termination message (if failed), and a boolean failure flag (failed=True when the object is inside a closed Furniture or has no Furniture)
    """

    action_zero = torch.zeros(action.shape, device=action.device)

    entity = env.full_world_graph.get_node_from_sim_handle(target_handle)

    # check for a furniture
    fur = env.full_world_graph.find_furniture_for_object(entity)
    if fur is None:
        # NOTE: This should only happen when not using ground_truth graph or when object is held
        termination_message = "Failed to pick! No Furniture found for the Object."
        failed = True
        print(f"None Furniture for Object: '{target_handle}'")
        return action_zero, termination_message, failed

    # check the Receptacle
    rec = env.full_world_graph.find_receptacle_for_object(entity)
    # currently any pick on articulated object requires open
    if (
        rec is not None
        and fur.is_articulated()
        and rec.sim_handle in env.perception.fur_to_rec[fur.sim_handle]["within"]
        and (not is_open(fur, env, threshold_for_ao_state))
    ):
        termination_message = "Failed to pick! Object is in a closed furniture, you need to open the furniture first."
        failed = True
        return action_zero, termination_message, failed
    else:
        return action, None, False


def check_if_the_object_is_held_by_agent(
    env, action: torch.Tensor, target_handle: str, this_agent_uid: int
) -> Tuple[torch.Tensor, Optional[str], bool]:
    """
    A function to check if an object is held by an agent.

    :param env: The Environment with a RearrangeSim ".sim"
    :param action: The action tensor (only used to shape a zero return tensor)
    :param target_handle: The ManagedObject handle of the target object.
    :param this_agent_uid: The integer unique id of the agent making the skill check. Used to construct the termination message clarifying whether the object is with the agent or another.

    :return: A tuple containing a new action (original or zero), a termination message (if failed), and a boolean failure flag (failed=True when the object is held)
    """
    action_zero = torch.zeros(action.shape, device=action.device)

    # Fetch the object id
    obj_idx = get_obj_from_handle(env.sim, target_handle).object_id
    env.world_graph[this_agent_uid].get_node_from_sim_handle(target_handle).name

    for agent_name in env.sim.agents_mgr.agent_names:
        try:
            agent_id = int(agent_name.split("_")[1])
        except ValueError:
            # this happens when agent_name is 'main_agent' in single-agent mode
            agent_id = 0

        agent_grasp_mgr = env.sim.agents_mgr[agent_id].grasp_mgr
        agent_is_grasping = agent_grasp_mgr.is_grasped
        grasped_obj_is_same = agent_grasp_mgr.snap_idx == obj_idx
        if agent_is_grasping and grasped_obj_is_same:
            termination_message = "Failed!"
            # check which agent is holding to construct the termination message
            if this_agent_uid == agent_id:
                termination_message = (
                    "Failed! This object is already held by this agent."
                )
            else:
                termination_message = "Failed! This object is held by another agent."

            failed = True
            return action_zero, termination_message, failed

    return action, None, False


def ee_distance_to_object(
    sim: habitat_sim.Simulator,
    articulated_agent_mgr: ArticulatedAgentManager,
    cur_agent_uid: int,
    object_handle: str,
    max_distance: float,
) -> Optional[float]:
    """
    This method approximates the distance from the agent's end effector to the nearest surface target object's bounding box.
    It also checks for occlusion between the agent base and the object, returning None if occluded.

    :param sim: The Simulator instance.
    :param articulated_agent: A Manipulator agent which has an end effector.
    :param object_handle: The ManagedObject instance handle for the object in question.
    :param max_distance: Maximum distance. Returns None if the distance exceeds this value.

    :return: float or None - The heuristic Euclidean distance from the agent's end effector to the target object boundary, or None if the object is occluded.
    """

    obj = get_obj_from_handle(sim, object_handle)
    obj_idx = obj.object_id
    cur_agent = articulated_agent_mgr[cur_agent_uid].articulated_agent
    cur_agent_ee_pos = cur_agent.ee_transform().translation
    agent_pos = cur_agent.base_pos
    # get the object_id for all links associated with all articulated agents so they can be ignored in navigation placement sampling
    # NOTE: with this code, occlusions can not be caused by the other agent being in the way. I.e. you can place, open/close, and change object states "through" the other agent.
    agent_object_ids = []
    for articulated_agent in articulated_agent_mgr.articulated_agents_iter:
        agent_object_ids.extend(
            [articulated_agent.sim_obj.object_id]
            + [*articulated_agent.sim_obj.link_object_ids.keys()]
        )
    target_object_ids = [obj_idx]
    if obj.is_articulated:
        target_object_ids.extend([*obj.link_object_ids.keys()])

    agent_height = 1.3
    # get the center of the object bounding box
    center = get_global_keypoints_from_object_id(sim, obj_idx)[0]
    # get the vector from the object to the agent end effector
    object_to_agent = cur_agent_ee_pos - center

    # if the object is held in the gripper this is the zero vector
    if object_to_agent.length() == 0:
        return 0

    # get object size along the vector
    size = get_obj_size_along(sim, obj_idx, object_to_agent)[0]

    # remove object size from the computed distance
    ee_distance = object_to_agent.length() - size

    # early exit if the object is further than max_distance
    if ee_distance > max_distance + max(obj.aabb.size()):
        return None

    # check for occlusion between the object and the agent
    is_occluded = snap_point_is_occluded(
        target=center,
        snap_point=agent_pos,
        height=agent_height,
        sim=sim,
        target_object_ids=target_object_ids,
        ignore_object_ids=agent_object_ids,
    )

    # load and check interaction surface points if available
    # NOTE: interaction surface points are pre-computed and cached points distributed on the object's surface which can be used for distance/occlusion checking to alleviate the brittle nature of single center point checks
    if obj.marker_sets.has_taskset("interaction_surface_points"):
        interaction_points = obj.marker_sets.get_task_link_markerset_points(
            "interaction_surface_points", "body", "primary"
        )
        global_points = obj.transform_local_pts_to_world(interaction_points, link_id=-1)
        p_to_ee_dists = [
            (point, (point - cur_agent_ee_pos).length()) for point in global_points
        ]
        # now a sorted list of points with closest first
        p_to_ee_dists.sort(key=lambda x: x[1])

        # check the interaction points first to see if a closer one exists
        for point, dist in p_to_ee_dists:
            # if the interaction point is closer or the center occlusion check failed, check occlusion for the interaction point
            if is_occluded or dist < ee_distance:
                point_occluded = snap_point_is_occluded(
                    target=point,
                    snap_point=agent_pos,
                    height=agent_height,
                    sim=sim,
                    target_object_ids=target_object_ids,
                    ignore_object_ids=agent_object_ids,
                )
                if not point_occluded:
                    return dist

    # last resort, return the center distance if possible
    if is_occluded:
        return None
    else:
        return ee_distance


def get_faucet_points(sim: habitat_sim.Simulator) -> Dict[str, List[mn.Vector3]]:
    """
    Load all Faucet MarkerSets in global space.

    :param sim: The Simulator instance.

    :return: A dictionary with keys for each object handle containing a faucet, and values as a list of faucet points.
    """
    objs = sutils.get_all_objects(sim)
    obj_markersets: Dict[str, List[mn.Vector3]] = {}
    for obj in objs:
        all_obj_marker_sets = obj.marker_sets
        if all_obj_marker_sets.has_taskset("faucets"):
            # this object has faucet annotations
            obj_markersets[obj.handle] = []
            faucet_marker_sets = all_obj_marker_sets.get_taskset_points("faucets")
            for link_name, link_faucet_markers in faucet_marker_sets.items():
                link_id = -1
                if link_name != "root":
                    link_id = obj.get_link_id_from_name(link_name)
                for _marker_subset_name, points in link_faucet_markers.items():
                    global_points = obj.transform_local_pts_to_world(points, link_id)
                    obj_markersets[obj.handle].extend(global_points)
    return obj_markersets
