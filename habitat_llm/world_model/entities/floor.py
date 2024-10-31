#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import habitat_sim
import magnum as mn
import numpy as np
from habitat.datasets.rearrange.navmesh_utils import unoccluded_navmesh_snap
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_handle,
    obj_next_to,
    snap_down,
)

from habitat_llm.world_model.entities.furniture import (
    Furniture,
    distance_to_other_samples,
    sort_proposed_samples_based_on_distance_to_agent,
)

if TYPE_CHECKING:
    from habitat.articulated_agents import ArticulatedAgentBase
    from habitat.tasks.rearrange.rearrange_grasp_manager import RearrangeGraspManager

    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model import Object


class Floor(Furniture):
    """
    This class represents floors
    """

    # Parameterized Constructor
    def __init__(self, name, properties):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle="floor")

    def sample_place_location(
        self,
        spatial_relation: str | None,
        spatial_constraint: str | None,
        reference_object: None | Object | Furniture,
        env: "EnvironmentInterface",
        agent: ArticulatedAgentBase,
        grasp_mgr: RearrangeGraspManager = None,
    ) -> List[Tuple[mn.Vector3, mn.Quaternion]]:
        """
        Compute valid placement locations on this furniture.
        :param spatial_relation: string representing spatial relation between object and floor.
        This is just here to match the superclass signature as the floor will always have the "on" spatial relation
        :param spatial_constraint: string representing spatial constraint between obj and reference obj. Only "next_to" is supported
        :param reference_object: node from world graph representing reference object
        :param env: an instance of environment interface
        :param agent: the articulated agent that will be placing the object
        :param grasp_mgr: the active grasp manager
        :return: A set of valid candidate placements (pos, orientation) sorted by distance to the agent.
        """
        cur_agent_ee_pos = agent.ee_transform().translation
        reference_handle = (
            reference_object.sim_handle if reference_object is not None else None
        )
        # Right now only "next_to" is supported
        if spatial_constraint is not None and spatial_constraint != "next_to":
            raise ValueError("spatial_constraint can only be 'next_to' or None")
        return sample_position_on_floor(
            env,
            agent=agent,
            proposition=spatial_constraint or "on",
            entity_pos=cur_agent_ee_pos,
            grasp_mgr=grasp_mgr,
            target_handle=reference_handle,
        )

    # Deep copy
    def __deepcopy__(self, memo):
        return Floor(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
        )


def get_floor_object_ids(
    sim: habitat_sim.Simulator,
    navmesh_point: mn.Vector3,
    ignore_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Get all object between the provided navmesh point and the stage floor which are not in the ignore list.

    :param sim: The Simulator instance.
    :param navmesh_point: The navmesh point below which to search for objects.
    :param ignore_ids: All object ids which should be ignored in the search for a support surface. Typically those belonging to an object or agent which may be standing or sitting at the search location.
    :return: A list of object ids for the floor. Always includes the stage_id.

    Uses a raycast to detect objects hit before the stage and then culls out the ignored ids.
    Example: when a floor rug is navigable but causes snap_down to fail, we add the rug as a support object id.
    Raises a ValueError if there is nothing below the provided point.
    """
    ray = habitat_sim.geo.Ray(navmesh_point, mn.Vector3(0, -1, 0))
    raycast_results = sim.cast_ray(ray)
    floor_object_ids = [habitat_sim.stage_id]
    if raycast_results.has_hits:
        for hit in raycast_results.hits:
            if hit.object_id == habitat_sim.stage_id:
                # stop once we hit the stage
                break
            floor_object_ids.append(hit.object_id)
    else:
        raise ValueError(
            f"Provided navmesh_point {navmesh_point} does not have anything below it. Is it actually a valid navmesh point?"
        )

    if ignore_ids is not None:
        # cull the ignores
        floor_object_ids = list(set(floor_object_ids) - set(ignore_ids))
    return floor_object_ids


def sample_position_on_floor(
    env_interface: "EnvironmentInterface",
    agent: ArticulatedAgentBase = None,
    proposition: str = "next_to",
    entity_pos: Union[np.ndarray, mn.Vector3] = None,
    grasp_mgr: RearrangeGraspManager = None,
    target_handle: str = None,
    is_articulated: bool = False,
    min_sample_distance: float = 0.10,
    sample_region_scale: float = 1,
    max_samples: int = 10,
    max_tries: int = 100,
) -> List[Tuple[mn.Vector3, mn.Quaternion]]:
    """Design for sampling on floor.
    :param env_interface: an env
    :param agent: the articulated agent
    :param proposition: next to or on
    :param entity_pos: the location of the entity
    :param agent: the articulated agent
    :param grasp_mgr: the grasp manager
    :param target_handle: if the target is an entity like a furniture or object, this is the handle
    :param is_articulated: is_articulated or not
    :param min_sample_distance: the minimum distance between two samples
    :param sample_region_scale: the scale of sampling region
    :param max_samples: the maximum number of samples to be included in the final return
    :param max_tries: the maximum number of tries to sample a position
    :return: A set of valid candidate placements (pos, orientation) sorted by distance to the agent.
    """

    assert proposition in [
        "next_to",
        "on",
    ], f"Only support 'next to' and 'on', but got {proposition}"

    num_tries = 0
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]] = []
    sampled_pos = None

    # cache agent's object ids for use within the loop
    agent_object_ids = [agent.sim_obj.object_id] + [
        *agent.sim_obj.link_object_ids.keys()
    ]
    # get the object_id for all links associated with all articulated agents so they can be ignored in navigation placement sampling
    agent_object_ids = []
    for articulated_agent in env_interface.sim.agents_mgr.articulated_agents_iter:
        agent_object_ids.extend(
            [articulated_agent.sim_obj.object_id]
            + [*articulated_agent.sim_obj.link_object_ids.keys()]
        )

    # collect the target object ids for occlusion checking
    target_object_ids = []
    if target_handle is not None:
        target_object = get_obj_from_handle(env_interface.sim, target_handle)
        target_object_ids = [target_object.object_id]
        if target_object.is_articulated:
            target_object_ids.extend([*target_object.link_object_ids.keys()])

    while len(sampled_poses) < max_samples and num_tries < max_tries:
        sampled_pos = unoccluded_navmesh_snap(
            pos=entity_pos,
            height=1.3,
            pathfinder=env_interface.sim.pathfinder,
            sim=env_interface.sim,
            island_id=env_interface.sim._largest_indoor_island_idx,
            target_object_ids=[],  # nothing should be hit to have a clear line of sight
            ignore_object_ids=agent_object_ids
            + target_object_ids,  # ignore the object and agent
        )
        # If the sampled_pos is None, then we fall back to use a safe snap point to get a valid pose
        if sampled_pos is None:
            sampled_pos = env_interface.sim.safe_snap_point(entity_pos)
        num_tries += 1

        # Cache the state of the grasped object
        cache_pos = grasp_mgr.snap_rigid_obj.translation
        cache_rot = grasp_mgr.snap_rigid_obj.rotation

        # Teleport the object to the sampled_pos
        grasp_mgr.snap_rigid_obj.translation = sampled_pos + mn.Vector3(0, 0.1, 0)
        # randomize the yaw orientation (around Y axis)
        rot = random.uniform(0, math.pi * 2.0)
        grasp_mgr.snap_rigid_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rot), mn.Vector3.y_axis()
        )
        snap_success = snap_down(
            env_interface.sim,
            grasp_mgr.snap_rigid_obj,
            max_collision_depth=0.2,
            support_obj_ids=get_floor_object_ids(
                env_interface.sim,
                grasp_mgr.snap_rigid_obj.translation,
                ignore_ids=agent_object_ids + target_object_ids,
            ),
            ignore_obj_ids=agent_object_ids,  # don't count impact with agent bodies
        )

        if snap_success:
            sampled_pos = grasp_mgr.snap_rigid_obj.translation
        else:
            grasp_mgr.snap_rigid_obj.translation = cache_pos
            grasp_mgr.snap_rigid_obj.rotation = cache_rot
            continue

        if proposition == "next_to":
            # Call sim next to function to check
            can_add = obj_next_to(
                env_interface.sim,
                grasp_mgr.snap_rigid_obj.object_id,
                target_object_ids[
                    0
                ],  # NOTE: check against the targetManagedObject.object_id in index 0
                hor_l2_threshold=0.5,  # NOTE: always use eval default for now
            )
        else:
            # If is "on", we do not check
            can_add = True

        if (
            distance_to_other_samples(sampled_pos, sampled_poses) > min_sample_distance
            and can_add
        ):
            sampled_poses.append((sampled_pos, grasp_mgr.snap_rigid_obj.rotation))

        # Snap the object back to its original position
        grasp_mgr.snap_rigid_obj.translation = cache_pos
        grasp_mgr.snap_rigid_obj.rotation = cache_rot

    if len(sampled_poses) == 0:
        return []

    # Sort the samples based on the distance to robot
    return sort_proposed_samples_based_on_distance_to_agent(sampled_poses, agent)
