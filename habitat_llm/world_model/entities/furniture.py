#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING, List, Tuple

import magnum as mn
import numpy as np
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_handle,
    obj_next_to,
    snap_down,
)

from habitat_llm.world_model.entity import Entity, Object

if TYPE_CHECKING:
    from habitat.articulated_agents import ArticulatedAgentBase
    from habitat.tasks.rearrange.rearrange_grasp_manager import RearrangeGraspManager

    from habitat_llm.agent.env import EnvironmentInterface


def sort_proposed_samples_based_on_distance_to_agent(
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]], agent: ArticulatedAgentBase
) -> List[np.ndarray]:
    """Sort the samples based on the distance to agent.
    :param sampled_poses: a list of placement tuples (point,orientation) to compare with
    :param agent: an ArticulatedAgent

    :return: the sorted list of placement points based on their L2 distance to the agent's base position
    """
    cur_base_pos = mn.Vector3(agent.base_pos)
    distance = [(cur_base_pos - sample[0]).length() for sample in sampled_poses]
    # Sort
    sort_i = sorted(range(len(distance)), key=lambda k: distance[k])
    return [sampled_poses[i] for i in sort_i]


def distance_to_other_samples(
    new_sample: mn.Vector3,
    samples: List[Tuple[mn.Vector3, mn.Quaternion]],
) -> float:
    """Compute the distance to other samples in the list.
    :param new_sample: a new placement point
    :param samples: a list of placement tuples (point, orientation) to compare with

    :return: the min L2 distance to other samples
    """
    if len(samples) == 0:
        return float("inf")
    distance = [(new_sample - sample[0]).length() for sample in samples]
    return min(distance)


class Furniture(Entity):
    """
    This class represents a furniture in the world
    which can contain objects
    """

    # Parameterized Constructor
    def __init__(self, name, properties, sim_handle=None):
        # Call Entity constructor
        super().__init__(name, properties, sim_handle)

    # Deep copy
    def __deepcopy__(self, memo):
        return Furniture(
            copy.deepcopy(self.name, memo),
            copy.deepcopy(self.properties, memo),
            copy.deepcopy(self.sim_handle, memo),
        )

    def is_articulated(self):
        """
        This method tells if the furniture is articulated or not
        """
        return self.properties.get("is_articulated", False)

    def sample_place_location(
        self,
        spatial_relation: str,
        spatial_constraint: str | None,
        reference_object: Object | None,
        env: "EnvironmentInterface",
        agent: ArticulatedAgentBase,
        grasp_mgr: RearrangeGraspManager = None,
    ) -> List[Tuple[mn.Vector3, mn.Quaternion]]:
        """
        Compute valid placement locations on this furniture.
        :param spatial_relation: string representing spatial relation between object and furniture
        :param spatial_constraint: string representing spatial constraint between obj and reference obj
        :param reference_object: node from world graph representing reference object
        :param env: an instance of environment interface
        :param agent: the articulated agent that will be placing the object
        :param grasp_mgr: the active grasp manager
        :return: A set of valid candidate placements (pos, orientation) sorted by distance to the agent.
        """
        return sample_position_on_furniture(
            spatial_relation,
            self,
            spatial_constraint,
            reference_object,
            env,
            agent=agent,
            grasp_mgr=grasp_mgr,
        )


def sample_position_on_furniture(
    spatial_relation: str,
    place_entity: Furniture,
    spatial_constraint: str | None,
    reference_object: Object | None,
    env_interface: "EnvironmentInterface",
    agent: ArticulatedAgentBase,
    grasp_mgr: RearrangeGraspManager,
    min_sample_distance: float = 0.10,
    sample_region_scale: float = 1,
    max_samples: int = 10,
    max_tries: int = 100,
) -> List[Tuple[mn.Vector3, mn.Quaternion]]:
    """
    Sample points on Receptacles on/inside/within both rigid furniture and articulated furniture (e.g. drawers/cabinets).

    :param spatial_relation: string representing the spatial relationship between the object and furniture
    :param place_entity: node from the world graph representing the entity on which to place an object
    :param spatial_constraint: string representing the spatial constraint between the object and a reference object
    :param reference_object: node from the world graph representing the reference object
    :param env_interface: an env
    :param agent: the articulated agent
    :param grasp_mgr: the grasping manager
    :param min_sample_distance: the minimum distance between two sampled placement locations
    :param sample_region_scale: a uniform scaling value for shrinking the sampling region of Receptacles which support scaling
    :param max_samples: the maximum number of samples to be included in the final return
    :param max_tries: the maximum number of tries to sample a position

    :return: A set of valid candidate placements (pos, orientation) sorted by distance to the agent.
    """

    # Throw if the spatial relation is invalid
    if spatial_relation not in ["on", "within"]:
        raise ValueError("spatial relation can only be 'on' and 'within'")

    # Throw if the spatial relation is invalid
    if spatial_constraint is not None and spatial_constraint != "next_to":
        raise ValueError("spatial constraint can only be 'next_to'")

    # Get fur to rec map
    fur_to_rec_map = env_interface.perception.fur_to_rec

    # Make sure that the furniture is in the fur_to_rec_map
    if place_entity.sim_handle not in fur_to_rec_map:
        raise ValueError(
            f"Entity with handle {place_entity.sim_handle} not found in fur_to_rec_map"
        )

    # Get the list of all receptacle which satisfy given spatial relationship
    candidate_rec = fur_to_rec_map[place_entity.sim_handle][spatial_relation]

    # Throw if no valid rec are found
    if len(candidate_rec) == 0:
        raise ValueError(
            f"Furniture {place_entity.name} has no receptacle for proposition {spatial_relation}"
        )

    # Declare container to store sampled poses
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]] = []
    num_tries = 0

    # Rejection sampling
    while len(sampled_poses) < max_samples and num_tries < max_tries:
        # Select a random Receptacle from the valid spatial subset
        rec = random.choice(candidate_rec)
        sampled_pos = rec.sample_uniform_global(env_interface.sim, sample_region_scale)
        num_tries += 1

        # Cache the state of the grasped object
        cache_pos = grasp_mgr.snap_rigid_obj.translation
        cache_rot = grasp_mgr.snap_rigid_obj.rotation

        # find id of the receptacle
        obj = get_obj_from_handle(env_interface.sim, rec.parent_object_handle)
        rec_link_id = rec.parent_link
        obj_id = obj.object_id
        if rec_link_id is not None and rec_link_id >= 0:
            obj_id = obj.link_ids_to_object_ids[rec_link_id]

        # Teleport the object to the sampled_pos
        grasp_mgr.snap_rigid_obj.translation = sampled_pos + mn.Vector3(0, 0.08, 0)
        # randomize the yaw orientation (around Y axis)
        rot = random.uniform(0, math.pi * 2.0)
        grasp_mgr.snap_rigid_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rot), mn.Vector3.y_axis()
        )
        snap_success = snap_down(
            env_interface.sim,
            grasp_mgr.snap_rigid_obj,
            support_obj_ids=[obj_id],
        )

        if snap_success:
            sampled_pos = grasp_mgr.snap_rigid_obj.translation
        else:
            grasp_mgr.snap_rigid_obj.translation = cache_pos
            grasp_mgr.snap_rigid_obj.rotation = cache_rot
            continue

        # Load rigid and articulated object managers
        rom = env_interface.sim.get_rigid_object_manager()
        # TODO: currently using the default l2_threshold=0.5 everywhere.
        # However, this is configurable per-proposition and should be pulled from config
        # (e.g. hor_l2_threshold=env_interface.env.env.env._env.current_episode.evaluation_propositions).
        # Call sim next to function to check
        can_add = True
        if spatial_constraint is not None and spatial_constraint == "next_to":
            can_add = obj_next_to(
                env_interface.sim,
                grasp_mgr.snap_rigid_obj.object_id,
                rom.get_object_by_handle(reference_object.sim_handle).object_id,
            )
        if (
            distance_to_other_samples(sampled_pos, sampled_poses) > min_sample_distance
            and can_add
        ):
            sampled_poses.append((sampled_pos, grasp_mgr.snap_rigid_obj.rotation))
        # Snap the object back to its original position
        grasp_mgr.snap_rigid_obj.translation = cache_pos
        grasp_mgr.snap_rigid_obj.rotation = cache_rot

        if len(sampled_poses) >= max_samples or num_tries >= max_tries:
            break

    if len(sampled_poses) == 0:
        return []

    # Sort the samples based on the distance to robot
    return sort_proposed_samples_based_on_distance_to_agent(sampled_poses, agent)
