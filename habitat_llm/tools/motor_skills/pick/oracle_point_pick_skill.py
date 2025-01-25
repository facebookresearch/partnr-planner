# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

import numpy as np
from habitat.sims.habitat_simulator.sim_utilities import (
    get_global_keypoints_from_object_id,
    get_obj_from_handle,
    get_obj_size_along,
)

from habitat_llm.tools.motor_skills.pick.oracle_pick_skill import OraclePickSkill


class OraclePointPickSkill(OraclePickSkill):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config, observation_space, action_space, batch_size, env, agent_uid
        )
        self.selection_snap_distance = config.selection_snap_distance

    def set_target(self, input_str: str, env: "EnvironmentInterface"):
        """
        Set the target for grasping based on global sim coordinates.
        The input string should be in the format of "x,y,z"

        :param input_str: The target position in world coordinates.
        :param env: The environment instance.
        """

        try:
            x, y, z = [float(val.strip()) for val in input_str.split(",")]
        except ValueError:
            raise ValueError("Input must be in format 'x,y,yaw' with numeric values")

        min_dist = float("inf")
        closest_obj = None
        # restrict the search to objects in the world graph to be consistent the name based
        # pick skill. To use all objects iterate over get_all_objects(env.sim) from habitat_sim utils
        objs = self.env.world_graph[self.agent_uid].get_all_objects()
        for obj in objs:
            sim_obj = get_obj_from_handle(env.sim, obj.sim_handle)
            # get the center of the object bounding box
            center = get_global_keypoints_from_object_id(env.sim, sim_obj.object_id)[0]
            # get the vector from the object center to target pick point
            object_to_target = np.array([x, y, z]) - center
            # get object size along the vector
            size = get_obj_size_along(env.sim, sim_obj.object_id, object_to_target)[0]
            # remove object size from the computed distance
            distance_to_surface = np.linalg.norm(object_to_target) - size
            if distance_to_surface is not None and distance_to_surface < min_dist:
                min_dist = distance_to_surface
                closest_obj = obj
        if min_dist < self.selection_snap_distance:
            super().set_target(closest_obj.name, env)
        else:
            self.termination_message = "No object found near the target position"
            self.failed = True
