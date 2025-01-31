# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import magnum as mn
import numpy as np
from habitat.datasets.rearrange.navmesh_utils import embodied_unoccluded_navmesh_snap

from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill


class OraclePointNavSkill(OracleNavSkill):
    def set_target(self, input_str: str, _):
        """
        Set the target position and rotation for the navigation skill in global sim coordinates.
        The input string should be in the format of "x,z,yaw"

        :param input_str: The target position in world coordinates.
        :param _: Unused to match super class signature
        """

        if self.target_is_set:
            return

        # Parse the input string into x,y,yaw values
        try:
            x, y, yaw = [float(val.strip()) for val in input_str.split(",")]
        except ValueError:
            raise ValueError("Input must be in format 'x,y,yaw' with numeric values")

        agent_object_ids, other_agent_object_ids = self.get_agent_object_ids()
        self.target_pos = np.array([x, 1.3, y])
        success = False
        attempts = 0
        # Snap to navmesh
        while not success and attempts < 200:
            self.target_base_pos, _, success = embodied_unoccluded_navmesh_snap(
                target_position=mn.Vector3(self.target_pos),
                height=1.3,  # TODO: hardcoded everywhere, should be config
                sim=self.env.sim,
                ignore_object_ids=agent_object_ids,  # ignore the agent's body in occlusion checking
                ignore_object_collision_ids=other_agent_object_ids,  # ignore the other agent's body in contact testing
                island_id=self.env.sim._largest_indoor_island_idx,  # from RearrangeSim
                min_sample_dist=0.25,  # approximates agent radius, doesn't need to be precise
                agent_embodiment=self.articulated_agent,
                orientation_noise=0.1,  # allow a bit of variation in body orientation
            )
            attempts += 1
        if not success:
            raise ValueError("Failed to find an unoccluded navmesh snap point")
        # Set flag to True to avoid resetting the target
        self.target_base_rot = yaw
        self.target_is_set = True
