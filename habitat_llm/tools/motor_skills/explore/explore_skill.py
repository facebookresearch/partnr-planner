# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import magnum as mn
import numpy as np
import quaternion
import torch

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy

try:
    from third_party.semantic_exploration.agents.sem_exp import (  # isort: skip
        Sem_Exp_Env_Agent,
    )
    import third_party.semantic_exploration.envs.utils.pose as pu  # isort: skip

except ImportError:
    Semantic_Mapping = None
    Sem_Exp_Env_Agent = None
    pu = None
    coco_categories = None


class ExploreSkill(SkillPolicy):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
        agent_uid,
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )

        self.env = env
        self.config = config
        # Exploration agent to process the state and output action
        self.agent = Sem_Exp_Env_Agent(config)
        self.first_call_map = True
        self.last_sim_location = None
        # Pass the info of the state information
        self.obs_info = {}
        # Debug
        self.previous_done = False
        # Get indices of base_velocity action
        self.action_range = find_action_range(self.action_space, "base_velocity")
        self.linear_velocity_index = self.action_range[0]
        self.angular_velocity_index = self.action_range[1] - 1

    def discrete_action_to_vel(self):
        """Transfer discrete action to velocity"""
        discrete_action = self.obs_info["action"]["action"]
        if discrete_action == 1:
            vel = [self.config.BASE_LIN_VEL, 0.0]  # go forward
        elif discrete_action == 2:
            vel = [0.0, self.config.BASE_ANGULAR_VEL]  # turn left
        elif discrete_action == 3:
            vel = [0.0, -self.config.BASE_ANGULAR_VEL]  # turn right
        else:
            vel = [0.0, 0.0]
        return vel

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""
        # Get the location of the agent
        agent_state = self.env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def update_obs_info(self, observations):
        """Update the input of the map"""
        self.obs_info = {}
        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.obs_info["sensor_pose"] = [dx, dy, do]
        # Get the image
        # size = (1, 256, 256, 3)
        rgb = np.array(observations["articulated_agent_arm_rgb"])[0]
        # size = (1, 256, 256, 1)
        depth = np.array(observations["articulated_agent_arm_depth"])[0]
        # size = (4, 256, 256)
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.obs_info["state"] = state
        # Update the step counter
        self.env.map.update_step_counter()

    def set_target(self, target, env):
        """Set the target (receptacle, object) of the skill"""
        self.target = target

    def fix_robot_leg(self):
        """
        Fix the robot leg's joint position
        """
        self.env.sim.articulated_agent.leg_joint_pos = (
            self.env.sim.articulated_agent.params.leg_init_params
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Initiate action as a vector of zeros
        action = torch.zeros(prev_actions.shape, device=masks.device)

        # If we need to reset the target id
        if self.previous_done:
            self.env.map.map_reset(self.target)
            self.agent.update_vis_image_goal(self.target.lower())
            self.previous_done = False

        # We update the map when calling for the first time
        if self.first_call_map:
            # Get the last location
            self.last_sim_location = self.get_sim_location()
            # Update the observation input for the map
            self.update_obs_info(observations)
            # Get the initial transformation
            init_trans = mn.Matrix4(
                self.env.sim.articulated_agent.sim_obj.transformation
            )
            # Initilize the map
            self.obs_info, planner_inputs = self.env.map.map_init(
                self.agent, self.obs_info, self.target, init_trans
            )
            # Reset the agent
            self.agent.reset(self.obs_info["state"].shape, self.target)
            # We have to feed the info information in to the planner agent
            # Update the info and get the observation input for the map
            # obs_info now contains action info
            _, _, _, self.obs_info = self.agent.plan_act_and_preprocess(
                planner_inputs[0], self.obs_info
            )
            # Set the first call of map to be false
            self.first_call_map = False
        else:
            # Update the observation input for the map
            self.update_obs_info(observations)
            self.obs_info, planner_inputs = self.env.map.map_update(
                self.agent, self.obs_info
            )
            # Update the agent to get the action
            _, _, done, self.obs_info = self.agent.plan_act_and_preprocess(
                planner_inputs[0], self.obs_info
            )

        if self.config.DEBUG_MODE:
            try:
                self.env.map.map_io(load=True)
            except Exception:
                print("cannot load the map")

            # Reset the robot's leg joints
            self.fix_robot_leg()

            # Mark the end of the episode
            self._has_reached_goal[cur_batch_idx] = 1.0

            return action, None

        vel = self.discrete_action_to_vel()

        # Mark if goal is reached
        at_goal = vel == [0, 0]

        if at_goal:
            self.previous_done = True
            # Save the map here for the future usage
            try:
                self.env.map.map_io(load=False)
            except Exception:
                print("cannot save the map due to PREBUILD_MAP_DIR being not exsited")

            # Indicate that goal has been reached
            self._has_reached_goal[cur_batch_idx] = 1.0

        else:
            # Indicate that goal has not been reached
            self._has_reached_goal[cur_batch_idx] = 0.0

        if self.env.map.i_step == self.config.max_skill_steps:
            try:
                self.env.map.map_io(load=False)
            except Exception:
                print("cannot save the map due to PREBUILD_MAP_DIR being not exsited")

        # Reset the robot's leg joints
        self.fix_robot_leg()

        action[cur_batch_idx, self.linear_velocity_index] = vel[0]
        action[cur_batch_idx, self.angular_velocity_index] = vel[1]

        return action, None
