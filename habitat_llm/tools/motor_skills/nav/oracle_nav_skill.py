# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List

import habitat_sim
import magnum as mn
import numpy as np
import torch
from habitat.datasets.rearrange.navmesh_utils import (
    SimpleVelocityControlEnv,
    compute_turn,
    embodied_unoccluded_navmesh_snap,
)
from habitat.sims.habitat_simulator.sim_utilities import (
    get_global_keypoints_from_object_id,
    get_obj_from_handle,
    get_obj_from_id,
)
from habitat.tasks.utils import get_angle

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import NAV_TARGET
from habitat_llm.utils.sim import (
    check_if_the_object_is_held_by_agent,
    get_receptacle_index,
)
from habitat_llm.world_model.entities.floor import Floor
from habitat_llm.world_model.entities.furniture import Furniture
from habitat_llm.world_model.entity import Object, Receptacle, Room


class OracleNavSkill(SkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )
        self.env = env
        # TODO: there may be cleaner ways to do this
        if f"agent_{self.agent_uid}_humanoidjoint_action" in action_space.spaces:
            self.motion_type = "human_joints"
        else:
            self.motion_type = "base_velocity"

        # pre-computed target pose for the ArticulatedAgent. See set_target.
        self.target_base_pos: mn.Vector3 = None
        self.target_base_rot: float = None
        self._has_reached_goal = torch.zeros(self._batch_size)

        # Define the velocity controller
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.dist_thresh = config.dist_thresh
        self.turn_thresh = config.turn_thresh
        self.forward_velocity = config.forward_velocity
        self.turn_velocity = config.turn_velocity
        self.sim_freq = config.sim_freq

        self.enable_backing_up = config.enable_backing_up

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent
        # by default we ignore the computed target rotation and instead face the navigation point
        # override this to respect the initial computed rotation
        self.face_object = config.get("face_object", True)

        self.do_teleport = False
        if "teleport" in config:
            self.do_teleport = config.teleport

        if self.do_teleport:
            # Get indices for teleport action
            target_pos_ends = find_action_range(
                self.action_space, f"agent_{self.agent_uid}_teleport"
            )
            self.target_pos_range = range(target_pos_ends[0], target_pos_ends[1])

        else:
            # Get indices for linear and angular velocities in the action tensor
            if self.motion_type != "human_joints":
                self.action_range = find_action_range(
                    self.action_space, f"agent_{self.agent_uid}_base_velocity"
                )
            else:
                self.action_range = find_action_range(
                    self.action_space, f"agent_{self.agent_uid}_humanoid_base_velocity"
                )
            self.linear_velocity_index = self.action_range[0]
            self.angular_velocity_index = self.action_range[1] - 1

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._has_reached_goal = torch.zeros(self._batch_size)
        self.target_is_set = False
        self.target_base_pos = None
        self.target_base_rot = None
        return

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""

        # Get room for agent
        room_node = self.env.world_graph[self.agent_uid].get_room_for_entity(
            f"agent_{self.agent_uid}"
        )
        return f"Walking in {room_node.name}"

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self.env.sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def is_collision(self, trans) -> bool:
        """
        The function checks if the agent collides with the object
        given the navmesh.
        """
        nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]])
            for xz in self.articulated_agent.params.navmesh_offsets
        ]  # type: ignore
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        cur_pos = [
            np.array([xz[0], self.articulated_agent.base_pos[1], xz[2]])
            for xz in cur_pos
        ]

        for pos in cur_pos:  # noqa: SIM110
            # Return true if the pathfinder says it is not navigable
            if not self.env.sim.pathfinder.is_navigable(pos):
                return True

        return False

    def fix_robot_leg(self):
        """
        Fix the robot leg's joint position
        """
        self.articulated_agent.leg_joint_pos = (
            self.articulated_agent.params.leg_init_params
        )

    def get_agent_object_ids(self) -> tuple[list[int], list[int]]:
        agent_object_ids: list[int] = []
        other_agent_object_ids: list[int] = []
        for articulated_agent in self.env.sim.agents_mgr.articulated_agents_iter:
            agent_object_ids.extend(
                [articulated_agent.sim_obj.object_id]
                + [*articulated_agent.sim_obj.link_object_ids.keys()]
            )
        return agent_object_ids, other_agent_object_ids

    def set_target(self, target_name: str, env):
        """
        Identify the target Entity (Receptacle, Object, Furniture) for the navigation skill and generate a target pose for the agent to reach the Entity.

        :param target_name: The name of the target Entity.
        """

        # Early return if the target is already set
        if self.target_is_set:
            return

        # Get Entity based on the target_name
        entity = self.env.world_graph[self.agent_uid].get_node_from_name(target_name)

        # Get sim handle of the target
        # TODO: handle CG entity here as well (read for place and pick)
        self.target_handle = entity.sim_handle

        # Set flag to True to avoid resetting the target
        self.target_is_set = True

        target_object_ids = []
        # get the object_id for all links associated with all articulated agents so they can be ignored in navigation placement sampling
        agent_object_ids, other_agent_object_ids = self.get_agent_object_ids()

        self.target_pos = None
        furniture_parent_handle = None
        if isinstance(entity, Object):
            if self._check_if_held_target():
                # early abort for grasped target Object
                # NOTE: the above function sets the failure flag and termination message
                return

            sim_obj = get_obj_from_handle(self.env.sim, self.target_handle)
            target_object_ids.append(sim_obj.object_id)

            # set target position from the world graph
            self.target_pos = entity.get_property("translation")
            # first try snapping to the Object center of mass, avoiding occlusion by anything else
            attempts = 0
            # track the nav point to object distance as we do rejection sampling
            obj_to_nav_point_dist = 0
            success = False
            # NOTE: Alex set a threshold here empirically (obj_to_nav_point_dist > 1.8) based on expected ee dist for pick. Should be re-evaluated.
            max_obj_to_nav_point_dist = 1.8
            # TODO: decide how to parameterize this distance or set from config
            while (
                obj_to_nav_point_dist > max_obj_to_nav_point_dist or not success
            ) and attempts < 200:
                (
                    self.target_base_pos,
                    self.target_base_rot,
                    success,
                ) = embodied_unoccluded_navmesh_snap(
                    target_position=mn.Vector3(self.target_pos),
                    height=1.3,  # TODO: hardcoded everywhere, should be config
                    sim=self.env.sim,
                    target_object_ids=target_object_ids,  # the target Entity's parts if applicable
                    ignore_object_ids=agent_object_ids,  # ignore the agent's body in occlusion checking
                    ignore_object_collision_ids=other_agent_object_ids,  # ignore the other agent's body in contact testing
                    island_id=self.env.sim._largest_indoor_island_idx,  # from RearrangeSim
                    min_sample_dist=0.25,  # approximates agent radius, doesn't need to be precise
                    agent_embodiment=self.articulated_agent,
                    orientation_noise=0.1,  # allow a bit of variation in body orientation
                )
                if success:
                    obj_to_nav_point_dist = (
                        mn.Vector3(self.target_base_pos) - mn.Vector3(self.target_pos)
                    ).length()
                attempts += 1
            if success and obj_to_nav_point_dist <= max_obj_to_nav_point_dist:
                # found an unoccluded pose for the agent to access the Object
                self.env.sim.dynamic_target = self.target_base_pos
                return

            # look for parent furniture (requires RearrangeSim for KRM)
            if self.env.sim.kinematic_relationship_manager is not None:
                furniture_parent_id = self.env.sim.kinematic_relationship_manager.relationship_graph.obj_to_parents.get(
                    sim_obj.object_id, None
                )
                if furniture_parent_id is None:
                    # no registered parent, so no more leads for how to navigate to the object
                    self.termination_message = f"Could not find a suitable nav target or parent Furniture for Entity Object '{target_name}'. Possibly inaccessible."
                    self.failed = True
                    return

                furniture_parent_obj = get_obj_from_id(
                    self.env.sim, furniture_parent_id
                )
                furniture_parent_handle = furniture_parent_obj.handle

        elif isinstance(entity, Receptacle):
            hab_rec = self.env.perception.receptacles[
                get_receptacle_index(entity.sim_handle, self.env.perception.receptacles)
            ]

            # set target position to the Receptacle's aabb center
            self.target_pos = hab_rec.get_global_transform(
                self.env.sim
            ).transform_point(hab_rec.bounds.center())
            furniture_parent_handle = hab_rec.parent_object_handle

        elif isinstance(entity, (Floor, Room)):
            # NOTE: Floor is Furniture, so this elif must come first
            # set target position from the world graph (in the room for this floor)
            self.target_pos = entity.get_property("translation")
        elif isinstance(entity, Furniture):
            furniture_parent_handle = entity.sim_handle
            if entity.sim_handle is None:
                raise ValueError(
                    f"Target Furniture.sim_handle == None. '{target_name}' could be a Floor node?."
                )
        else:
            raise ValueError(
                f"Oracle Nav skill accepts target Entities of types: Object, Furniture, and Receptacle. The provided Entity with name: '{entity.name}' is of unsupported type '{entity.__class__.__name__}'."
            )

        # find all object_ids associated with the parent furniture
        furniture_parent_object_ids = []
        # get the aabb to enable volume sampling
        furniture_aabb = None
        parent_obj = None
        if furniture_parent_handle is not None:
            parent_obj = get_obj_from_handle(self.env.sim, furniture_parent_handle)
            if parent_obj is None:
                raise ValueError(
                    f"Could not find parent object with sim handle '{furniture_parent_handle}'."
                )
            if self.target_pos is None:
                # if not already set, set the target from the Furniture object's center
                self.target_pos = get_global_keypoints_from_object_id(
                    self.env.sim, parent_obj.object_id
                )[0]
                furniture_aabb = parent_obj.aabb
            furniture_parent_object_ids = [parent_obj.object_id]
            if isinstance(parent_obj, habitat_sim.physics.ManagedArticulatedObject):
                furniture_parent_object_ids.extend([*parent_obj.link_object_ids.keys()])

        # look for interaction points to use if center sample fails
        global_points = [self.target_pos]
        if parent_obj is not None and parent_obj.marker_sets.has_taskset(
            "interaction_surface_points"
        ):
            interaction_points = parent_obj.marker_sets.get_task_link_markerset_points(
                "interaction_surface_points", "body", "primary"
            )
            global_points.extend(
                parent_obj.transform_local_pts_to_world(interaction_points, link_id=-1)
            )

        # now try to snap close to the target position allowing impact with the furniture as non-occlusion
        attempts = 0
        # NOTE: Alex set a threshold here empirically (obj_to_nav_point_dist > 1.8) based on expected ee dist for open/close/pick/place/etc... Should be re-evaluated.
        max_fur_to_nav_point_dist = 1.8
        # TODO: decide how to parameterize this distance or set from config
        fur_to_nav_point_dist = 0
        success = False
        while (
            not success or fur_to_nav_point_dist > max_fur_to_nav_point_dist
        ) and attempts < 200:
            target_variation = mn.Vector3()  # none by default
            target_position = self.target_pos
            if attempts < len(global_points):
                target_position = global_points[attempts]
            elif furniture_aabb is not None:
                # this means we are using the Furniture's center as the target point and the center target sample failed
                # we'll re-sample the target variation from a unit gaussian scaled to the bounding box
                gaus_samp = (
                    np.random.normal(0, 1.0, size=(2, 1)) / 3.5
                )  # NOTE: unit gaussian probability approaches 0 at abs(x)==3, scale the result
                target_variation = mn.Vector3(
                    gaus_samp[0], 0, gaus_samp[1]
                ) * parent_obj.transformation.transform_vector(
                    furniture_aabb.size() / 2.0
                )
                target_position = mn.Vector3(self.target_pos) + target_variation
            # NOTE: we allow multiple attempts because sampling does not guarantee finding a valid pose even if one exists.
            (
                self.target_base_pos,
                self.target_base_rot,
                success,
            ) = embodied_unoccluded_navmesh_snap(
                target_position=mn.Vector3(target_position),
                height=1.3,  # TODO: hardcoded everywhere, should be config
                sim=self.env.sim,
                target_object_ids=target_object_ids
                + furniture_parent_object_ids,  # the target Entity and its parent Furniture's parts as applicable
                ignore_object_ids=agent_object_ids,  # ignore the agent's body in occlusion checking
                ignore_object_collision_ids=other_agent_object_ids,  # ignore the other agent's body in contact testing
                island_id=self.env.sim._largest_indoor_island_idx,  # from RearrangeSim
                min_sample_dist=0.25,  # approximates agent radius, doesn't need to be precise
                agent_embodiment=self.articulated_agent,
                orientation_noise=0.1,  # allow a bit of variation in body orientation
            )
            if success:
                fur_to_nav_point_dist = (
                    mn.Vector3(self.target_base_pos) - mn.Vector3(target_position)
                ).length()
            attempts += 1
        if success and fur_to_nav_point_dist <= max_fur_to_nav_point_dist:
            self.env.sim.dynamic_target = self.target_base_pos
            return
        # if we're here, we failed to find a placement
        self.termination_message = f"Could not find a suitable nav target for {target_name}. Possibly inaccessible."
        self.failed = True

    def rotation_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.articulated_agent.sim_obj.transformation)
        # Initialize the velocity controller
        vc = SimpleVelocityControlEnv(120.0)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.articulated_agent.base_pos
        # Set the trans to be agent location
        trans.translation = self.articulated_agent.base_pos

        while abs(angle) > self.turn_thresh:
            # Compute the robot facing orientation
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
            trans = vc.act(trans, vel)
            cur_pos = trans.translation

            if self.is_collision(trans):
                return True

        return False

    def _check_if_held_target(self) -> bool:
        """
        Helper function to check if the target is an Object and if it is grasped by another agent.
        This should be called every step to early abort in case the target is picked up during nav.
        #NOTE: this function sets the termination message and failed flags before returning the result
        """
        # Currently it is set to None for floor nodes (which are typed Furniture)
        if self.target_handle is not None:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(
                self.target_handle
            )
            if isinstance(target_node, Object):
                dummy_action = torch.zeros(1, 1)  # placeholder action isn't used
                # Early exit if the object is being held by the other agent.
                (
                    dummy_action,  # zero action if failed
                    self.termination_message,
                    self.failed,
                ) = check_if_the_object_is_held_by_agent(
                    self.env, dummy_action, self.target_handle, self.agent_uid
                )
                if self.failed:
                    return True

        return False

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # We do not feed any velocity command
        action = torch.zeros(prev_actions.shape, device=masks.device)

        if self.failed:
            return action, self.termination_message

        # check for early abort if the target object is picked by another agent during nav
        if self._check_if_held_target():
            # NOTE: termination message and failed flags set in the check function
            return action, None

        if self.target_base_rot is None:
            self.failed = True

            return action, None
        # NOTE: this is set in super().set_target() from entity.translation
        # The location of the target objects
        obj_targ_pos = np.array(self.target_pos)

        if self.do_teleport:
            # One hot indicator stating that agent should take teleport action.
            action[cur_batch_idx, self.target_pos_range[0]] = 1.0

            # Set the agent's base_pos and base_rot from the pre-computed target pose
            action[cur_batch_idx, self.target_pos_range[1:4]] = torch.tensor(
                self.target_base_pos, dtype=torch.float32
            ).to(action.device)
            action[cur_batch_idx, self.target_pos_range[4]] = self.target_base_rot

            # teleported agent has reached the goal
            self._has_reached_goal[cur_batch_idx] = 1
            return action, None

        # Compute the shortest path from the current position to the target position
        # Get the base transformation for the robot
        base_T = self.articulated_agent.base_transformation
        # Find the paths
        curr_path_points = self._path_to_point(self.target_base_pos)
        # Get the robot position
        robot_pos = np.array(self.articulated_agent.base_pos)

        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")

        # Compute distance and angle to target
        if len(curr_path_points) == 1:
            curr_path_points += curr_path_points

        cur_nav_targ = np.array(curr_path_points[1])
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        # Compute relative target
        rel_targ = cur_nav_targ - robot_pos
        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        rel_targ = rel_targ[[0, 2]]
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
        dist_to_final_nav_targ = np.linalg.norm(
            (np.array(self.target_base_pos) - robot_pos)[[0, 2]],
        )
        if not self.face_object and dist_to_final_nav_targ < self.dist_thresh:
            # if we want to use the target rotation instead of facing the object
            # set the angle to target_base_rot only if we are already close to the nav goal
            rel_pos = np.array(
                [np.cos(self.target_base_rot), -np.sin(self.target_base_rot)]
            )
            rel_targ = rel_pos
        angle_to_target = get_angle(robot_forward, rel_targ)
        angle_to_obj = get_angle(robot_forward, rel_pos)
        # Compute the distance
        at_goal = (
            dist_to_final_nav_targ < self.dist_thresh
            and angle_to_obj < self.turn_thresh
        )

        if self.motion_type == "base_velocity":
            # Planning to see if the robot needs to do back-up
            need_move_backward = False
            if (
                dist_to_final_nav_targ >= self.dist_thresh
                and angle_to_target >= self.turn_thresh
                and not at_goal
            ):
                # check if there is a collision caused by rotation
                # if it does, we should block the rotation, and
                # only move backward
                need_move_backward = self.rotation_collision_check(
                    cur_nav_targ,
                )

            if need_move_backward and self.enable_backing_up:
                # Backward direction
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                # Compute relative target
                rel_targ = cur_nav_targ - robot_pos
                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                # Get the angles
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                # Compute the distance
                dist_to_final_nav_targ = np.linalg.norm(
                    (self.target_base_pos - robot_pos)[[0, 2]],
                )
                at_goal = (
                    dist_to_final_nav_targ < self.dist_thresh
                    and angle_to_obj < self.turn_thresh
                )

            if not at_goal:
                if dist_to_final_nav_targ < self.dist_thresh:
                    # TODO: this does not account for the sampled pose's final rotation
                    # Look at the object target position when getting close
                    vel = compute_turn(
                        rel_pos,
                        self.turn_velocity,
                        robot_forward,
                    )
                elif angle_to_target < self.turn_thresh:
                    # Move forward towards the target
                    vel = [self.forward_velocity, 0]
                else:
                    # Look at the target waypoint
                    vel = compute_turn(
                        rel_targ,
                        self.turn_velocity,
                        robot_forward,
                    )
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            if need_move_backward:
                vel[0] = -1 * vel[0]

            # Reset the robot's leg joints
            self.fix_robot_leg()

            # Populate the actions tensor
            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]
        else:
            if not at_goal:
                if dist_to_final_nav_targ < self._config.dist_thresh:
                    # Look at the object
                    vel = compute_turn(
                        rel_pos,
                        self.turn_velocity,
                        robot_forward,
                    )
                elif angle_to_target < self.turn_thresh:
                    # Move forward towards the target
                    vel = [self.forward_velocity, 0]
                else:
                    # Look at the target waypoint
                    vel = compute_turn(
                        rel_targ,
                        self.turn_velocity,
                        robot_forward,
                    )
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            # Populate the actions tensor
            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]

        return action, None

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        return (self._has_reached_goal[batch_idx] > 0.0).to(masks.device)

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleNavSkill.

        Right now only allowing objects, furniture and rooms as this is all the
        planner can reason about.

        :return: List of argument types.
        """
        return [NAV_TARGET]
