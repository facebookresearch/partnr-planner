#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This file contains the definition of the actiosn available to the agent.
"""
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

import habitat
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
import magnum as mn
import numpy as np
from gym import spaces
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.config.default_structured_configs import ActionConfig
from habitat.core.registry import registry
from habitat.core.spaces import ActionSpace
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.object_state_machine import (
    get_state_of_obj,
    set_state_of_obj,
)
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_baselines.utils.common import get_num_actions
from hydra.core.config_store import ConfigStore

from habitat_llm.utils.sim import ee_distance_to_object

# Add your actions to the HabitatSimActions
if not HabitatSimActions.has_action("oracle_pick_action"):
    HabitatSimActions.extend_action_space("oracle_pick_action")
if not HabitatSimActions.has_action("oracle_place_action"):
    HabitatSimActions.extend_action_space("oracle_place_action")
if not HabitatSimActions.has_action("oracle_open_action"):
    HabitatSimActions.extend_action_space("oracle_open_action")
if not HabitatSimActions.has_action("oracle_close_action"):
    HabitatSimActions.extend_action_space("oracle_close_action")


# Method to find action range
# An equivalent method exists in habitat-lab but its buggy
def find_action_range(action_space: ActionSpace, search_key: str) -> Tuple[int, int]:
    """
    Returns the start and end indices of an action key in the action tensor. If
    the key is not found, a Value error will be thrown.
    :param action_space: The set of all actions we consider.
    :param search_key: The action for which we want to find the range.
    """

    start_idx = 0
    found = False
    end_idx = get_num_actions(action_space[search_key])
    for k in action_space:
        if k == search_key:
            found = True
            break
        start_idx += get_num_actions(action_space[k])
    if not found:
        raise ValueError(f"Could not find {search_key} action in {action_space}")
    return start_idx, start_idx + end_idx


############################################################
# Define your custom actions below
############################################################
@registry.register_task_action
class HumanoidBaseVelAction(HumanoidJointAction):
    """
    Class to move the humanoid. Given a forward velocity and rotation computes
    joints to move the humanoid accordingly.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        HumanoidJointAction.__init__(self, *args, **kwargs)
        self.humanoid_controller = self.lazy_inst_humanoid_controller(task, config)
        self.initialized_pose = False
        self._lin_speed = self._config.lin_speed
        self._ang_speed = self._config.ang_speed

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if not hasattr(task, "humanoid_controller") or task.humanoid_controller is None:
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[self._agent_index]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)

            task.humanoid_controller = humanoid_controller

        return task.humanoid_controller

    def reset_controller(self):
        if not self.initialized_pose:
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
            self.initialized_pose = True

    def step(self, *args, **kwargs):
        self.reset_controller()
        ctrl_freq = self._sim.ctrl_freq
        lin_vel, ang_vel = kwargs[self._action_arg_prefix + "humanoid_base_vel"]
        lin_vel = np.clip(lin_vel, -1, 1) * self._lin_speed
        ang_vel = np.clip(ang_vel, -1, 1) * self._ang_speed
        # for now only rotate or move forward

        if lin_vel == 0 and ang_vel == 0:
            self.humanoid_controller.calculate_stop_pose()
        else:
            lin_delta = lin_vel * 1.0 / ctrl_freq
            rot_delta = ang_vel * 1.0 / ctrl_freq
            # Move towards the target
            self.humanoid_controller.translate_and_rotate_with_gait(
                lin_delta, rot_delta
            )
        base_action = self.humanoid_controller.get_pose()
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
        HumanoidJointAction.step(self, *args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        super().reset()
        self.initialized_pose = False

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "humanoid_base_vel": spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )


@registry.register_task_action
class OraclePickAction(ArticulatedAgentAction):
    """This action snaps the object specified by the object index if grip flag is true."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    @property
    def action_space(self):
        action_spaces = {
            self._action_arg_prefix
            + "oracle_grip_action": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + "pick_action_object_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
        }
        return spaces.Dict(action_spaces)

    def step(self, *args, **kwargs):
        grip_flag = kwargs[self._action_arg_prefix + "oracle_grip_action"][0]
        obj_idx = int(kwargs[self._action_arg_prefix + "pick_action_object_idx"][0])

        if grip_flag:
            keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))
            self.cur_grasp_mgr.snap_to_obj(
                obj_idx,
                force=True,
                rel_pos=mn.Vector3(0.1, 0.0, 0.0),
                keep_T=keep_T,
            )

        return


@registry.register_task_action
class OraclePlaceAction(ArticulatedAgentAction):
    """
    This action places the snapped object on the object or surface directly below the provided target_position.
    When the placement location is invalid, object remains grasped and the state is unchanged.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    @property
    def action_space(self):
        action_spaces = {
            self._action_arg_prefix
            + "release_flag": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=float,
            ),
            self._action_arg_prefix
            + "target_position": spaces.Box(
                shape=(3,),
                low=-sys.maxsize,
                high=sys.maxsize,
                dtype=float,
            ),
        }
        return spaces.Dict(action_spaces)

    def step(self, *args, **kwargs):
        release_flag = kwargs[self._action_arg_prefix + "release_flag"][0]
        target_position = kwargs[self._action_arg_prefix + "target_position"]

        if release_flag:
            snapped_obj_id = self.cur_grasp_mgr._snapped_obj_id
            obj_to_place = self._sim.get_rigid_object_manager().get_object_by_id(
                snapped_obj_id
            )
            cur_obj_pos = obj_to_place.translation

            # Place the object at the target position to start the placement snapping
            obj_to_place.translation = target_position

            # Collect the agent ids for use with the snapdown function
            # this allows the snapdown to ignore the agent.
            # Otherwise a sampled placement location underneath the end-effector or another agent will fail.
            agent_object_ids = []
            for articulated_agent in self._sim.agents_mgr.articulated_agents_iter:
                agent_object_ids.extend(
                    [articulated_agent.sim_obj.object_id]
                    + [*articulated_agent.sim_obj.link_object_ids.keys()]
                )

            snap_success = False
            # raycast downward from the target position to find the expected support surface.
            ray = habitat_sim.geo.Ray(target_position, mn.Vector3(0, -1.0, 0))
            raycast_results = self._sim.cast_ray(ray)
            if raycast_results.has_hits():
                for hit in raycast_results.hits:
                    support_surface_id = hit.object_id
                    if support_surface_id != obj_to_place.object_id:
                        # set max_collision_depth and add agent as support object ids following the pattern in the floor point sampling code
                        snap_success = sutils.snap_down(
                            self._sim,
                            obj_to_place,
                            max_collision_depth=0.2,
                            support_obj_ids=[support_surface_id],
                            ignore_obj_ids=agent_object_ids,
                        )
                        if not snap_success:
                            # NOTE: failed because there is too much collision in the snapped position with the support surface or other objects
                            pass
                        elif (
                            self._sim._kinematic_mode
                            and support_surface_id != habitat_sim.stage_id
                        ):
                            # snapping was successful, add the kinematic relationship
                            self._sim.kinematic_relationship_manager.relationship_graph.add_relation(
                                support_surface_id, obj_to_place.object_id, "ontop"
                            )
                            self._sim.kinematic_relationship_manager.update_snapshots()
                        # at this point we've completed snapping to the support surface and failed or succeeded
                        break
            else:
                # NOTE: failed because placing into the void, no support surface
                pass

            # process success or failure
            if snap_success:
                self.cur_grasp_mgr.desnap(True)
            else:
                # the action failed, put the object back in its original location
                obj_to_place.translation = cur_obj_pos


@registry.register_task_action
class OracleOpenAction(ArticulatedAgentAction):
    """This action opens the default link of the specified articulated furniture object."""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    @property
    def action_space(self):
        # Adding numerals to enforce order during alphabetical sort
        action_spaces = {
            self._action_arg_prefix
            + "1_oa_open_flag": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + "2_oa_object_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
            self._action_arg_prefix
            + "3_oa_is_surface_flag": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + "4_oa_surface_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
        }

        return spaces.Dict(action_spaces)

    def step(self, *args, **kwargs):
        should_open = kwargs[self._action_arg_prefix + "1_oa_open_flag"][0]
        object_id = kwargs[self._action_arg_prefix + "2_oa_object_idx"][0]

        # TODO: refactor this to take either the furniture + link OR to the surface (backsolved to link)
        is_surface = kwargs[self._action_arg_prefix + "3_oa_is_surface_flag"][0]
        # NOTE: This is the index in the shared global Receptacles list
        surface_index = kwargs[self._action_arg_prefix + "4_oa_surface_idx"][0]

        if should_open:
            # get the specified ao
            ao = None
            if is_surface:
                # NOTE: Maybe remove this path, but for now, get the ao from the Receptacle
                rec = self._sim.receptacles[self._sim.receptacles.keys()[surface_index]]
                ao = sutils.get_obj_from_handle(self._sim, rec.parent_object_handle)
            else:
                ao = sutils.get_obj_from_id(self._sim, object_id)

            # query or compute the default link
            default_link = sutils.get_ao_default_link(ao, compute_if_not_found=True)
            if default_link is None:
                # NOTE: no link to open, so silently succeed without changing state
                pass
            else:
                # open the default link
                sutils.open_link(ao, default_link)
                if self._sim._kinematic_mode:
                    self._sim.kinematic_relationship_manager.apply_relations()


@registry.register_task_action
class OracleCloseAction(OracleOpenAction):
    """This action closes the receptacle"""

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.does_want_terminate = False

    @property
    def action_space(self):
        # Adding numerals to enforce order during alphabetical sort

        action_spaces = {
            self._action_arg_prefix
            + "1_ca_close_flag": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + "2_ca_object_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
            self._action_arg_prefix
            + "3_ca_is_surface_flag": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + "4_ca_surface_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
        }
        return spaces.Dict(action_spaces)

    def step(self, *args, **kwargs):
        should_close = kwargs[self._action_arg_prefix + "1_ca_close_flag"][0]
        object_id = kwargs[self._action_arg_prefix + "2_ca_object_idx"][0]
        is_surface = kwargs[self._action_arg_prefix + "3_ca_is_surface_flag"][0]
        surface_index = kwargs[self._action_arg_prefix + "4_ca_surface_idx"][0]

        if should_close:
            # get the specified ao
            ao = None
            if is_surface:
                # NOTE: Maybe remove this path, but for now, get the ao from the Receptacle
                rec = self._sim.receptacles[self._sim.receptacles.keys()[surface_index]]
                ao = sutils.get_obj_from_handle(self._sim, rec.parent_object_handle)
            else:
                ao = sutils.get_obj_from_id(self._sim, object_id)

            # query or compute the default link
            default_link = sutils.get_ao_default_link(ao, compute_if_not_found=True)
            if default_link is None:
                # NOTE: no link to open, so silently succeed without changing state
                pass
            else:
                # close the default link
                sutils.close_link(ao, default_link)
                if self._sim._kinematic_mode:
                    self._sim.kinematic_relationship_manager.apply_relations()


@registry.register_task_action
class TeleportAction(ArticulatedAgentAction):
    """
    Teleports the agent to some position, can also specify two positions, that will indicate
    that the agent should be in position A and looking at position B
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._task = kwargs["task"]

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "should_teleport": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                self._action_arg_prefix
                + "teleport_action_position": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                self._action_arg_prefix
                + "teleport_action_yaw": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
            }
        )

    def step(self, *args, **kwargs):
        should_teleport = kwargs[self._action_arg_prefix + "should_teleport"][0]
        if should_teleport != 0:
            position_agent = kwargs[
                self._action_arg_prefix + "teleport_action_position"
            ]

            self.cur_articulated_agent.base_pos = position_agent
            yaw = kwargs[self._action_arg_prefix + "teleport_action_yaw"]
            self.cur_articulated_agent.base_rot = yaw
            if type(self.cur_articulated_agent) == KinematicHumanoid:
                # Update the humanoid controller
                self._task.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )


class OracleObjectStateAction(ArticulatedAgentAction):
    """Modifies the object state of an object
    Requires an instance of RearrangeSim which has been initialized with an object state manager
    (see agent/env/environment_interface.py)
    When called, the action assigns `value` to the specified object state

    """

    def __init__(
        self,
        *args,
        config,
        sim: RearrangeSim,
        action_name: str,
        object_state: str,
        value: Any,
        maximum_distance: float = 1.5,
        **kwargs,
    ):
        """

        :param sim: The simulation environment.
        :paraam action_name: The name of the action
        :param object_state: The string indicating the object state to be modified.
        :param value: The value to be assigned to the object state.
        :param maximum_distance: The maximum distance for the agent to be able to modify the object state.
        """
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self.action_name = action_name
        self.object_state = object_state
        self.value = value
        self.maximum_distance = maximum_distance

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            self._action_arg_prefix
            + f"oracle_{self.action_name}_action": spaces.Box(
                shape=(1,),
                low=0,
                high=1,
                dtype=int,
            ),
            self._action_arg_prefix
            + f"oracle_{self.action_name}_action_object_idx": spaces.Box(
                shape=(1,),
                low=0,
                high=sys.maxsize,
                dtype=int,
            ),
        }
        return spaces.Dict(action_spaces)

    def step(self, *args, **kwargs):
        """
        Performs a step in the OracleObjectStateAction.

        If the action flag is not 0, sets `self.object_state` to `self.value` in the simulator for the specified object
        if the distance from the agent to the object is within self.maximum_distance.
        """
        action_flag = kwargs[
            self._action_arg_prefix + f"oracle_{self.action_name}_action"
        ][0]
        if action_flag != 0:
            obj_idx = int(
                kwargs[
                    self._action_arg_prefix
                    + f"oracle_{self.action_name}_action_object_idx"
                ][0]
            )
            obj = get_obj_from_id(self._sim, obj_idx)
            dist = ee_distance_to_object(
                self._sim,
                self._sim.agents_mgr,
                self._agent_index,
                obj.handle,
                max_distance=self.maximum_distance,
            )
            if dist is None or dist > self.maximum_distance:
                print(
                    f"Agent too far from object to change {self.object_state}: Distance {dist}"
                )
            else:
                set_state_of_obj(obj, self.object_state, self.value)


@registry.register_task_action
class OraclePowerOnAction(OracleObjectStateAction):
    """Modifies the 'is_powered_on' object state of an object
    Requires an instance of RearrangeSim which has been initialized with an object state manager
    (see agent/env/environment_interface.py)
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(
            *args,
            config=config,
            sim=sim,
            action_name="power_on",
            value=True,
            object_state="is_powered_on",
            **kwargs,
        )


@registry.register_task_action
class OraclePowerOffAction(OracleObjectStateAction):
    """Modifies the 'is_powered_on' object state of an object
    Requires an instance of RearrangeSim which has been initialized with an object state manager
    (see agent/env/environment_interface.py)
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(
            *args,
            config=config,
            sim=sim,
            action_name="power_off",
            value=False,
            object_state="is_powered_on",
            **kwargs,
        )


@registry.register_task_action
class OracleCleanAction(OracleObjectStateAction):
    """Modifies the 'is_clean' object state of an object
    Requires an instance of RearrangeSim which has been initialized with an object state manager
    (see agent/env/environment_interface.py)
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(
            *args,
            config=config,
            sim=sim,
            action_name="clean",
            object_state="is_clean",
            value=True,
            **kwargs,
        )


@registry.register_task_action
class OracleFillAction(OracleObjectStateAction):
    """Modifies the 'is_filled' object state of an object
    Requires an instance of RearrangeSim which has been initialized with an object state manager
    (see agent/env/environment_interface.py)
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(
            *args,
            config=config,
            sim=sim,
            action_name="fill",
            object_state="is_filled",
            value=True,
            **kwargs,
        )


@registry.register_task_action
class OraclePourAction(OracleObjectStateAction):
    """Sets the 'is_filled' object state of an object to 1 of the agent is close enough to the object
    and currently holding another object with the 'is_filled' state set to 1.
    """

    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(
            *args,
            config=config,
            sim=sim,
            action_name="pour",
            object_state="is_filled",
            value=True,
            **kwargs,
        )

    def step(self, *args, **kwargs):
        """
        Performs a step in the OracleObjectStateAction.

        If the action flag is not 0, sets `self.object_state` to `self.value` in the simulator for the specified object
        if the distance from the agent to the object is within self.maximum_distance.
        """
        action_flag = kwargs[
            self._action_arg_prefix + f"oracle_{self.action_name}_action"
        ][0]
        if action_flag != 0:
            obj_idx = int(
                kwargs[
                    self._action_arg_prefix
                    + f"oracle_{self.action_name}_action_object_idx"
                ][0]
            )
            if not self.cur_grasp_mgr.is_grasped:
                print("Unable to pour: Agent is not holding an object")
                return
            obj = sutils.get_obj_from_id(self._sim, self.cur_grasp_mgr.snap_idx)
            if not get_state_of_obj(obj, "is_filled"):
                print("Unable to pour: The held object is not filled")
                return
            obj = sutils.get_obj_from_id(self._sim, obj_idx)
            dist = ee_distance_to_object(
                self._sim,
                self._sim.agents_mgr,
                self._agent_index,
                obj.handle,
                max_distance=self.maximum_distance,
            )
            if dist is None or dist > self.maximum_distance:
                print(
                    f"Agent too far from object to change {self.object_state}: Distance {dist}"
                )
            else:
                set_state_of_obj(obj, self.object_state, self.value)


############################################################
# Define your action configs below
############################################################


@dataclass
class TeleportActionConfig(ActionConfig):
    r"""
    Teleport action config.
    """
    type: str = "TeleportAction"
    name: str = "teleport"


@dataclass
class HumanoidBaseVelocityActionConfig(ActionConfig):
    r"""
    Base velocity for the humanoid.
    """
    type: str = "HumanoidBaseVelAction"
    name: str = "humanoid_base_velocity"
    num_joints: int = 54
    lin_speed: float = 10.0
    ang_speed: float = 10.0


@dataclass
class OraclePickActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will snap the object to robot arm
    """
    type: str = "OraclePickAction"
    name: str = "oracle_pick_action"
    dimensionality: int = 2


@dataclass
class OraclePlaceActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will snap the object to target position
    """
    type: str = "OraclePlaceAction"
    name: str = "oracle_place_action"
    dimensionality: int = 4


@dataclass
class OracleOpenActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will open the given articulated object
    """
    type: str = "OracleOpenAction"
    name: str = "oracle_open_action"
    dimensionality: int = 4


@dataclass
class OracleCloseActionConfig(ActionConfig):
    r"""
    In Rearrangement tasks only, the action that will close the given articulated object
    """
    type: str = "OracleCloseAction"
    name: str = "oracle_close_action"
    dimensionality: int = 4


@dataclass
class OraclePowerOnActionConfig(ActionConfig):
    r"""
    Action that will modify the 'is_powered_on' object state of an object to 1
    """
    type: str = "OraclePowerOnAction"
    name: str = "oracle_power_on_action"
    dimensionality: int = 2


@dataclass
class OraclePowerOffActionConfig(ActionConfig):
    r"""
    Action that will modify the 'is_powered_on' object state of an object to 0
    """
    type: str = "OraclePowerOffAction"
    name: str = "oracle_power_off_action"
    dimensionality: int = 2


@dataclass
class OracleCleanActionConfig(ActionConfig):
    r"""
    Action that will modify the 'is_clean' object state of an object to 1
    """
    type: str = "OracleCleanAction"
    name: str = "oracle_clean_action"
    dimensionality: int = 2


@dataclass
class OracleFillActionConfig(ActionConfig):
    r"""
    Action that will modify the 'is_filled' object state of an object to 1
    """
    type: str = "OracleFillAction"
    name: str = "oracle_fill_action"
    dimensionality: int = 2


@dataclass
class OraclePourActionConfig(ActionConfig):
    r"""
    Action that will modify the 'is_filled' object state of an object to 1
    if an object is held which is already filled
    """
    type: str = "OraclePourAction"
    name: str = "oracle_pour_action"
    dimensionality: int = 2


############################################################
# Register your actions below
############################################################

ALL_ACTIONS: List[ActionConfig] = [
    OraclePickActionConfig,
    OraclePlaceActionConfig,
    OracleOpenActionConfig,
    OracleCloseActionConfig,
    OraclePowerOnActionConfig,
    OraclePowerOffActionConfig,
    TeleportActionConfig,
    OracleCleanActionConfig,
    OracleFillActionConfig,
    OraclePourActionConfig,
]


def register_actions(conf):
    with habitat.config.read_write(conf):
        for conf_agent in conf.evaluation.agents.values():
            agent_uid = conf_agent["uid"]
            for action_config in ALL_ACTIONS:
                ActionConfig = action_config()
                ActionConfig.agent_index = agent_uid
                conf.habitat.task.actions[
                    f"agent_{agent_uid}_{ActionConfig.name}"
                ] = ActionConfig


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.actions.humanoid_base_velocity",
    group="habitat/task/actions",
    name="humanoid_base_velocity",
    node=HumanoidBaseVelocityActionConfig,
)


cs.store(
    package="habitat.task.actions.teleport",
    group="habitat/task/actions",
    name="teleport",
    node=TeleportActionConfig,
)
