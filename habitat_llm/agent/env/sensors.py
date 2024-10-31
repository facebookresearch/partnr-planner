# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import habitat
import numpy as np
from gym import spaces
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import MultiObjSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat.tasks.utils import cartesian_to_polar


@registry.register_sensor
class DynamicNavGoalPointGoalSensor(UsesArticulatedAgentInterface, Sensor):
    """
    GPS and compass sensor relative to the starting object position or goal
    position.
    """

    cls_uuid: str = "dynamic_goal_to_agent_gps_compass"

    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return DynamicNavGoalPointGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        target = self._sim.dynamic_target
        transform = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        dir_vector = transform.inverted().transform_point(target)
        rho, phi = cartesian_to_polar(dir_vector[0], dir_vector[1])
        return np.array([rho, -phi], dtype=np.float32)


@registry.register_sensor
class NavGoalPointGoalSensor(UsesArticulatedAgentInterface, Sensor):
    """
    GPS and compass sensor relative to the starting object position or goal
    position. This is for point nav skill
    """

    cls_uuid: str = "goal_to_agent_gps_compass"

    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return NavGoalPointGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        target = self._sim.dynamic_target
        transform = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        dir_vector = transform.inverted().transform_point(target)
        rho, phi = cartesian_to_polar(dir_vector[0], dir_vector[1])
        return np.array([rho, -phi], dtype=np.float32)


@dataclass
class DynamicNavGoalPointGoalSensorConfig(LabSensorConfig):
    type: str = "DynamicNavGoalPointGoalSensor"


@dataclass
class NavGoalPointGoalSensorConfig(LabSensorConfig):
    type: str = "NavGoalPointGoalSensor"


@registry.register_sensor
class DynamicTargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "dynamic_obj_start_sensor"

    def _get_uuid(self, *args, **kwargs):
        return DynamicTargetStartSensor.cls_uuid

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim

        target = self._sim.dynamic_target
        transform = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        dir_vector = transform.inverted().transform_point(target)
        return np.array(dir_vector, dtype=np.float32).reshape(-1)


@dataclass
class DynamicTargetStartSensorConfig(LabSensorConfig):
    type: str = "DynamicTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@registry.register_sensor
class DynamicTargetGoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "dynamic_obj_goal_sensor"

    def _get_uuid(self, *args, **kwargs):
        return DynamicTargetGoalSensor.cls_uuid

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        target = self._sim.dynamic_target
        transform = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        dir_vector = transform.inverted().transform_point(target)
        return np.array(dir_vector, dtype=np.float32).reshape(-1)


@dataclass
class DynamicTargetGoalSensorConfig(LabSensorConfig):
    type: str = "DynamicTargetGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


ALL_SENSORS = [
    DynamicNavGoalPointGoalSensorConfig,
    DynamicTargetStartSensorConfig,
    DynamicTargetGoalSensorConfig,
    NavGoalPointGoalSensorConfig,
]


SENSOR_MAPPINGS = {
    "dynamic_goal_to_agent_gps_compass": "goal_to_agent_gps_compass",
    "dynamic_obj_start_sensor": "obj_start_sensor",
    "dynamic_obj_goal_sensor": "obj_goal_sensor",
}


def register_sensors(conf):
    with habitat.config.read_write(conf):
        for sensor_config in ALL_SENSORS:
            SensorConfig = sensor_config()
            conf.habitat.task.lab_sensors[SensorConfig.type] = SensorConfig


def remove_visual_sensors(conf):
    with habitat.config.read_write(conf):
        conf.habitat.gym.obs_keys = [
            gkey for gkey in conf.habitat.gym.obs_keys if "rgb" not in gkey
        ]
        for agent in conf.habitat.simulator.agents:
            sim_sensor = conf.habitat.simulator.agents[agent].sim_sensors
            updated_sim_sensors = {
                sensor_name: sensor
                for sensor_name, sensor in sim_sensor.items()
                if "rgb" not in sensor_name
            }
            conf.habitat.simulator.agents[agent].sim_sensors = updated_sim_sensors
