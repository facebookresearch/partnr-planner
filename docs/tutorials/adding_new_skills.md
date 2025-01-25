# Adding a new low-level skills

This example shows how to add new low-level motor skills for the robot agent (pointnav, pick, and place). All motor skills require defining [Sensor](habitat_llm/agent/env/sensors.py), [ArticulatedAgentAction](habitat_llm/agent/env/actions.py), and [SkillPolicy](habitat_llm/tools/motor_skills/skill.py), or [NnSkillPolicy](habitat_llm/tools/motor_skills/nn_skill.py).

## Add a new Sensor
We can add a new sensor by extending the Sensor class. Below is an example of adding a sensor to query the robot's end effector pose.
```python
import numpy as np
import quaternion
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface

@registry.register_sensor
class EEPoseSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pose"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPoseSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(7,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        ee_T_base = self._sim.get_agent_data(self.agent_id).articulated_agent.ee_transform()
        ee_pos = ee_T_base.translation
        ee_quat = quaternion.from_rotation_matrix(ee_T_base.rotation())

        return np.array([*ee_pos, *ee_quat], dtype=np.float32)

@dataclass
class EEPoseSensorConfig(LabSensorConfig):
    r"""
    Rearrangement only. The cartesian coordinates and rotation (7 floats) of the arm's end effector in the frame of reference of the robot's base.
    """

    type: str = "EEPoseSensor"

# Add the sensor
ALL_SENSORS = [
    DynamicNavGoalPointGoalSensorConfig,
    DynamicTargetStartSensorConfig,
    DynamicTargetGoalSensorConfig,
    NavGoalPointGoalSensorConfig,
    EEPoseSensorConfig,
]
```

## Add a new Action
We can add a new action by extending the ArticulatedAgentAction class. Below is an example of adding an action space which uses end-effector control.

```python
@registry.register_task_action
class ArmEEAction(ArticulatedAgentAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm."""

    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        self.ee_target: Optional[np.ndarray] = None
        self.ee_index: Optional[int] = 0
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self._ik_helper.calc_fk(
            np.array(self._sim.articulated_agent.arm_joint_pos)
        )

        self.ee_target = cur_ee

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-1, high=1, dtype=np.float32)

    def get_T_matrix(self, position, rotation=None):
        T_matrix = np.eye(4)
        T_matrix[:3, 3] = position
        if rotation is not None:
            T_matrix[:3, :3] = R.from_euler("xyz", rotation).as_matrix()
        return T_matrix

    def step(self, ee_pose, **kwargs):
        # Get the current end effector pose
        self.ee_target, self.ee_rot_target = self._ik_helper.calc_fk(
            np.array(self.cur_articulated_agent.arm_joint_pos)
        )
        T_curr = self.get_T_matrix(self.ee_target, self.ee_rot_target)

        # Get the desired delta end effector pose
        T_delta = self.get_T_matrix(ee_pose[:3], ee_pose[3:])

        # Calculate the new end effector pose
        T_new = np.dot(T_curr, T_delta)
        self.ee_target = T_new[:3, 3]
        self.ee_rot_target = self.get_euler_from_matrix(T_new)

        # Ensure that the IK helper's joints match the current articulate agent's joints
        joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)
        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        # Use IK to calculate the joint positions to reach the desired EE pose
        des_joint_pos = self._ik_helper.calc_ik(
            self.ee_target, self.ee_rot_target
        )
        des_joint_pos = list(des_joint_pos)

        # Set the articulate agent's joints to the desired values
        if self._sim.habitat_config.kinematic_mode:
            self.set_joint_pos_kinematic(des_joint_pos)
            self.cur_articulated_agent.arm_joint_pos = des_joint_pos
            self.cur_articulated_agent.fix_joint_values = des_joint_pos
        else:
            self.cur_articulated_agent.arm_motor_pos = des_joint_pos
```

## Add a new Skill class
If needed, you can modify the [SkillPolicy](habitat_llm/tools/motor_skills/skill.py), or [NnSkillPolicy](habitat_llm/tools/motor_skills/nn_skill.py) to override any methods as necessary (i.e., different reset conditions, modification to the `_internal_act`, etc.). In this example, we follow the [PlaceSkillPolicy](habitat_llm/tools/motor_skills/place/nn_place_skill.py), and do not need to modify any methods.

## Modify the Hydra Configs
In order to use this new skill, we can modify the hydra configuration to add the sensor, action, and skill.

### Add the Sensor to the config
The agent's sensors are defined in [habitat_llm/conf/habitat_conf/task/rearrange_easy_multi_agent_nn.yaml](habitat_llm/conf/habitat_conf/task/rearrange_easy_multi_agent_nn.yaml). We can modify this config to use the EEPoseSensor sensor we defined earlier.
```yaml
    - /habitat/task/lab_sensors:
        - relative_resting_pos_sensor
        - target_start_sensor
        - goal_sensor
        - joint_sensor
        - end_effector_sensor
        - is_holding_sensor
        - end_effector_pose_sensor
        - target_start_gps_compass_sensor
        - target_goal_gps_compass_sensor
    - _self_
```
### Add the Action to the config
The agent's action space is defined in [habitat_llm/conf/habitat_conf/task/rearrange_easy_multi_agent_nn.yaml](habitat_llm/conf/habitat_conf/task/rearrange_easy_multi_agent_nn.yaml). We can modify this config to use the ArmEEAction action space we defined earlier.
```yaml
  agent_0_arm_action:
    grip_controller: MagicGraspAction
    type: "ArmAction"
    arm_controller: "ArmEEAction"
    arm_joint_mask: [1,1,0,1,0,1,1]
    arm_joint_dimensionality: 5
    arm_joint_limit: [[-1.5708,1.5708],[-3.1415,0.0000],[0,3.1415],[-1.5708,1.5708],[-1.5708,1.5708]]
    ee_ctrl_lim: 0.015
    ee_rot_ctrl_lim: 0.015
```

### Add the Skill to the config
The agent's motor skill configs are defined in [habitat_llm/conf/agent/nn_rearrange_agent_motortoolsonly.yaml](habitat_llm/conf/agent/nn_rearrange_agent_motortoolsonly.yaml). To use a new motor skill (for an example, a new place policy), we can copy [habitat_llm/conf/tools/motor_skills/nn_place.yaml](habitat_llm/conf/tools/motor_skills/nn_place.yaml) and create a new config file, for an example `nn_place_ee.yaml`. The main configs to modify here are the checkpoint to load, the observation space, and the action space:
```yaml
    load_ckpt_file             : "data/models/skills/place/place_latest_sample_ee.pth"
    obs_space                  : ['obj_goal_sensor', 'articulated_agent_jaw_depth', 'ee_pose', 'is_holding']
    action_space               : ["arm_action", "base_velocity", "empty"]
```
After creating this new `nn_place_ee.yaml` config file, we can modify the original motor config file to include this new place skill [habitat_llm/conf/agent/nn_rearrange_agent_motortoolsonly.yaml](habitat_llm/conf/agent/nn_rearrange_agent_motortoolsonly.yaml).
```yaml
  defaults:

  - /tools/motor_skills@tools.motor_skills:
    # This yaml will be sequentially changed to use
    # neural network skills
    - nn_nav # neural network point nav skill
    - nn_pick # neural network pick skill
    - nn_place_ee # neural network place skill with end effector control
    - nn_open # neural network open skill
    - nn_close # neural network close skill
    - nn_rearrange # neural network rearrange skill
    - nn_explore # explore skill that calls neural network point nav skill
    - wait

env: habitat
```

### Run using LLM planner + skills
We can run the new skills with the LLM planner by using:
```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/decentralized_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct \
    evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct \
    device=cpu \
    agent@evaluation.agents.agent_0.config=nn_rearrange_agent_motortoolsonly \
    habitat_conf/task=rearrange_easy_multi_agent_nn \
```
