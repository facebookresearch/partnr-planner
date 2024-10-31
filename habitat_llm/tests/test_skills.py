#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

from habitat_llm.agent.env.dataset import CollaborationDatasetV0

# append the path of the
# parent directory
sys.path.append("..")

import re

import habitat.sims.habitat_simulator.sim_utilities as sutils
import magnum as mn
import numpy as np
import pytest
from habitat.sims.habitat_simulator.object_state_machine import set_state_of_obj
from hydra import compose, initialize
from hydra.utils import instantiate

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.utils import setup_config
from habitat_llm.utils.sim import ee_distance_to_object, find_receptacles, init_agents
from habitat_llm.world_model import Furniture, Receptacle

seed = 47668090
config_path = "examples/planner_multi_agent_demo_config.yaml"

# CLI for local test: MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m pytest habitat_llm/tests/test_skills.py


def check_arm_reset(llm_env):
    """Check if the arm is being reset"""
    robot_id = 0
    arm_joint_pos = llm_env.env_interface.sim.agents_mgr[
        robot_id
    ].articulated_agent.arm_joint_pos
    target_arm_joint_pos = np.array([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])
    distance_between_target_and_current_joint_pos_threshold = 0.25  # in radian
    return (
        np.linalg.norm(arm_joint_pos - target_arm_joint_pos)
        < distance_between_target_and_current_joint_pos_threshold
    )


def check_nav_skill_done(llm_env):
    """Check if the navigation skill is actually finished"""
    robot_id = 0
    robot_pos = llm_env.env_interface.sim.agents_mgr[
        robot_id
    ].articulated_agent.base_pos
    target_pos = mn.Vector3(llm_env.env_interface.sim.dynamic_target)
    distance_between_agent_and_target_threshold = 1.75  # in meter
    return (
        robot_pos - target_pos
    ).length() < distance_between_agent_and_target_threshold


def check_pick_skill_done(llm_env):
    """Check if the pick skill is actually finished"""
    robot_id = 0
    # Get the ee location
    cur_ee_pos = (
        llm_env.env_interface.sim.agents_mgr[robot_id]
        .articulated_agent.ee_transform()
        .translation
    )

    # Get the target object
    entity = llm_env.env_interface.world_graph[robot_id].get_node_from_name("cushion_0")

    rom = llm_env.env_interface.sim.get_rigid_object_manager()
    target_cur_pos = rom.get_object_by_handle(entity.sim_handle).translation
    distance_between_target_and_current_object_threshold = 0.01  # in meter
    return (
        cur_ee_pos - target_cur_pos
    ).length() < distance_between_target_and_current_object_threshold and check_arm_reset(
        llm_env
    )


def check_place_skill_done(llm_env):
    """Check if the place skill is actually finished"""
    robot_id = 0
    # Get the target object location
    target_place_pos = mn.Vector3(llm_env.env_interface.sim.dynamic_target)

    # Get the target object
    entity = llm_env.env_interface.world_graph[robot_id].get_node_from_name("cushion_0")

    rom = llm_env.env_interface.sim.get_rigid_object_manager()
    target_cur_pos = rom.get_object_by_handle(entity.sim_handle).translation
    distance_between_target_and_current_object_threshold = 1.5
    return (
        target_place_pos - target_cur_pos
    ).length() < distance_between_target_and_current_object_threshold and check_arm_reset(
        llm_env
    )


def get_receptacle_and_joint_idx(sim, receptacle_name, aom):
    """
    This method fetches the receptacle and joint indices provided the receptacle name. "receptacle_name" can be parent object (Fridge) or any child surface (fridge_shelf_2). This method checks if the object referenced by "receptacle_name" is the parent object or a child surface and returns the joint indices accordingly.
    """
    for r in find_receptacles(sim):
        if receptacle_name == r.parent_object_handle:
            # If "receptacle_name" is parent
            rec = aom.get_object_by_handle(receptacle_name)
            joint_idx = list(range(len(rec.joint_positions)))
            return rec, joint_idx
    return None, None


def check_open_skill_done(llm_env):
    """Check if the open skill is actually finished"""
    target_object_name = "2d93837fd0fb80b2f2e9b0d3d55eb6506b22d4c1_:0000"
    # Get sim
    sim = llm_env.env_interface.sim
    ao = sutils.get_obj_from_handle(sim, target_object_name)
    default_link = sutils.get_ao_default_link(ao, compute_if_not_found=True)

    # The drawer is in the open state
    assert sutils.link_is_open(ao, default_link, threshold=0.01)
    # The arm is being reset
    assert check_arm_reset(llm_env)
    return True


def check_close_skill_done(llm_env):
    """Check if the close skill is actually finished"""
    target_object_name = "2d93837fd0fb80b2f2e9b0d3d55eb6506b22d4c1_:0000"
    # Get sim
    sim = llm_env.env_interface.sim
    ao = sutils.get_obj_from_handle(sim, target_object_name)
    default_link = sutils.get_ao_default_link(ao, compute_if_not_found=True)

    # The drawer is in the open state
    assert sutils.link_is_closed(ao, default_link, threshold=0.01)
    # The arm is being reset
    assert check_arm_reset(llm_env)
    return True


def get_object_state(llm_env, target_name, object_state):
    robot_id = 0
    entity = llm_env.env_interface.world_graph[robot_id].get_node_from_name(target_name)
    handle = entity.sim_handle
    sim = llm_env.env_interface.sim
    state_dict = sim.object_state_machine.get_snapshot_dict(sim)
    return state_dict[object_state][handle]


def set_object_state(llm_env, target_name, object_state, status):
    robot_id = 0
    entity = llm_env.env_interface.world_graph[robot_id].get_node_from_name(target_name)
    rigid_object = (
        llm_env.env_interface.sim.get_rigid_object_manager().get_object_by_handle(
            entity.sim_handle
        )
    )
    set_state_of_obj(rigid_object, object_state, status)


def execute_skill(high_level_skill_actions, llm_env):
    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    skill_name = high_level_skill_actions[0][0]

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        low_level_actions, responses = llm_env.process_high_level_actions(
            high_level_skill_actions, observations
        )

        assert (
            len(low_level_actions) > 0
        ), f"No low level actions returned. Response: {responses.values()}"

        # Check if the agent finishes
        if any(responses.values()):
            skill_done = True

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)
        # Increase steps
        skill_steps += 1
    return responses, {"skill_steps": skill_steps}


def check_skills_execution(high_level_skill_actions, llm_env):
    """Skill execution"""
    responses, _ = execute_skill(high_level_skill_actions, llm_env)

    skill_name = high_level_skill_actions[0][0]
    # Check the responses
    assert responses == {
        0: "Successful execution!"
    }, f"{skill_name} skill fails. Responses: {responses}"
    # Check skill goal satisfaction
    error_msg = f"{skill_name} skill does not satisfy the goal."
    if skill_name == "Navigate":
        assert check_nav_skill_done(llm_env), error_msg
    elif skill_name == "Pick":
        assert check_pick_skill_done(llm_env), error_msg
    elif skill_name == "Place":
        assert check_place_skill_done(llm_env), error_msg
    elif skill_name == "Open":
        assert check_open_skill_done(llm_env), error_msg
    elif skill_name == "Close":
        assert check_close_skill_done(llm_env), error_msg
    else:
        assert 0, f"Skill {skill_name} is not supported."


@pytest.mark.parametrize(
    "skill_config",
    [
        "centralized_evaluation_runner_multi_agent_nn_skills",
        "centralized_evaluation_runner_multi_agent",
    ],
)
def test_neural_network_and_oracle_skills(skill_config):
    """Test for neural network/oracle skills: point nav, pick, and place skills.
    We will do nav, pick, nav, and place with pre-defined action sequences.
    """
    robot_id = 0
    # Set up hydra config
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_path,
            overrides=[
                f"evaluation={skill_config}",  # We test neural network/oracle skills
                "device=cpu",  # We test cpu version of the skills
                "habitat_conf/task=rearrange_easy_multi_agent_nn",  # We use the action space for nn skills
                "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",  # We change the config to human 0 since only human 0 in the CI testing dataset
                "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",  # We change the config to human 0 since only human 0 in the CI testing dataset
                "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",  # We test with a specific dataset
                "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
                "habitat.environment.iterator_options.shuffle=False",  # We do not shuffle the dataset
                "llm@evaluation.planner.plan_config.llm=mock",
            ],
        )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, seed)

    # Set up habitat config env
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Set up the env
    env_interface = EnvironmentInterface(config)

    # Initialize the base environment interface
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)

    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()

    ##########################################
    ###       Open Drawer Skill Test       ###
    ##########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)
    # Do navigation to the drawer
    drawer_handle = "2d93837fd0fb80b2f2e9b0d3d55eb6506b22d4c1_:0000"
    drawer_name = (
        env_interface.world_graph[robot_id].get_node_from_sim_handle(drawer_handle).name
    )
    cushion_handle = "pillow_9_:0000"
    cushion_name = (
        env_interface.world_graph[robot_id]
        .get_node_from_sim_handle(cushion_handle)
        .name
    )
    chair_handle = "7d8617011c72509329256bd40de9ec5547688286_:0000"
    chair_name = (
        env_interface.world_graph[robot_id].get_node_from_sim_handle(chair_handle).name
    )

    high_level_nav_actions = {0: ("Navigate", drawer_name, None)}
    check_skills_execution(high_level_nav_actions, planner)

    high_level_open_actions = {0: ("Open", drawer_name, None)}
    check_skills_execution(high_level_open_actions, planner)

    ##########################################
    ###       Close Drawer Skill Test      ###
    ##########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)s
    high_level_close_actions = {0: ("Close", drawer_name, None)}
    check_skills_execution(high_level_close_actions, planner)

    #########################################
    ### First Point Navigation Skill Test ###
    #########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)
    high_level_nav_actions = {0: ("Navigate", cushion_name, None)}
    # Ideally, it should be finished in 244 steps for NN skills
    check_skills_execution(high_level_nav_actions, planner)

    #########################################
    ###           Pick Skill Test         ###
    #########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)
    high_level_pick_actions = {0: ("Pick", cushion_name, None)}
    # Ideally, it should be finished in 55 steps for NN skills
    check_skills_execution(high_level_pick_actions, planner)

    ##########################################
    ### Second Point Navigation Skill Test ###
    ##########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)
    high_level_nav_actions = {0: ("Navigate", chair_name, None)}
    # Ideally, it should be finished in 192 steps for NN skills
    check_skills_execution(high_level_nav_actions, planner)

    ##########################################
    ###          Place Skill Test          ###
    ##########################################
    # Pre-define the skill execution sequence for the robot agent (agent 0)
    high_level_place_actions = {
        0: ("Place", f"{cushion_name},on,{chair_name},none,none", None)
    }
    # Ideally, it should be finished in 54 steps for NN skills
    check_skills_execution(high_level_place_actions, planner)

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface


@pytest.mark.parametrize(
    "skill_config",
    [
        "centralized_evaluation_runner_multi_agent_nn_skills",
        "centralized_evaluation_runner_multi_agent",
    ],
)
def test_object_state_skills(skill_config):
    """Test for neural network/oracle skills: point nav, pick, and place skills.
    We will do nav, pick, nav, and place with pre-defined action sequences.
    """
    robot_id = 0
    # Set up hydra config
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_path,
            overrides=[
                f"evaluation={skill_config}",
                "device=cpu",
                "habitat_conf/task=rearrange_easy_multi_agent_nn",
                "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",
                "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",
                "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",
                "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
                "habitat.environment.iterator_options.shuffle=False",
                "llm@evaluation.planner.plan_config.llm=mock",
                "+agent@evaluation.agents.agent_0.config=../../agent/oracle_rearrange_object_states_agent",
            ],
        )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, seed)

    # Set up habitat config env
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Set up the env
    env_interface = EnvironmentInterface(config)

    # Initialize the base environment interface
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)

    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()

    # Check that the object states are appearing in the find object tool results
    object_list = (
        planner.agents[0].get_tool_from_name("FindObjectTool")._get_object_list()
    )
    expected_pattern = r"cushion_\d+ is in/on chair_\d+ and \d+.\d+ meters away from the agent in bedroom_2. It has the following states: is_clean: False"
    assert re.search(expected_pattern, object_list) is not None
    expected_pattern = r"toy_construction_set_\d+ is in/on table_\d+ and \d+.\d+ meters away from the agent in bedroom_2. It has the following states: is_clean: True"
    assert re.search(expected_pattern, object_list) is not None

    # Check that faucets are appearing in the receptacle tool results
    rec_list = (
        planner.agents[0]
        .get_tool_from_name("FindReceptacleTool")
        ._get_receptacles_list()
    )
    expected_pattern = r"cabinet_\d+ in kitchen with components: faucet"
    assert re.search(expected_pattern, rec_list)

    def distance_to(target_name, env_interface=env_interface):
        robot_id = 0
        handle = (
            env_interface.world_graph[robot_id]
            .get_node_from_name(target_name)
            .sim_handle
        )

        distance = ee_distance_to_object(
            env_interface.sim, env_interface.sim.agents_mgr, 0, handle, max_distance=100
        )
        # set distance to 100 when occluded for testing
        if distance is None:
            distance = 100
        return distance

    # add washer dryers able to be powered for this test
    power_state = env_interface.sim.object_state_machine.active_states[0]
    power_state.accepted_semantic_classes.append("washer_dryer")
    env_interface.sim.object_state_machine.initialize_object_state_map(
        env_interface.sim
    )

    start_position = env_interface.sim.agents_mgr[0].articulated_agent.base_pos
    washer_handle = "1c19a987e1bd8c206353d18066b117cc0b21e841_:0001"
    washer_name = (
        env_interface.world_graph[robot_id].get_node_from_sim_handle(washer_handle).name
    )
    cushion_handle = "pillow_9_:0000"
    cushion_name = (
        env_interface.world_graph[robot_id]
        .get_node_from_sim_handle(cushion_handle)
        .name
    )
    object_distance_threshold = 1.5

    set_object_state(planner, cushion_name, "is_clean", False)
    assert not get_object_state(planner, cushion_name, "is_clean")

    # Test the clean action

    high_level_nav_action = {0: ("Navigate", cushion_name, None)}
    high_level_close_actions = {0: ("Clean", cushion_name, None)}
    # the agent starts far from the cushion
    assert distance_to(cushion_name) > object_distance_threshold
    result, _ = execute_skill(high_level_nav_action, planner)
    result, _ = execute_skill(high_level_close_actions, planner)
    # It moves close
    assert distance_to(cushion_name) < object_distance_threshold
    # and successfully changes the object state
    assert get_object_state(planner, cushion_name, "is_clean")
    # the agent should not be able to power on the cushion
    result, _ = execute_skill({0: ("PowerOn", cushion_name, None)}, planner)

    # Check failures for skills on objects which do not afford the manipulated state
    assert (
        result[0]
        == "Unexpected failure! - The targeted object does not afford the state: is_powered_on"
    )
    result, _ = execute_skill({0: ("Fill", cushion_name, None)}, planner)
    assert (
        result[0]
        == "Unexpected failure! - The targeted object does not afford the state: is_filled"
    )
    result, _ = execute_skill({0: ("Pour", cushion_name, None)}, planner)
    assert (
        result[0]
        == "Unexpected failure! - The targeted object does not afford the state: is_filled"
    )

    # Test the oracle power on action
    high_level_close_actions = {0: ("PowerOn", washer_name, None)}

    # Check that the current duration is set
    assert planner.agents[0].get_tool_from_name("PowerOn").skill.current_duration == 2
    # change the desired range for the next invocation of the skill
    planner.agents[0].get_tool_from_name("PowerOn").skill.duration_range = (
        5,
        10,
    )
    # The power on should fail because the agent is too far
    result, info = execute_skill(high_level_close_actions, planner)
    # Check that the duration is taken into account
    assert info["skill_steps"] == 2
    # now the current duration should be sampled from the range we set above. This happens now
    # because reset is called immediately after the skill ends, instead of on the first step
    desired_duration = (
        planner.agents[0].get_tool_from_name("PowerOn").skill.current_duration
    )
    assert desired_duration >= 5
    result, info = execute_skill(high_level_close_actions, planner)
    # Check that the duration is respected
    assert info["skill_steps"] == desired_duration

    # Test that response string does not indicate success
    assert "success" not in result[0].lower()
    assert not get_object_state(
        planner, washer_name, "is_powered_on"
    ), "Agent should be too far away"

    # now nav to the washer and power on
    high_level_nav_action = {0: ("Navigate", washer_name, None)}
    high_level_close_actions = {0: ("PowerOn", washer_name, None)}
    execute_skill(high_level_nav_action, planner)
    execute_skill(high_level_close_actions, planner)
    assert distance_to(washer_name) < object_distance_threshold
    assert get_object_state(
        planner, washer_name, "is_powered_on"
    ), "Power on skill failed"

    # move the agent away from the object again
    env_interface.sim.agents_mgr[
        0
    ].articulated_agent.base_pos = start_position + np.array([2, 0, 0])
    assert distance_to(washer_name) > object_distance_threshold
    high_level_close_actions = {0: ("PowerOff", washer_name, None)}
    execute_skill(high_level_nav_action, planner)
    execute_skill(high_level_close_actions, planner)
    assert distance_to(washer_name) < object_distance_threshold
    assert not get_object_state(
        planner, washer_name, "is_powered_on"
    ), "Power off skill failed"

    # Switch to the episode with fillable objects to test the fill action
    env_interface.reset_environment()
    start_position = env_interface.sim.agents_mgr[0].articulated_agent.base_pos

    # Check that the other object states are appearing in the object list
    object_list = (
        planner.agents[0].get_tool_from_name("FindObjectTool")._get_object_list()
    )
    expected_pattern = r"kettle_\d+ is in/on table_\d+ and \d+.\d+ meters away from the agent in bedroom_2. It has the following states: is_clean: False, is_powered_on: False, is_filled: False"
    assert re.search(expected_pattern, object_list) is not None

    kettle_handle = "6e35f1ab01230ea2687cade5e71e4a3abf07262e_:0000"
    kettle_name = (
        env_interface.world_graph[robot_id].get_node_from_sim_handle(kettle_handle).name
    )
    plant_handle = "Ecoforms_Plant_Saucer_S14MOCHA_:0000"
    plant_name = (
        env_interface.world_graph[robot_id].get_node_from_sim_handle(plant_handle).name
    )

    # make plants fillable
    fill_state = env_interface.sim.object_state_machine.active_states[2]
    fill_state.accepted_semantic_classes.append("plant_saucer")
    env_interface.sim.object_state_machine.initialize_object_state_map(
        env_interface.sim
    )

    # Test the fill and clean at faucet actions
    assert get_object_state(planner, kettle_name, "is_clean") == False
    # the agent starts far from the kettle; -0.5m to ensure > 1.5m distance.
    env_interface.sim.agents_mgr[
        0
    ].articulated_agent.base_pos = start_position + np.array([-0.5, 0, 0])
    assert distance_to(kettle_name) > object_distance_threshold
    high_level_nav_action = {0: ("Navigate", kettle_name, None)}
    high_level_close_actions = {0: ("Fill", kettle_name, None)}

    result, _ = execute_skill(high_level_nav_action, planner)
    result, _ = execute_skill(high_level_close_actions, planner)

    # It moves close
    assert distance_to(kettle_name) < object_distance_threshold
    # it cannot change the state because the faucet is too far
    assert (
        result[0]
        == "Unexpected failure! - The object is not close enough to a water source"
    )
    assert not get_object_state(planner, kettle_name, "is_filled")

    # move the agent away from the object again; -1.25m to ensure > 1.5m distance.
    env_interface.sim.agents_mgr[
        0
    ].articulated_agent.base_pos = start_position + np.array([-1.25, 0, 0])
    assert distance_to(kettle_name) > object_distance_threshold
    # Same should happen for cleaning

    high_level_nav_action = {0: ("Navigate", kettle_name, None)}
    high_level_clean_actions = {0: ("Clean", kettle_name, None)}

    result, _ = execute_skill(high_level_nav_action, planner)
    result, _ = execute_skill(high_level_clean_actions, planner)

    # It moves close
    assert distance_to(kettle_name) < object_distance_threshold
    # it cannot change the state because the faucet is too far
    assert (
        result[0]
        == "Unexpected failure! - Object 373 requires faucet to clean, but agent is not near faucet."
    )
    assert not get_object_state(planner, kettle_name, "is_clean")

    # find the object which contains the faucet
    faucet_object_name = (
        env_interface.world_graph[robot_id]
        .get_node_from_sim_handle("6acafe4f37b17ff0c33d22c8dc3fede1bddc67de_:0000")
        .name
    )
    distance_to(faucet_object_name, env_interface)

    # move the kettle to the faucet
    high_level_close_actions = {
        0: ("Rearrange", f"{kettle_name},on,{faucet_object_name},none,none", None)
    }
    result, _ = execute_skill(high_level_close_actions, planner)

    # now we are close to the faucet
    assert distance_to(faucet_object_name, env_interface) < object_distance_threshold
    high_level_close_actions = {
        0: ("Rearrange", f"{kettle_name},on,{faucet_object_name},none,none", None)
    }

    high_level_clean_actions = {0: ("Clean", kettle_name, None)}
    result, _ = execute_skill(high_level_clean_actions, planner)
    assert get_object_state(planner, kettle_name, "is_clean")
    assert result[0] == "Successful execution!"
    assert (
        env_interface.world_graph[robot_id]
        .get_node_from_name(kettle_name)
        .properties["states"]["is_clean"]
        is True
    )

    high_level_close_actions = {0: ("Fill", kettle_name, None)}
    result = execute_skill(high_level_close_actions, planner)
    # the kettle is able to be filled
    assert get_object_state(planner, kettle_name, "is_filled")
    assert (
        env_interface.world_graph[robot_id]
        .get_node_from_name(kettle_name)
        .properties["states"]["is_filled"]
        is True
    )

    # Test the oracle pour
    result, _ = execute_skill({0: ("Navigate", plant_name, None)}, planner)
    result, info = execute_skill({0: ("Pour", plant_name, None)}, planner)
    assert (
        result[0]
        == "Unexpected failure! - Unable to pour: Agent is not holding an object"
    )

    result, info = execute_skill({0: ("Navigate", kettle_name, None)}, planner)
    result, info = execute_skill({0: ("Open", faucet_object_name, None)}, planner)
    result, info = execute_skill({0: ("Pick", kettle_name, None)}, planner)
    assert result[0] == "Successful execution!"

    # The kettle is filled so we can pour now

    result, _ = execute_skill({0: ("Navigate", plant_name, None)}, planner)
    result, info = execute_skill({0: ("Pour", plant_name, None)}, planner)
    assert result[0] == "Successful execution!"
    assert get_object_state(planner, plant_name, "is_filled")
    assert (
        env_interface.world_graph[robot_id]
        .get_node_from_name(plant_name)
        .properties["states"]["is_filled"]
        is True
    )

    # if the kettle is empty pouring fails
    set_object_state(planner, kettle_name, "is_filled", False)
    result, info = execute_skill({0: ("Pour", plant_name, None)}, planner)
    assert (
        result[0]
        == "Unexpected failure! - Unable to pour: The held object is not filled"
    )

    # Test the cleaning furnitures
    table_name = (
        env_interface.world_graph[robot_id]
        .get_node_from_sim_handle("a858cba39573583f6aff0a31d237bcebabaaf503_:0000")
        .name
    )
    assert (
        env_interface.world_graph[robot_id]
        .get_node_from_name(table_name)
        .properties["states"]["is_clean"]
        is False
    )
    execute_skill({0: ("Navigate", table_name, None)}, planner)
    result, info = execute_skill({0: ("Clean", table_name, None)}, planner)
    assert result[0] == "Successful execution!"
    # The world graph should be updated too
    assert (
        env_interface.world_graph[robot_id]
        .get_node_from_name(table_name)
        .properties["states"]["is_clean"]
        is True
    )

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface


@pytest.mark.parametrize(
    "skill_config",
    [
        "centralized_evaluation_runner_multi_agent_nn_skills",
        "centralized_evaluation_runner_multi_agent",
    ],
)
def test_floors(skill_config):
    """Test for neural network/oracle skills to ensure they interact with floors correctly"""
    robot_id = 0
    # Set up hydra config
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_path,
            overrides=[
                f"evaluation={skill_config}",
                "device=cpu",
                "habitat_conf/task=rearrange_easy_multi_agent_nn",
                "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",
                "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",
                "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",
                "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
                "habitat.environment.iterator_options.shuffle=False",
                "llm@evaluation.planner.plan_config.llm=mock",
                "+agent@evaluation.agents.agent_0.config=../../agent/oracle_rearrange_object_states_agent",
            ],
        )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, seed)

    # Set up habitat config env
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Set up the env; only floor-init episodes
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    dataset.episodes = [ep for ep in dataset.episodes if ep.episode_id in {"2", "3"}]
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Initialize the base environment interface
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)

    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()

    object_list = (
        planner.agents[0].get_tool_from_name("FindObjectTool")._get_object_list()
    )
    # check that the object list reports that the cushion is on the floor
    expected_pattern = r"multiport_hub_\d+ is in/on counter_\d+ and \d+.\d+ meters away from the agent in kitchen_1"
    assert re.search(expected_pattern, object_list) is not None
    expected_pattern = r"cushion_\d+ is in/on floor_kitchen_\d+ and \d+.\d+ meters away from the agent in kitchen_1"
    assert re.search(expected_pattern, object_list) is not None
    cushion_name = "cushion_1"
    multiport_hub_name = "multiport_hub_0"
    counter_name = "counter_22"

    # Move the hub from the counter to the floor
    high_level_close_actions = {
        0: (
            "Rearrange",
            f"{multiport_hub_name},on,floor_living_room_1,none,none",
            None,
        )
    }
    result, _ = execute_skill(high_level_close_actions, planner)
    assert result[0] == "Successful execution!"

    # assert that the cushion is on the floor now
    hub_node = env_interface.world_graph[robot_id].get_node_from_name(
        multiport_hub_name
    )
    neighbors = env_interface.world_graph[robot_id].get_neighbors_of_type(
        hub_node, Furniture
    )
    assert len(neighbors) == 1
    assert neighbors[0].name == "floor_living_room_1"

    # Move the cushion from the floor to the counter
    high_level_close_actions = {
        0: ("Rearrange", f"{cushion_name},on,{counter_name},none,none", None)
    }
    execute_skill(high_level_close_actions, planner)

    # assert that the cushion is on the counter now
    cushion_node = env_interface.world_graph[robot_id].get_node_from_name(cushion_name)
    containers = env_interface.world_graph[robot_id].get_neighbors_of_type(
        cushion_node, Receptacle
    )
    assert len(containers) == 1
    furn = env_interface.world_graph[robot_id].find_furniture_for_receptacle(
        containers[0]
    )
    assert furn.name == counter_name

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface
