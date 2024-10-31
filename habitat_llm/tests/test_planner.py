#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import gc
import os
from typing import Dict
from unittest.mock import Mock

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict
from torch import multiprocessing as mp

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.examples.planner_demo import run_planner
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.sim import init_agents

DATASET_OVERRIDES = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",  # We test with a specific dataset
    "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",  # We change the config to human 0 since only human 0 in the CI testing dataset
    "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",  # We change the config to human 0 since only human 0 in the CI testing dataset
]


def get_config(config_file, overrides):
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_file,
            overrides=overrides,
        )
    HydraConfig().cfg = config
    # emulate a regular hydra initialization
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "outputs/test"
    return config


def setup_env(config):
    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # Initialize the environment interface for the agent
    env = EnvironmentInterface(config)
    return env


def test_oracle_planner_teleport():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "planner@evaluation.planner=dag_centralized_planner",
            "+evaluation.agents.agent_0.config.tools.motor_skills.oracle_rearrange.skill_config.nav_skill_config.teleport=True",
            "+evaluation.agents.agent_1.config.tools.motor_skills.oracle_rearrange.skill_config.nav_skill_config.teleport=True",
            "+evaluation.agents.agent_0.config.tools.motor_skills.oracle_nav.skill_config.teleport=True",
            "+evaluation.agents.agent_1.config.tools.motor_skills.oracle_nav.skill_config.teleport=True",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    max_num_steps = 40
    config = setup_config(config, 0)
    env_interface = setup_env(config)
    oracle_planner_conf = config.evaluation.planner
    # Set the planner
    planner = instantiate(oracle_planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()
    observations = env_interface.get_observations()

    current_instruction = env_interface.env.env.env._env.current_episode.instruction
    plan_info = []
    task_done = False
    cstep = 0

    while not task_done and cstep < max_num_steps:
        low_level_actions, planner_info, task_done = planner.get_next_action(
            current_instruction, observations, env_interface.world_graph
        )
        if task_done:
            continue

        total_actions = len(
            [
                (action_id, action)
                for action_id, action in planner_info["high_level_actions"].items()
                if action[0] != "Wait" and planner_info["replanned"][action_id]
            ]
        )
        cstep += total_actions
        if len(planner_info["high_level_actions"]):
            plan_info.append(copy.deepcopy(planner_info))

        if len(low_level_actions) > 0:
            obs, reward, done, info = env_interface.step(low_level_actions)
            observations = env_interface.parse_observations(obs)

    actions = [step_info["high_level_actions"][0] for step_info in plan_info]
    gt_responses = [
        ("Navigate", "cushion_0", ""),
        ("Pick", "cushion_0", ""),
        ("Navigate", "chair_35", ""),
        ("Place", "cushion_0, on, chair_35, none, none", ""),
        ("Navigate", "toy_construction_set_1", ""),
        ("Pick", "toy_construction_set_1", ""),
        ("Navigate", "table_47", ""),
        ("Wait", "", ""),
        ("Place", "toy_construction_set_1, on, table_47, none, none", ""),
    ]
    for i in range(len(actions)):
        assert actions[i] in gt_responses

    # Make sure the task is successful:
    assert info["task_percent_complete"] == 1.0
    assert info["task_state_success"]

    # Every action should take 4 steps (nav pick nav place)
    assert cstep == 8

    # Should be able to finish the task
    assert task_done

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface
    gc.collect()


def test_oracle_planner():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=["planner@evaluation.planner=dag_centralized_planner"]
        + DATASET_OVERRIDES,
    )
    default_agent_uid = 0

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    max_num_steps = 5000
    config = setup_config(config, 0)
    env_interface = setup_env(config)
    oracle_planner_conf = config.evaluation.planner
    # Set the planner
    planner = instantiate(oracle_planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()
    observations = env_interface.get_observations()

    current_instruction = env_interface.env.env.env._env.current_episode.instruction
    plan_info = []
    task_done = False
    cstep = 0
    while not task_done and cstep < max_num_steps:
        low_level_actions, planner_info, task_done = planner.get_next_action(
            current_instruction, observations, env_interface.world_graph
        )
        if task_done:
            continue
        if len(planner_info["responses"]):
            plan_info.append(copy.deepcopy(planner_info))
        if len(low_level_actions) > 0:
            obs, reward, done, info = env_interface.step(low_level_actions)
            observations = env_interface.parse_observations(obs)

    actions = [step_info["high_level_actions"][0] for step_info in plan_info]

    gt_responses = [
        ("Pick", "cushion_0", ""),
        ("Navigate", "chair_35", ""),
        ("Place", "cushion_0, on, chair_35, none, none", ""),
        ("Navigate", "toy_construction_set_1", ""),
        ("Pick", "toy_construction_set_1", ""),
        ("Navigate", "table_47", ""),
        ("Wait", "", ""),
        ("Place", "toy_construction_set_1, on, table_47, none, none", ""),
    ]

    for i in range(len(actions)):
        assert actions[i] in gt_responses

    # Make sure that the parent of toy_construction_set_1 is changed to table_46 in the world graph
    # TODO: Somehow cushion_0 is snapped into the furniture that is close to chair_34, but not chair_34
    # Possibly due to the fact that there is a wired behavior in the snap down function or sample placement function
    entity = env_interface.world_graph[default_agent_uid].get_node_from_name(
        "toy_construction_set_1"
    )
    parent_of_entity = env_interface.world_graph[
        default_agent_uid
    ].find_furniture_for_object(entity)
    assert parent_of_entity.name == "table_47"

    # Make sure the task is successful:
    assert info["task_percent_complete"] == 1.0
    assert info["task_state_success"]

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface
    gc.collect()


def test_llm_planner():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=["llm@evaluation.planner.plan_config.llm=mock"] + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)
    llm_planner_conf = config.evaluation.planner
    # Set the planner
    planner = instantiate(llm_planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)

    planner.reset()
    observations = env_interface.get_observations()

    current_instruction = env_interface.env.env.env._env.current_episode.instruction

    low_level_actions, planner_info, task_done = planner.get_next_action(
        current_instruction, observations, env_interface.world_graph
    )
    first_prompt = planner.curr_prompt
    first_prompt = first_prompt.split("Here are some examples")[0].strip()
    gt_prompt = """- Overview:
Solve the given multi-agent planning problem as best as you can. The task assigned to you will be situated in a house and will generally involve navigating to objects, picking and placing them on different receptacles to achieve rearrangement. Below is the detailed description of the actions you can use for solving the task. You can assign them to Agent_0 and/or Agent_1 as required.

- Close: Used for closing an articulated entity. You must provide the name of the furniture you want to close. Example (Close[chest_of_drawers_1])
- FindAgentActionTool: Should be used to find current and past state history of other agent.
- FindObjectTool: Used to find the exact name/names of the object/objects of interest. If you want to find names of objects on specific receptacles or furnitures, please include that in the query. Example (FindObjectTool[toys on the floor] or FindObjectTool[apples])
- FindReceptacleTool: Used to know the exact name of a receptacle. A receptacle is a furniture or entity (like a chair, table, bed etc.) where you can place an object. Example (FindReceptacleTool[a kitchen counter])
- FindRoomTool: Used to know the exact name of a room in the house. A room is a region in the house where furniture is placed. Example (FindRoomTool[a room which might have something to eat])
- Navigate: Used for navigating to an entity. You must provide the name of the entity you want to navigate to. Example (Navigate[counter_22])
- Open: Used for opening an articulated entity. You must provide the name of the furniture you want to open. Example (Open[chest_of_drawers_1])
- Pick: Used for picking up an object. You must provide the name of the object to be picked. The agent cannot hold more than one object at a time. Example (Pick[cup_1])
- Place: Used for placing an object on a target location. You need to provide the name of the object to be placed, the name of the furniture where it should be placed, spatial relation ("on" or "within") describing the relation between the object and furniture. The object to be placed must already be held by the agent (i.e. picked previously). In addition to these, you can request to place the object near another object. For that you can optionally provide a spatial constraints ("next_to") and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API template - Place[<object_to_be_placed>, <spatial_relation>, <furniture to be placed on>, <spatial_constraint>, <reference_object>]. spatial_constraint and reference_object should be set to "None" when necessary.
- Rearrange: Used for moving an object from its current location to the target location. You need to provide the name of the object to be moved, the name of the furniture where is should be moved, spatial relation ("on" or "within") describing the relation between the object and furniture. This will automatically pick the specified object and move to the target furniture and attempt to place it. In addition to these, you can request to place the object near another object. For that you can optionally provide a spatial constraints ("next_to") and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API template Rearrange[<object_to_be_moved>, <spatial_relation>, <furniture to be placed on>, <spatial_constraint>, <reference_object>]. spatial_constraint and reference_object should be set to "None" when necessary.
- Wait: Used to make agent stay idle for some time. Example (Wait[])"""

    assert first_prompt == gt_prompt

    # Destroy envs
    env_interface.env.close()
    del planner
    del env_interface
    gc.collect()


def test_thoughtless_llm_planner():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation=decentralized_evaluation_runner_multi_agent",
            "planner@evaluation.agents.agent_0.planner=llm_decentralized_thoughtless_planner",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
            "planner@evaluation.agents.agent_1.planner=llm_decentralized_thoughtless_planner",
            "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)
    config.evaluation.agents.agent_0.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }
    config.evaluation.agents.agent_1.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }

    # Set the planner
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    observations = env_interface.get_observations()
    planner0 = eval_runner.planner[0]
    planner1 = eval_runner.planner[1]
    planner0.llm.generate = Mock(return_value="Navigate[chair_17]")
    planner1.llm.generate = Mock(return_value="Navigate[table_30]")
    expected_base_prompt = """Solve the given multi-agent planning problem as best as you can. The task assigned to you will be situated in a house and will generally involve navigating to objects, picking and placing them on different receptacles to achieve rearrangement. Below is the detailed description of the actions you can use for solving the task.

Task: test instruction

Current Environment:
Furniture:
living_room_1: floor_living_room_1, table_10, couch_18, shelves_26, chair_28, chair_29, table_32, chair_33, chair_34, chair_35, chair_36, table_46, table_47, stand_55, chest_of_drawers_56
other_room_1: floor_other_room_1
bathroom_1: floor_bathroom_1, chair_23, table_37, unknown_38, bench_48
bathroom_2: floor_bathroom_2, unknown_14, table_50
bedroom_1: floor_bedroom_1, chair_16, table_22, bed_45, bench_49, chest_of_drawers_52, chest_of_drawers_54
laundryroom_1: floor_laundryroom_1, washer_dryer_11, washer_dryer_12, shelves_15
entryway_1: floor_entryway_1, bench_51
bedroom_2: floor_bedroom_2, table_13, chair_17, bed_21, table_24, table_25, chair_27, stool_31, bench_44, table_59, chest_of_drawers_61
hallway_1: floor_hallway_1, table_30
kitchen_1: floor_kitchen_1, counter_19, chair_20, cabinet_39, cabinet_40, chair_41, chair_42, chair_43, counter_53, cabinet_57, fridge_58, unknown_60

The following furnitures have a faucet: cabinet_57
Objects:
cushion_0: chair_17
toy_construction_set_1: table_25

Previous actions:
"""
    end_token = "<end_act>"

    trailing_prompt = "\n\nNext Agent_Action:<|reserved_special_token_0|>"

    # get actions through the eval runner
    _, planner_info, _ = eval_runner.get_low_level_actions(
        "test instruction", observations, env_interface.world_graph
    )
    print(planner0.llm.generate.call_args[0][0])

    # check that the planners got called with the correct prompts
    planner0.llm.generate.assert_called_once_with(
        expected_base_prompt + "No previous actions taken" + trailing_prompt, end_token
    )
    planner1.llm.generate.assert_called_once_with(
        expected_base_prompt + "No previous actions taken" + trailing_prompt, end_token
    )

    planner_info["sim_step_count"] = 1
    # update the action history with the planner results
    eval_runner.update_agent_state_history(planner_info)
    eval_runner.update_agent_action_history(planner_info)

    # force replan
    planner0.replan_required = True
    planner1.replan_required = True

    # when calling for low level actions again, the action history should be updated
    eval_runner.get_low_level_actions(
        "test instruction", observations, env_interface.world_graph
    )

    # check that the planners got called with the new correct prompts including the action history
    action_history_0 = "Agent_Action: Navigate[chair_17]\nAction Result: "
    action_history_1 = "Agent_Action: Navigate[table_30]\nAction Result: "
    planner0.llm.generate.assert_called_with(
        expected_base_prompt + action_history_0 + trailing_prompt, end_token
    )
    planner1.llm.generate.assert_called_with(
        expected_base_prompt + action_history_1 + trailing_prompt, end_token
    )

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()


def test_react_based_llm_planner_rag_format():
    default_agent_uid = 0
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation='decentralized_evaluation_runner_multi_agent'",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
            "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
            "instruct@evaluation.agents.agent_0.planner.plan_config.instruct=few_shot_decentralized_partial_obs_coordinated_robot_spatial",
            "instruct@evaluation.agents.agent_1.planner.plan_config.instruct=few_shot_decentralized_partial_obs_coordinated_human_spatial",
            "evaluation.agents.agent_0.planner.plan_config.enable_rag=True",  # Enable RAG
            "evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/test_rag/react_based_rag_dataset/]",  # Set the RAG path
            "evaluation.agents.agent_0.planner.plan_config.rag_data_source_name=[2024_08_01_train_mini.json.gz]",  # Set the RAG dataset source
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)
    config.evaluation.agents.agent_0.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }
    config.evaluation.agents.agent_1.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }

    # Set the planner
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    planner0 = eval_runner.planner[0]

    # Assert to see if the expected example prompt included in prompt_example
    prompt_example = planner0.instruct.prompt
    expected_example_prompt = "Exit!\n{eot_tag}\n"
    assert expected_example_prompt in prompt_example

    # Assert to see the location of the prompt
    example_prompt_index = prompt_example.rfind(expected_example_prompt)
    expected_example_prompt_index = 10969
    assert example_prompt_index == expected_example_prompt_index

    # Assert if RAG's dataset
    assert len(planner0.rag.data_dict) == 63

    # Assert if RAG's dataset's contain trace
    assert all(
        len(planner0.rag.data_dict[index]["trace"]) != 0
        for index in planner0.rag.data_dict
    )

    # Assert if RAG can get the correct text
    test_instruction = "Help me prepare the bedroom for a cozy evening.  Place a plant container, a stuffed toy, a cushion and a sushi mat on the table"
    scores, indices = planner0.rag.retrieve_top_k_given_query(test_instruction, 1, 0)
    assert pytest.approx(scores[0], 0.1) == 1.0
    assert indices[0] == 0

    # Assert if the prompt example is being added correctly
    test_instruction = "Help me move the electric kettle, kettle, candle, and sponge from the kitchen to the living room"
    result_prompt, _ = planner0.prepare_prompt(
        input_instruction=test_instruction,
        world_graph=env_interface.world_graph[default_agent_uid],
    )
    assert test_instruction in result_prompt

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()


def test_planner_demo_multiproc():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "planner@evaluation.planner=dag_centralized_planner",
            "num_proc=2",
            "+evaluation.agents.agent_0.config.tools.motor_skills.oracle_rearrange.skill_config.nav_skill_config.teleport=True",
            "+evaluation.agents.agent_1.config.tools.motor_skills.oracle_rearrange.skill_config.nav_skill_config.teleport=True",
            "+evaluation.agents.agent_0.config.tools.motor_skills.oracle_nav.skill_config.teleport=True",
            "+evaluation.agents.agent_1.config.tools.motor_skills.oracle_nav.skill_config.teleport=True",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    seed = 47668090
    fix_config(config)
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    num_episodes = len(dataset.episodes)
    mp_ctx = mp.get_context("forkserver")
    proc_infos = []
    ochunk_size = num_episodes // config.num_proc
    # Prepare chunked datasets
    chunked_datasets = []
    # TODO: we may want to chunk by scene
    start = 0
    for i in range(config.num_proc):
        chunk_size = ochunk_size
        if i < (num_episodes % config.num_proc):
            chunk_size += 1
        end = min(start + chunk_size, num_episodes)
        indices = slice(start, end)
        chunked_datasets.append(indices)
        start += chunk_size

    for episode_index_chunk in chunked_datasets:
        episode_subset = dataset.episodes[episode_index_chunk]
        new_dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )

        parent_conn, child_conn = mp_ctx.Pipe()
        proc_args = (config, new_dataset, child_conn)
        p = mp_ctx.Process(target=run_planner, args=proc_args)
        p.start()
        proc_infos.append((parent_conn, p))

    # Get back info
    all_stats_episodes: Dict[str, Dict] = {
        str(i): {} for i in range(config.num_runs_per_episode)
    }
    updated_episodes = []
    for conn, proc in proc_infos:
        stats_episodes = conn.recv()
        updated_episodes.append(len(stats_episodes["0"]))
        for run_id, stats_run in stats_episodes.items():
            all_stats_episodes[str(run_id)].update(stats_run)
        proc.join()
    # Check that all the episodes are collected
    assert len(all_stats_episodes["0"]) == 4
    # Check episodes are well distributed
    assert sorted(updated_episodes) == [2, 2]

    # check that traces and statistics are written
    assert os.path.isfile("outputs/test/results/episode_result_log.csv")
    assert os.path.isfile("outputs/test/results/run_result_log.csv")


def test_zero_shot_react_planner():
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation=decentralized_evaluation_runner_single_agent",
            "planner@evaluation.agents.agent_0.planner=llm_zero_shot_react_planner",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=llama",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=llama",
            "agent@evaluation.agents.agent_0.config=oracle_rearrange_agent_motortoolsonly",
            "evaluation.agents.agent_0.planner.plan_config.objects_response_include_states=True",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)
    config.evaluation.agents.agent_0.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }

    # Set the planner
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    observations = env_interface.get_observations()
    planner0 = eval_runner.planner[0]
    planner0.llm.generate = Mock(return_value="non action")

    # get actions through the eval runner
    _, planner_info, _ = eval_runner.get_low_level_actions(
        "test instruction", observations, env_interface.world_graph
    )

    # check that the agent role and furniture list is included in the prompt
    assert (
        'You are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.'
        in planner0.curr_prompt
    )
    furniture_list = """living_room_1: floor_living_room_1, table_10, couch_18, shelves_26, chair_28, chair_29, table_32, chair_33, chair_34, chair_35, chair_36, table_46, table_47, stand_55, chest_of_drawers_56
other_room_1: floor_other_room_1
bathroom_1: floor_bathroom_1, chair_23, table_37, unknown_38, bench_48
bathroom_2: floor_bathroom_2, unknown_14, table_50
bedroom_1: floor_bedroom_1, chair_16, table_22, bed_45, bench_49, chest_of_drawers_52, chest_of_drawers_54
laundryroom_1: floor_laundryroom_1, washer_dryer_11, washer_dryer_12, shelves_15
entryway_1: floor_entryway_1, bench_51
bedroom_2: floor_bedroom_2, table_13, chair_17, bed_21, table_24, table_25, chair_27, stool_31, bench_44, table_59, chest_of_drawers_61
hallway_1: floor_hallway_1, table_30
kitchen_1: floor_kitchen_1, counter_19, chair_20, cabinet_39, cabinet_40, chair_41, chair_42, chair_43, counter_53, cabinet_57, fridge_58, unknown_60"""
    assert "The following furnitures have a faucet: cabinet_57" in planner0.curr_prompt
    assert furniture_list in planner0.curr_prompt
    assert "Task: test instruction" in planner0.curr_prompt

    # make sure responses are added to the prompt with the udpated object list
    assert planner0.curr_prompt.endswith(
        '<|start_header_id|>user<|end_header_id|>\n\nResult: SyntaxError in Action directive. Opening "[" or closing "]" square bracket is missing!\nObjects: cushion_0: chair_17 in bedroom_2. States: clean: False\ntoy_construction_set_1: table_25 in bedroom_2. States: clean: True<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    )
    planner0.llm.generate = Mock(
        return_value="Thought: I should explore the living room\nNavigate[living_room_1]"
    )

    # get actions through the eval runner
    eval_runner.get_low_level_actions(
        "test instruction", observations, env_interface.world_graph
    )

    # make sure responses are added to the prompt with the udpated object list
    assert planner0.curr_prompt.endswith(
        "Thought: I should explore the living room\nNavigate[living_room_1]\nAssigned!<|eot_id|>"
    )

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()
