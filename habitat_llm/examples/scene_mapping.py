#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script implements structured episodes over a collection of scenes, which
ask the agent to go to each furniture within the scene and save a RGBD+pose trajectory.
This trajectory is then used to create a map of the scenes through Concept-Graphs.
"""

import sys

# append the path of the
# parent directory
sys.path.append("..")


from habitat_llm.utils import cprint, setup_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_sensors,
)

from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
)
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


# Method to load agent planner from the config
def run_planner():
    # Setup a seed
    seed = 47668090

    # setup required overrides
    DATASET_OVERRIDES = [
        "habitat.dataset.data_path=data/datasets/path/to/val/scenes",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
    ]
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False"
    ]
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=True",
        "evaluation.output_dir=./outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[main_agent]",
    ]

    EPISODE_OVERRIDES = [
        "+episode_indices=[2,87,370,444,515,590,435,390,555,50,452,355]"
    ]  # USE FOR VAL SCENES

    # Setup config
    config_base = get_config(
        "examples/single_agent_scene_mapping.yaml",
        overrides=DATASET_OVERRIDES
        + SENSOR_OVERRIDES
        + LLM_OVERRIDES
        + TRAJECTORY_OVERRIDES
        + EPISODE_OVERRIDES,
    )
    config = setup_config(config_base, seed)

    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # Initialize the environment interface for the agent
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    if config.get("episode_indices", None) is not None:
        episode_subset = [dataset.episodes[x] for x in config.episode_indices]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Instantiate the agent planner
    # NOTE: we don't strictly need this but it has good helper functions to make
    # scripted execution easy
    eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)

    # book-keeping and verbosity
    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print the agent list
    cprint(f"Agent List: {eval_runner.agent_list}", "blue")
    if env_interface._single_agent_mode:
        cprint("Single agent mode", "green")
    cprint("---------------------------------------\n", "blue")
    num_episodes = len(env_interface.env.episodes)
    processed_scenes = set()
    robot_agent_uid = config.robot_agent_uid

    # initial reset to load first episode
    for idx in range(num_episodes):
        env_interface.reset_environment()
        eval_runner.reset()
        cur_episode = env_interface.env.env.env._env.current_episode
        cur_episode.episode_id = idx
        scene_id = cur_episode.scene_id
        if str(scene_id) in processed_scenes:
            print(f"Skipping scene {scene_id}. Already mapped.")
            continue
        print(
            f"Processing scene: {scene_id}, episode: {idx+1}/{num_episodes}, processed scenes: {len(processed_scenes)}/10"
        )
        if len(processed_scenes) == 10:
            break
        observations = env_interface.get_observations()

        # get the list of all rooms in this house
        rooms = env_interface.world_graph[robot_agent_uid].get_all_nodes_of_type(Room)

        print(f"---Total number of rooms in this house: {len(rooms)}---\n\n")
        while rooms:
            print(f"{len(rooms)} more room to go...")
            current_room = rooms.pop()
            hl_action_name = "Explore"
            hl_action_input = current_room.name
            hl_action_done = False
            print(f"Executing high-level action: {hl_action_name} on {hl_action_input}")
            while not hl_action_done:
                # Get response and/or low level actions
                low_level_action, response = eval_runner.planner.agents[
                    0
                ].process_high_level_action(
                    hl_action_name, hl_action_input, observations
                )
                low_level_action = {0: low_level_action}
                obs, reward, done, info = env_interface.step(low_level_action)
                # Refresh observations
                observations = env_interface.parse_observations(obs)
                # Store third person frames for generating video
                hl_dict = {0: (hl_action_name, hl_action_input)}
                eval_runner._store_for_video(observations, hl_dict)

                # figure out how to get completion signal
                if response:
                    print(f"\tResponse: {response}")
                    hl_action_done = True
            print(
                f"\tCompleted high-level action: {hl_action_name} on {hl_action_input}"
            )

        if eval_runner.frames:
            eval_runner._make_video(scene_id)
        processed_scenes.add(str(scene_id))
    env_interface.sim.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the scene mapping routine",
        "blue",
    )

    # Run planner
    run_planner()

    cprint(
        "\nEnd of the single-agent, scene-mapping routine",
        "blue",
    )
