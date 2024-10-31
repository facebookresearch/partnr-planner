#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import List, Tuple, Any


# append the path of the
# parent directory
sys.path.append("..")

import omegaconf
import hydra

from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import execute_skill, DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)


# Method to load agent planner from the config
@hydra.main(
    config_path="../conf", config_name="examples/skill_runner_default_config.yaml"
)
def run_skills(config: omegaconf.DictConfig) -> None:
    """
    The main function for executing the skill_runner tool. A default config is provided.
    See the `main` function for example CLI command to run the tool.

    :param config: input is a habitat-llm config from Hydra. Can contain CLI overrides.
    """
    fix_config(config)
    # Setup a seed
    seed = 47668090
    # Setup some hardcoded config overrides (e.g. the metadata path)
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat skill testing."

    if not config.evaluation.save_video:
        remove_visual_sensors(config)

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # We register custom measures
    register_measures(config)

    # create the dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    # Initialize the environment interface for the agent
    env_interface = EnvironmentInterface(config, dataset=dataset)

    ##########################################
    # select and initialize the desired episode by index or id
    # NOTE: use "+skill_runner_episode_index=2" in CLI to set the episode index ( e.g. episode 2)
    # NOTE: use "+skill_runner_episode_id=<id>" in CLI to set the episode id ( e.g. episode "")
    assert not (
        hasattr(config, "skill_runner_episode_index")
        and hasattr(config, "skill_runner_episode_id")
    ), "Episode selection options are mutually exclusive."
    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        print(f"Loading episode_index = {episode_index}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        print(f"Loading episode_id = {episode_id}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
    env_interface.reset_environment()
    ###########################################

    # Initialize the planner
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)
    planner.reset()

    sim = env_interface.sim

    # show the topdown map if requested
    if hasattr(config, "skill_runner_show_topdown"):
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        dbv.peek("stage").show()
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent()
        dbv.remove_dbv_agent()

    # get make video flag from config if set
    make_video = config.get("skill_runner_make_video", True)

    ############################
    # done with setup, prompt the user and start running skills

    # available skills
    skills = {
        "Navigate": "Navigate <agent_index> <entity_name>",
        "Open": "Open <agent_index> <entity_name>",
        "Close": "Close <agent_index> <entity_name>",
        "Pick": "Pick <agent_index> <entity_name>",
        # Place skill requires 5 arguments, comma separated, no spaces:
        "Place": "Place <agent_index> <entity_name_0,relation_0,entity_name_1,relation_1,entity_name_2>",
    }
    exit_skill = "exit"
    help_skill = "help"
    entity_skill = "entities"
    pdb_skill = "debug"
    cumulative_video_skill = "make_video"

    cprint("Welcome to skill_runner!", "green")
    cprint(
        f"Current Episode (id=={sim.ep_info.episode_id}) is running in scene {sim.ep_info.scene_id} with info: {sim.ep_info.info}.",
        "green",
    )

    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    help_text = f"Available skills are {skills}. Type a skill to begin.\n alternatively type one of: \n  '{exit_skill}' - exit the program \n  '{help_skill}' - display help text \n  '{entity_skill}' - display all available entities"
    cprint(help_text, "green")

    # setup a sequence of commands to run immediately without manual input
    scripted_commands: List[str] = []

    # collect debug frames to create a final video
    cumulative_frames: List[Any] = []

    command_index = 0
    # history of skill commands and their responses
    command_history: List[Tuple[str, str]] = []
    while True:
        cprint("Enter Command", "blue")
        if len(scripted_commands) > command_index:
            user_input = scripted_commands[command_index]
            print(user_input)
        else:
            user_input = input("> ")

        selected_skill = None

        if user_input == exit_skill:
            print("==========================")
            print("Exiting. Command History:")
            for ix, t in enumerate(command_history):
                print(f" [{ix}]: '{t[0]}' -> '{t[1]}'")
            print("==========================")
            exit()
        elif user_input == help_skill:
            cprint(help_text, "green")
        elif user_input == entity_skill:
            print_all_entities(env_interface.perception.gt_graph)
        elif user_input == pdb_skill:
            # peek an entity
            dbv = DebugVisualizer(sim, config.paths.results_dir)
            dbv.create_dbv_agent()
            # NOTE: do debugging calls here
            # example to peek an entity: dbv.peek(env_interface.world_graph.get_node_from_name('table_50').sim_handle).show()
            breakpoint()
            dbv.remove_dbv_agent()
        elif user_input == cumulative_video_skill:
            # create a video of all accumulated frames thus far and play it
            if len(cumulative_frames) > 0:
                dvu = DebugVideoUtil(
                    env_interface, env_interface.conf.paths.results_dir
                )
                dvu.frames = cumulative_frames
                dvu._make_video(postfix="cumulative", play=True)
        elif user_input in skills:
            # fill information piece by piece
            selected_skill = user_input
            # get the agent index
            agent_ix = input("Agent Index (0=robot, 1=human) = ")
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                continue
            target_entity_name = input("Target Entity = ")
        elif user_input.split(" ")[0] in skills:
            # attempt to parse full skill definition from string
            skill_components = user_input.split(" ")
            selected_skill = skill_components[0]
            agent_ix = skill_components[1]
            if agent_ix not in ["0", "1"]:
                cprint("... invalid Agent Index, aborting.", "red")
                continue
            target_entity_name = skill_components[2]
        else:
            cprint("... invalid command.", "red")

        # configure and run the skill
        if selected_skill is not None:
            high_level_skill_actions = {
                int(agent_ix): (selected_skill, target_entity_name, None)
            }

            ############################
            # run the skill
            try:
                responses, _, frames = execute_skill(
                    high_level_skill_actions,
                    planner,
                    vid_postfix=f"{command_index}_",
                    make_video=make_video,
                )
                command_history.append((user_input, responses[int(agent_ix)]))
                skill_name = high_level_skill_actions[int(agent_ix)][0]
                print(
                    f"{skill_name} completed. Response = '{responses[int(agent_ix)]}'"
                )
                cumulative_frames.extend(frames)
            except Exception as e:
                failure_string = f"Failed to execute skill with exception: {str(e)}"
                print(failure_string)
                command_history.append((user_input, failure_string))
            command_index += 1


##########################################
# CLI Example:
# HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.skill_runner hydra.run.dir="."
# or
# python habitat_llm/examples/skill_runner.py
#
##########################################
# Script Specific CLI overrides:
#
# (mutually exclusive)
# - '+skill_runner_episode_index=0' - initialize the episode with the specified index within the dataset
# - '+skill_runner_episode_id=' - initialize the episode with the specified "id" within the dataset
#
# - '+skill_runner_show_topdown=True' - show a topdown view of the scene upon initialization for context
#
##########################################
# Other useful CLI overrides:
#
# - 'habitat.dataset.data_path="<path to dataset .json.gz>"' - set the desired episode dataset
#
if __name__ == "__main__":
    cprint(
        "\nStart of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )

    # Run the skills
    run_skills()

    cprint(
        "\nEnd of the example program to run custom skill commands in a CollaborationEpisode.",
        "blue",
    )
