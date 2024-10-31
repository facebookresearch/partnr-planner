#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sys
import time
import os
import json
import glob

# append the path of the
# parent directory
sys.path.append("..")

import hydra
from tqdm import tqdm
from torch import multiprocessing as mp
import traceback


from habitat_llm.utils import cprint, setup_config, fix_config


from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)

from habitat_llm.agent.env.dataset import CollaborationDatasetV0


# Function to write data to the CSV file
def write_to_csv(file_name, result_dict):
    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())

        # Check if the file is empty (to write headers)
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(result_dict)


def compute_stats(config):
    fpath = os.path.join(
        config.evaluation.output_dir,
        "episode_checks",
        config.habitat.dataset.data_path.split("/")[-1],
    )
    summary_json = f"{fpath}_summary.json"
    files = glob.glob(f"{fpath}/*")
    total = len(files)
    avg_stats = {"task_percent_complete": 0, "task_state_success": 0}
    no_breaks = 0
    summary_content = {}
    for file in files:
        with open(file, "r") as f:
            content = json.load(f)
        no_breaks += content["success_init"]
        if content["success_init"]:
            for key in ["task_percent_complete", "task_state_success"]:
                avg_stats[key] += content["info"][key]
        ep_name = file.split("/")[-1]
        summary_content[ep_name] = content

    print(f"No exceptions: {no_breaks*100/total:.2f}%")
    for key, val in avg_stats.items():
        print(f"{key}: {val/no_breaks}")

    with open(summary_json, "w+") as f:
        f.write(json.dumps(summary_content))


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_eval(config):
    fix_config(config)
    # Setup a seed
    # seed = 48212516
    seed = 47668090
    t0 = time.time()
    # Setup config
    config = setup_config(config, seed)
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    num_episodes = len(dataset.episodes)

    if config.num_proc == 1:
        episode_subset = dataset.episodes
        new_dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
        run_verifier(config, new_dataset)
    else:
        # Process episodes in parallel
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
            p = mp_ctx.Process(target=run_verifier, args=proc_args)
            p.start()
            proc_infos.append((parent_conn, p))

        # Get back info
        for conn, proc in proc_infos:
            try:
                conn.recv()
            except Exception:
                pass
            proc.join()

    e_t = time.time() - t0
    print("Elapsed Time: ", e_t)

    compute_stats(config)


def get_output_file(config, env_interface):
    dataset_file = env_interface.conf.habitat.dataset.data_path.split("/")[-1]
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    output_file = os.path.join(
        config.evaluation.output_dir,
        "episode_checks",
        dataset_file,
        f"{episode_id}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


def process_success(config, env_interface):
    output_file = get_output_file(config, env_interface)
    metrics = env_interface.env.env.env._env.get_metrics()
    success_dict = {
        "success_init": True,
        "info": {
            metric_name: metrics[metric_name]
            for metric_name in ["task_percent_complete", "task_state_success"]
        },
    }
    with open(output_file, "w+") as f:
        f.write(json.dumps(success_dict))


def process_error(config, env_interface):
    output_file = get_output_file(config, env_interface)
    exc_string = traceback.format_exc()
    failure_dict = {"success_init": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def run_verifier(config, dataset: CollaborationDatasetV0 = None, conn=None):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # Setup interface with the simulator if the planner depends on it
    if config.env == "habitat":
        # Remove sensors if we are not saving video
        remove_visual_sensors(config)

        # We register the dynamic habitat sensors
        register_sensors(config)

        # We register custom actions
        register_actions(config)

        # We register custom measures
        register_measures(config)

        # Initialize the environment interface for the agent
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
        process_success(config, env_interface)
    except Exception:
        process_error(config, env_interface)

    # failures = []
    num_episodes = len(env_interface.env.episodes)
    for _ in tqdm(range((num_episodes))):
        # Reset env_interface (moves onto the next episode in the dataset)
        try:
            env_interface.reset_environment()
            process_success(config, env_interface)
        except Exception:
            process_error(config, env_interface)

    # aggregate metrics across all runs.
    if conn is not None:
        conn.send([0])

    env_interface.env.close()
    del env_interface

    if conn is not None:
        # Potentially we may want to send something

        conn.close()

    return


if __name__ == "__main__":
    cprint(
        "\nStart of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run planner
    run_eval()

    cprint(
        "\nEnd of the example program to demonstrate multi-agent planner demo.",
        "blue",
    )
