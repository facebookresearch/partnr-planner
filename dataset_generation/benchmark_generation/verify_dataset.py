#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

# isort: skip_file

import time
import os
import json
import glob
from typing import Dict, List

from omegaconf import OmegaConf, open_dict

import habitat
from hydra import compose, initialize
import numpy as np
from torch import multiprocessing as mp
import traceback

import tqdm


from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    SameArgConstraint,
)
from habitat_llm.sims.collaboration_sim import CollaborationSim
from habitat_llm.utils import cprint


from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

from habitat_llm.agent.env import (
    register_measures,
    register_sensors,
)
import habitat.sims.habitat_simulator.sim_utilities as sutils

from habitat_llm.agent.env.dataset import CollaborationDatasetV0, CollaborationEpisode


def summarize_verifications(save_results_dir: str):
    """
    Aggregates the metrics (percent complete and success) and exceptions
    unpon episode initialization. Displays in text and saves to file.
    """
    summary_json = os.path.join(save_results_dir, "summary.json")
    files = glob.glob(os.path.join(save_results_dir, "*"))
    total = len(files)
    avg_stats = {"task_percent_complete": 0, "task_state_success": 0}
    no_breaks = 0
    summary_content = {}
    for file in files:
        if file.endswith("summary.json"):
            continue
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


def get_output_file(save_results_dir, eid):
    os.makedirs(save_results_dir, exist_ok=True)
    output_file = os.path.join(save_results_dir, f"{eid}.json")
    return output_file


def process_success_at_init(env, save_results_dir, eid):
    output_file = get_output_file(save_results_dir, eid)
    metrics = env.get_metrics()
    success_dict = {
        "success_init": True,
        "info": {
            metric_name: metrics[metric_name]
            for metric_name in ["task_percent_complete", "task_state_success"]
        },
    }
    with open(output_file, "w+") as f:
        f.write(json.dumps(success_dict))


def process_error_at_init(save_results_dir, eid):
    output_file = get_output_file(save_results_dir, eid)
    exc_string = traceback.format_exc()
    failure_dict = {"success_init": False, "info": str(exc_string)}
    with open(output_file, "w+") as f:
        f.write(json.dumps(failure_dict))


def assert_eval_handles_exist(env: habitat.Env) -> None:
    """
    Not all object handles are explicitly checked for existence during evaluation
    inference. Here we explicitly assert that they do exist.
    """
    ep: CollaborationEpisode = env.current_episode
    sim: CollaborationSim = env.sim
    handles_missing = set()
    for prop in ep.evaluation_propositions:
        for k in {
            "object_handles",
            "receptacle_handles",
            "entity_handles_a",
            "entity_handles_b",
        }:
            if k not in prop.args:
                continue
            for handle in prop.args[k]:
                if sutils.get_obj_from_handle(sim, handle) is None:
                    handles_missing.add(handle)

    if len(handles_missing):
        raise AssertionError(
            f"Evaluation handles do not exist in sim: {handles_missing}"
        )


def assert_constraint_args_exist(env: habitat.Env) -> None:
    """Constraints reference proposition arguments. Assert that these arguments exist."""
    task_eval_log = env.get_metrics()["task_evaluation_log"]
    state_sequence = task_eval_log["state_sequence"]
    propositions = task_eval_log["propositions"]
    constraints = task_eval_log["constraints"]

    if len(state_sequence) == 0:
        raise AssertionError("empty state sequence. Did you call reset?")

    actual_prop_result_keys = {
        prop_idx: set(prop_result.info.keys())
        for prop_idx, prop_result in enumerate(state_sequence[-1])
    }

    for c in constraints:
        if not isinstance(c, (SameArgConstraint, DifferentArgConstraint)):
            continue

        for prop_idx, arg_name in c.arg_names.items():
            if arg_name not in actual_prop_result_keys[prop_idx]:
                raise AssertionError(
                    f"Arg name `{arg_name}` missing from PropositionResult"
                    f" for proposition {str(propositions[prop_idx])}."
                )


def group_by_scene(episodes: List[CollaborationEpisode]) -> List[CollaborationEpisode]:
    """
    Group episodes by scene before passing to the episode iterator.
    This way we can track episode IDs prior to calling env.reset().
    """
    sort_ks: Dict[str, int] = {}
    for e in episodes:
        if e.scene_id not in sort_ks:
            sort_ks[e.scene_id] = len(sort_ks)

    return sorted(episodes, key=lambda e: sort_ks[e.scene_id])  # type: ignore[arg-type]


def run_verifier(
    config,
    save_results_dir: str,
    dataset: CollaborationDatasetV0 = None,
    conn=None,
    show_progress: bool = True,
):
    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    os.environ["GLOG_minloglevel"] = "3"  # noqa: SIM112
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(HabitatConfigPlugin)

    with open_dict(config):
        config.habitat.simulator.agents_order = sorted(
            config.habitat.simulator.agents.keys()
        )

        # Propagate the metadata folder config to the simulator
        config_dict = OmegaConf.create(
            OmegaConf.to_container(config.habitat, resolve=True)
        )
        # TODO: refactor this. We shouldn't need to copy configs into other subconfigs to pass information. This is done now because CollaborationSim needs metadata paths for init.
        config_dict.simulator.metadata = config.habitat.dataset.metadata
        config.habitat = config_dict

    register_sensors(config)
    register_measures(config)

    dataset.episodes = group_by_scene(dataset.episodes)
    env = habitat.Env(config=config, dataset=dataset)
    env.sim.dynamic_target = np.zeros(3)

    range_func = tqdm.trange if show_progress else range
    for i in range_func(len(env.episodes)):
        eid = env.episodes[i].episode_id
        try:
            env.reset()
            assert_eval_handles_exist(env)
            assert_constraint_args_exist(env)
            process_success_at_init(env, save_results_dir, eid)
        except Exception:
            process_error_at_init(save_results_dir, eid)

    # aggregate metrics across all runs.
    if conn is not None:
        conn.send([0])

    env.close()
    del env

    if conn is not None:
        # Potentially we may want to send something
        conn.close()

    return


def log_dataset_verification_process(
    save_results_dir: str, num_episodes: int, sleep_time: int = 2
) -> None:
    """
    Monitors the save_results_dir, tracking how many episodes have been verified based
    on the number of files that exist. Returns when all files have been generated.

    Args:
        save_results_dir (str): path to the verification directory to watch
        num_episodes (int): target number of episode verification files
        sleep_time (int): number of seconds to wait between queries
    """

    def is_result_f(fname):
        s = fname.split(".")
        if not s[0].isdigit():
            return False
        return s[1] == "json"

    with tqdm.tqdm(total=num_episodes) as pbar:
        pbar.set_description("Verification Progress")

        n_files = 0
        while True:
            time.sleep(sleep_time)

            if not os.path.exists(save_results_dir):
                continue

            results_files = [f for f in os.listdir(save_results_dir) if is_result_f(f)]
            delta = len(results_files) - n_files
            pbar.update(delta)
            n_files += delta
            if n_files >= num_episodes:
                break


def verify_dataset_parallel(
    dataset_path: str,
    save_results_dir: str,
    num_proc: int = 1,
    show_progress: bool = True,
):
    # Set up hydra config
    with initialize(version_base=None, config_path="../../habitat_llm/conf"):
        config = compose(
            config_name="benchmark_gen/evaluation_validation.yaml",
            overrides=[
                "+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata",
                f"habitat.dataset.data_path={dataset_path}",
            ],
        )

    config = OmegaConf.create(config)
    t0 = time.time()
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    dataset.episodes = group_by_scene(dataset.episodes)

    if num_proc == 1:
        run_verifier(config, save_results_dir, dataset)
    else:
        # Process episodes in parallel
        mp_ctx = mp.get_context("forkserver")
        proc_infos = []
        num_episodes = len(dataset.episodes)
        ochunk_size = num_episodes // num_proc
        chunked_datasets = []
        start = 0
        for i in range(num_proc):
            chunk_size = ochunk_size
            if i < (num_episodes % num_proc):
                chunk_size += 1
            end = min(start + chunk_size, num_episodes)
            indices = slice(start, end)
            chunked_datasets.append(indices)
            start += chunk_size

        if show_progress:
            parent_conn, child_conn = mp_ctx.Pipe()
            p = mp_ctx.Process(
                target=log_dataset_verification_process,
                args=(save_results_dir, num_episodes),
            )
            p.start()
            proc_infos.append((parent_conn, p))

        for episode_index_chunk in chunked_datasets:
            episode_subset = dataset.episodes[episode_index_chunk]
            new_dataset = CollaborationDatasetV0(
                config=config.habitat.dataset, episodes=episode_subset
            )

            parent_conn, child_conn = mp_ctx.Pipe()
            proc_args = (config, save_results_dir, new_dataset, child_conn, False)
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

    summarize_verifications(save_results_dir)


if __name__ == "__main__":
    """
    A script that verifies episodes successful load and eval measures don't crash.

    To run:
    python dataset_generation/benchmark_generation/verify_dataset.py \
        --dataset-path=path/to/dataset_name.json.gz \
        --save-results-dir=data/episode_checks/dataset_name \
        --num-proc=5
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--save-results-dir", type=str, default="")
    parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    save_results_dir = args.save_results_dir
    if save_results_dir == "":
        dset_name = args.dataset_path.split("/")[-1].split(".")[0]
        save_results_dir = f"data/episode_checks/{dset_name}"

    verify_dataset_parallel(args.dataset_path, save_results_dir, args.num_proc)
