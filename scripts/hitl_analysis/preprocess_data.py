# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import gzip
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List

import tqdm


def count_files_in_subfolders(folder_path: str) -> int:
    """Prints the number of files in each subfolder. Returns the sum of this quantity"""
    subfolder_with_n_files: Dict[int, int] = defaultdict(int)
    for _, _, files in os.walk(folder_path):
        subfolder_with_n_files[len(files)] += 1

    tot = 0
    for k in sorted(subfolder_with_n_files):
        v = subfolder_with_n_files[k]
        tot += k * v
        print(f"Subfolders with {k} file(s): {v}")

    return tot


def order_session_dirs_by_timestamp(dataset_path: str) -> List[str]:
    """
    Returns a list of session directories in `dataset_path` sorted by the start timestamp.
    """
    session_dirs = []  # tuple of (dirname, timestamp)
    for session_dir in os.listdir(dataset_path):
        session_file = os.path.join(dataset_path, session_dir, "session.json.gz")
        with gzip.open(session_file, "rt") as f:
            session_data = json.load(f)
        timestamp = session_data["session"]["start_timestamp"]
        session_dirs.append((session_dir, timestamp))
    return [x[0] for x in sorted(session_dirs, key=lambda x: x[1])]


def copy_files_to_directory(file_list: List[str], target_directory: str) -> None:
    """
    Copies all files in `file_list` to `target_directory`.
    If `target_directory` already exists, remove it.
    """
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.makedirs(target_directory)

    for file_name in file_list:
        if not os.path.isfile(file_name):
            print(f"Error: {file_name} does not exist.")
            continue
        target_file_path = os.path.join(target_directory, os.path.basename(file_name))
        shutil.copy(file_name, target_file_path)


def count_empty_dicts(dict_list):
    """
    Count intermediate empty dicts in a list of dicts.
    Since the HITL session consists of many frames where nothing happens, the recorded
    frames are empty. This function counts the in-between empty dicts.
    """
    counts = []
    empty_count = 0

    for dictionary in dict_list:
        if dictionary:
            if empty_count > 0:
                counts.append(empty_count)
                empty_count = 0
        else:
            empty_count += 1

    if empty_count > 0:
        counts.append(empty_count)

    return counts


def analyze_frames_single_learn(frames):
    tasks_agent_0, tasks_agent_1, all_tasks, old_task_percent_complete = 0, 0, 0, 0
    for frame in frames:
        if not len(frame):
            continue

        user_event = frame["users"][-1]["events"]
        if len(frame["users"]) == 2:
            agent_event = frame["users"][0]["events"]
        else:
            agent_event = (
                frame["agent_states"][0]["events"]
                if "events" in frame["agent_states"][0]
                else []
            )

        if (
            len(user_event)
            and frame["task_percent_complete"] > old_task_percent_complete
        ):
            tasks_agent_1 += 1
            all_tasks += 1
            old_task_percent_complete = frame["task_percent_complete"]
        elif (
            len(agent_event)
            and frame["task_percent_complete"] > old_task_percent_complete
        ):
            tasks_agent_0 += 1
            all_tasks += 1
            old_task_percent_complete = frame["task_percent_complete"]
        elif len(user_event):
            if user_event[0]["type"] == ("place"):
                all_tasks += 1
        elif len(agent_event) and agent_event[0]["type"] == ("place"):
            all_tasks += 1

    return tasks_agent_0, tasks_agent_1, all_tasks


def compute_derived_metrics(target_dir_best: str) -> Dict[str, Any]:
    """
    Returns a dictionary of episode-based metrics supporting:
        - human/robot task division
        - exploration efficiency
    """
    ratio_agent_0 = {}
    ratio_agent_1 = {}
    ratio_extraneous_actions = {}
    explore_steps = {}
    remaining_num = {}

    files = os.listdir(target_dir_best)
    pbar = tqdm.tqdm(total=len(files))
    pbar.set_description("Derived Metrics")
    for file in files:
        pbar.update(1)
        filename = os.path.join(target_dir_best, file)
        tasks_agent_0, tasks_agent_1 = 0, 0

        with gzip.open(filename, "rt") as f:
            data = json.load(f)

        episode_id = data["episode"]["episode_id"]
        frames = data["frames"]
        counts = count_empty_dicts(frames)
        if not len(counts):
            counts = [0, 0]

        steps = 0
        for frame in frames:
            steps += 1
            if not len(frame):
                continue
            for _id, user in enumerate(frame["users"]):
                if len(user["events"]):
                    explore_steps[episode_id] = steps - 1 - counts[0]
                    break

        remaining_num[episode_id] = len(frames) - counts[0] - counts[-1]

        if frames[-1]["task_percent_complete"] < 0.99:
            continue

        tasks_agent_0, tasks_agent_1, all_tasks = analyze_frames_single_learn(frames)

        ratio_0 = tasks_agent_0 / (tasks_agent_0 + tasks_agent_1)
        ratio_1 = tasks_agent_1 / (tasks_agent_0 + tasks_agent_1)
        ratio_extra = (all_tasks - tasks_agent_0 - tasks_agent_1) / all_tasks

        ratio_agent_0[episode_id] = ratio_0
        ratio_agent_1[episode_id] = ratio_1
        ratio_extraneous_actions[episode_id] = ratio_extra

    return {
        "ratio_agent_0": ratio_agent_0,
        "ratio_agent_1": ratio_agent_1,
        "ratio_extraneous_actions": ratio_extraneous_actions,
        "explore_steps": explore_steps,
        "remaining_num": remaining_num,
    }


def preprocess_data(collection_path: str, recompute: bool = False) -> None:
    raw_path = os.path.join(collection_path, "raw")
    processed_path = os.path.join(collection_path, "processed")
    metrics_file = os.path.join(processed_path, "processed_metrics.json")
    best_dir = os.path.join(processed_path, "best")

    # check if data has already been processed
    if os.path.exists(best_dir) and os.path.exists(metrics_file) and not recompute:
        with open(metrics_file) as f:
            metrics = json.load(f)

        if "ratio_agent_0" in metrics:
            return

        metrics = {**metrics, **compute_derived_metrics(best_dir)}
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        return

    os.makedirs(processed_path, exist_ok=True)

    eid_to_round_needed: Dict[str, int] = {}
    eid_to_pc: Dict[str, float] = {}
    eid_to_success: Dict[str, int] = {}
    eid_to_explanation: Dict[str, str] = {}
    eid_to_filename: Dict[str, str] = {}
    all_users: List[str] = []
    pbar = tqdm.tqdm(total=count_files_in_subfolders(raw_path))
    for d in order_session_dirs_by_timestamp(raw_path):
        root = os.path.join(raw_path, d)
        for file in os.listdir(root):
            pbar.update(1)
            if file == "session.json.gz":
                continue
            if not file.endswith(".json.gz"):
                continue
            filename = os.path.join(root, file)
            with gzip.open(filename, "rt") as f:
                data = json.load(f)
            finished = data["episode"]["finished"]
            if not finished:
                continue
            user_id = data["users"][0]["connection_record"]["user_id"]
            if not user_id.strip().isdigit():
                continue
            if "is_tutorial" in data["episode"]["episode_info"]:
                is_tutorial = data["episode"]["episode_info"]["is_tutorial"]
                if is_tutorial:
                    continue
            if user_id not in all_users:
                all_users.append(user_id)

            episode_id = data["episode"]["episode_id"]

            if "metrics" in data:
                task_percent_complete = data["metrics"]["task_percent_complete"]
                task_explanation = data["metrics"]["task_explanation"]
            else:
                task_percent_complete = data["episode"]["task_percent_complete"]
                task_explanation = data["episode"]["task_explanation"]

            task_success = int(task_percent_complete > 0.99)

            if episode_id in eid_to_success:
                old_pc = eid_to_pc[episode_id]

                if not eid_to_success[episode_id]:
                    eid_to_round_needed[episode_id] += 1

                if task_percent_complete > old_pc:
                    eid_to_pc[episode_id] = task_percent_complete
                    eid_to_success[episode_id] = task_success
                    eid_to_explanation[episode_id] = task_explanation
                    eid_to_filename[episode_id] = filename
            else:
                eid_to_pc[episode_id] = task_percent_complete
                eid_to_success[episode_id] = task_success
                eid_to_round_needed[episode_id] = 1
                eid_to_filename[episode_id] = filename

    # if episode failed, its success round is -1
    for eid, success in eid_to_success.items():
        if not success:
            eid_to_round_needed[eid] = -1

    # store all best and failed episodes in separate folders
    copy_files_to_directory(list(eid_to_filename.values()), best_dir)

    fail_list = [
        fname for eid, fname in eid_to_filename.items() if eid_to_success[eid] == 0
    ]
    copy_files_to_directory(fail_list, os.path.join(processed_path, "failed"))

    # TODO: Add additional metrics
    processed_metrics = {
        "eid_to_round_needed": eid_to_round_needed,
        "eid_to_pc": eid_to_pc,
        "eid_to_success": eid_to_success,
        "eid_to_explanation": eid_to_explanation,
        "eid_to_filename": eid_to_filename,
        "all_users": all_users,
        **compute_derived_metrics(best_dir),
    }

    with open(metrics_file, "w") as f:
        json.dump(processed_metrics, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-path",
        type=str,
        required=False,
        default="data/hitl_data/2024-10-02-object-states/p5_single_train_10k",
    )
    parser.add_argument(
        "--recompute",
        action=argparse.BooleanOptionalAction,
        help="recompute the best/failed directories and initial metrics",
    )
    args = parser.parse_args()
    preprocess_data(args.collection_path, bool(args.recompute))
