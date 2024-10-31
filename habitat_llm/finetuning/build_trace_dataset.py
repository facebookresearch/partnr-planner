#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json  # Added import
import os
import pickle
import re
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
from tqdm import tqdm

from habitat_llm.evaluation.evaluation_runner import ActionHistoryElement
from habitat_llm.llm.instruct.utils import (
    PERCEPTION_TOOL_STRINGS,
    STOP_WORD,
    build_single_step_prompt,
)
from habitat_llm.world_model.world_graph import WorldGraph


def extract_assistant_text_end(text):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>.*?<\|eot_id\|>"
    matches = re.finditer(pattern, text, re.DOTALL)
    end_indices = [m.end() for m in matches]
    return end_indices


# New code to process all text files in the specified directory
def process_directory_react(directory, output_directory, pc_filter=0.75):
    """
    Given a folder with multiple traces, generate react like traces.
    """
    stats_directory = os.path.join(directory, "stats")
    dataset_name = os.path.basename(directory.rstrip("/"))
    good_episodes = 0
    total_episodes = 0
    total_traces = 0
    for filename in tqdm(os.listdir(stats_directory)):
        total_episodes += 1
        ep_id = filename.split(".json")[0]
        stats_file = os.path.join(stats_directory, filename)
        with open(stats_file, "r") as f:
            stats = json.load(f)
        if not stats["success"]:
            continue
        stats_string = stats["stats"]
        stats_dict = json.loads(stats_string)
        if stats_dict["task_percent_complete"] < pc_filter:
            continue
        good_episodes += 1
        prompt_file = os.path.join(
            directory, "prompts", "0", f"prompt-episode_{ep_id}_0-0.txt"
        )
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()
        results = extract_assistant_text_end(content)
        for i in range(len(results)):
            end_index = results[i]
            sample = content[:end_index]
            if i < len(results) - 1:
                post = content[end_index:]
                result_line = post.split("\n")[2]
                assert result_line.startswith("Result:")
                result = result_line.split(": ")[1].strip()
                to_write = result.lower() == "successful execution!"
            else:
                to_write = True
            if to_write:
                total_traces += 1
                output_file = os.path.join(
                    output_directory, dataset_name, ep_id, f"sample_{i}.txt"
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(sample)

    print(
        f"Processed traces saved to: {output_file}. {good_episodes} good episodes out of {total_episodes} total episodes. {good_episodes/total_episodes*100:.2f}%"
    )
    print(f"Total traces: {total_traces}")


def process_file(pkl_file: str, args):
    """
    Given a pkl file with episode information, generate a per step txt file for finetuning a planning model.
    :param pkl_file: path to the file we want to process
    :param args: the information we use to build dataset traces
    """
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    pkl_samples = []

    instruction = data["instruction"]
    action_history = data["action_history"]
    assert len(action_history.keys()) < 3

    all_actions: List[ActionHistoryElement] = sum(action_history.values(), [])

    all_actions.sort(key=lambda x: (x.timestamp, x.info["log_time"]))
    agent_done = {agent_id: False for agent_id in action_history}

    if len(all_actions) == 0:
        print("skipping empty: ", pkl_file)
        return 0

    # If an episode had 100% success rate before the end, stop there.
    if args.early_stop:
        tpc = [
            action.info["planner_info"]["stats"]["task_percent_complete"]
            for action in all_actions
        ]
        if np.max(tpc) < args.percent_cut:
            print("skipping incomplete: ", pkl_file)
            return 0

    elif (
        all_actions[-1].info["planner_info"]["stats"]["task_percent_complete"]
        < args.percent_cut
    ):
        print("skipping incomplete: ", pkl_file)
        return 0

    # Filter all_actions if last_action_only is set. This generate a single trace which
    # allows for easier debugging
    if args.last_action_only:
        agent_0_actions = [action for action in all_actions if action.agent_uid == 0]
        all_actions = [agent_0_actions[-1]] if agent_0_actions else []

    for step in all_actions:
        # Skip actions for agents that are already done
        if agent_done[step.agent_uid]:
            continue

        # Override action to "Done" if task_state_success is true
        if (
            args.early_stop
            and step.info["planner_info"]["stats"]["task_state_success"] == 1
        ):
            # breakpoint()
            step.action = ("Done", None)
            agent_done[step.agent_uid] = True

        # Skip perception tools
        if step.action[0] in PERCEPTION_TOOL_STRINGS:
            continue

        # Skip actions that don't result in success if the flag is active
        if not (
            "success" in step.response.lower()
            or step.action[0] == "Done"
            # successful navigation does not have a response
            or (step.response is None and step.action[0] == "Navigate")
        ):
            continue

        if step.action[0] == "Done":
            agent_done[step.agent_uid] = True
        # Filter action history for only actions before the current action
        filtered_action_history = {}
        for agent_id, actions in action_history.items():
            filtered_actions = [
                action for action in actions if action.timestamp < step.timestamp
            ]
            filtered_action_history[int(agent_id)] = filtered_actions

        prefix = build_single_step_prompt(
            instruction,
            WorldGraph(graph=step.world_graph[int(step.agent_uid)].graph),
            step.agent_uid,
            filtered_action_history,
        )

        arg_string = "" if step.action[1] is None else step.action[1]

        target_string = f"{ step.action[0] }[{ arg_string }]{STOP_WORD}"

        sample = prefix + target_string
        if args.debug_info:
            st_str = json.dumps(step.info["planner_info"]["stats"])
            sample += f"\n{st_str}"
        if step.agent_uid == 0:
            pkl_samples.append(sample)

    # Create a folder for each pkl file and write its samples
    pkl_name = os.path.splitext(os.path.basename(pkl_file))[0]
    output_folder = os.path.join(args.output_dir, pkl_name)
    os.makedirs(output_folder, exist_ok=True)
    for i, sample in enumerate(pkl_samples):
        output_file = os.path.join(
            output_folder, f"sample_{i}.txt"
        )  # Change file extension to .txt if not outputting JSON
        with open(output_file, "w") as f:
            sample = sample.replace(
                "Action Wait[] is still in progress.", "Successful execution!"
            )
            f.write(sample)

    # Update total samples count
    if len(pkl_samples) == 0:
        print("no samples in : ", pkl_file)
    return len(pkl_samples)


def process_files(args):
    """
    Create prompts from a directory of traces
    """
    pkl_files = []
    total_samples = 0
    nworkers = args.num_workers
    dataset_name = os.path.basename(args.path.rstrip("/"))
    early_stop_str = "_early_stop" if args.early_stop else ""
    args.output_dir = (
        args.output_dir + f"_filter_{args.percent_cut}{early_stop_str}/{dataset_name}"
    )
    path = args.path
    path_str = f"{path}/detailed_traces/*.pkl"
    print(f"Checking {path_str}")
    pkl_files_for_path = glob.glob(path_str)
    pkl_files.extend(pkl_files_for_path)
    print(f"Found {len(pkl_files_for_path)} pkl files in path: {path}")

    if args.episode_id:
        pkl_files = [pkl_file for pkl_file in pkl_files if args.episode_id in pkl_file]

    print(f"Total pkl files: {len(pkl_files)}")
    if nworkers == 1:
        res = []
        for pkl_file in tqdm(pkl_files, total=len(pkl_files)):
            res.append(process_file(pkl_file, args))
    else:
        with Pool(nworkers) as p:
            res = list(
                tqdm(
                    p.imap(partial(process_file, args=args), pkl_files),
                    total=len(pkl_files),
                )
            )
    res = np.array(res)
    total_samples = res.sum()
    successful_pkl_files = (res > 0).sum()

    print(
        f"Wrote {total_samples} total samples across {len(pkl_files)} folders in {args.output_dir}"
    )
    print(f"Number of successful pkl files: {successful_pkl_files}")
    print(f"Success rate: {successful_pkl_files / len(pkl_files):.2%}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--path", type=str, required=True, help="Path for the pkl files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory to save output json files",
    )

    parser.add_argument(
        "--react",
        action="store_true",
        help="Whether the traces should be in react style",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="number of worker processes to use (default: 1)",
    )
    parser.add_argument(
        "--percent_cut",
        type=float,
        default=1.0,
        help="up to what success rate we should be filtering (default: 1)",
    )

    parser.add_argument(
        "--debug_info",
        action="store_true",
        help="output the resulting text in zero-shot format",
    )
    parser.add_argument(
        "--last-action-only",
        action="store_true",
        help="process only the last action for agent 0",
    )
    parser.add_argument(
        "--last-step",
        action="store_true",
        help="process only the last step for agent 0",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="process only the last action for agent 0",
    )

    parser.add_argument(
        "--episode-id",
        type=str,
        help="process only the specified episode ID",
    )

    args = parser.parse_args()
    if not args.react:
        process_files(args)
    else:
        process_directory_react(args.path, args.output_dir, args.percent_cut)
