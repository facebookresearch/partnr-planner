#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os

from tqdm import tqdm


def calculate_averages(folder_path):
    json_files = glob.glob(
        os.path.join(folder_path, "**", "stats", "*.json"), recursive=True
    )
    print("Number of finished episodes: ", len(json_files))
    data = []
    for f in tqdm(json_files, desc="Processing files"):
        with open(f, "r") as file:
            data.append(json.load(file))

    # Filter out rows without 'stats' key
    valid_data = [row for row in data if "stats" in row and row["stats"] != "{}"]
    for row in valid_data:
        row["stats"] = json.loads(row["stats"])

    total_count = len(data)
    valid_count = len(valid_data)

    if total_count == 0:
        print("No data found.")
        return

    print(f"Percentage of non-crashed data: {valid_count/total_count}")

    if not valid_data:
        print("No valid episodes found.")
        return

    task_percent_complete_sum = sum(
        float(row["stats"]["task_percent_complete"]) for row in valid_data
    )
    task_state_success_sum = sum(
        float(row["stats"]["task_state_success"]) for row in valid_data
    )
    sim_steps_sum = sum(int(row["stats"]["sim_step_count"]) for row in valid_data)
    replanning_count_sum = sum(
        int(row["stats"]["replanning_count_0"]) for row in valid_data
    )

    avg_task_percent_complete = task_percent_complete_sum / valid_count
    avg_task_state_success = task_state_success_sum / valid_count
    avg_sim_steps = sim_steps_sum / valid_count
    avg_replanning_count = replanning_count_sum / valid_count

    print(
        f"Average task_percent_complete for non-crashed episodes: {avg_task_percent_complete:.2f}"
    )
    print(
        f"Average task_state_success for non-crashed episodes: {avg_task_state_success:.2f}"
    )
    print(f"Average sim_steps for non-crashed episodes: {avg_sim_steps:.2f}")
    print(
        f"Average replanning_count for non-crashed episodes: {avg_replanning_count:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average stats from JSON files in a folder"
    )
    parser.add_argument("folder_path", help="Path to the folder containing stats files")
    args = parser.parse_args()

    calculate_averages(args.folder_path)
