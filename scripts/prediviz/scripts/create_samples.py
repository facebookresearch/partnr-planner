#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

import argparse
import json
import os
import random
import shutil

import numpy as np

README_ASSETS_DIR = "scripts/prediviz/assets/readme_assets"


def create_samples(
    output_dir: str, sample_set_size: int, num_directories: int, overlap_samples: int
) -> None:
    with open(os.path.join(output_dir, "run_data.json"), "r") as f:
        data = json.load(f)

    episodes = data["episodes"]
    if sample_set_size <= 0:
        sample_set = random.sample(episodes, len(episodes))
    else:
        sample_set = random.sample(episodes, sample_set_size)

    random.shuffle(sample_set)

    common_examples = sample_set[:overlap_samples]
    exclusive_examples = sample_set[overlap_samples:]
    exclusive_lists = []
    start_index = 0
    num_examples_per_list = len(exclusive_examples) // num_directories
    for i in range(num_directories):
        end_index = start_index + num_examples_per_list
        if i < len(exclusive_examples) % num_directories:
            end_index += 1
        exclusive_lists.append(exclusive_examples[start_index:end_index])
        start_index = end_index

    for list_idx, exclusive_example_list in enumerate(exclusive_lists):
        sample_dir = os.path.join(output_dir, f"sample_{list_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        assets_dir = os.path.join(sample_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        new_episodes = []
        for episode_idx, episode in enumerate(exclusive_example_list):
            new_episode: dict = {}
            for key, value in episode.items():
                if key == "viz_paths":
                    new_episode[key] = {}
                    for step_id, image_path in value.items():
                        image_name = image_path.split("/")[-1]  # step ID
                        current_idx = episode_idx
                        os.makedirs(
                            os.path.join(assets_dir, f"viz_{current_idx}"),
                            exist_ok=True,
                        )
                        new_episode["sample_idx"] = current_idx
                        new_image_name = image_name
                        new_image_path = os.path.join(
                            assets_dir, f"viz_{current_idx}", f"{new_image_name}"
                        )
                        shutil.copy(image_path, new_image_path)
                        new_episode[key][step_id] = os.path.join(
                            "./assets", f"viz_{current_idx}", f"{new_image_name}"
                        )

                else:
                    new_episode[key] = value
            new_episodes.append(new_episode)

        for episode_idx, episode in enumerate(common_examples):
            new_episode = {}
            for key, value in episode.items():
                if key == "viz_paths":
                    new_episode[key] = {}
                    for step_id, image_path in value.items():
                        image_name = image_path.split("/")[-1]  # step ID
                        current_idx = episode_idx + len(exclusive_example_list)
                        os.makedirs(
                            os.path.join(assets_dir, f"viz_{current_idx}"),
                            exist_ok=True,
                        )
                        new_episode["sample_idx"] = current_idx
                        new_image_name = image_name
                        new_image_path = os.path.join(
                            assets_dir, f"viz_{current_idx}", f"{new_image_name}"
                        )
                        shutil.copy(image_path, new_image_path)
                        new_episode[key][step_id] = os.path.join(
                            "./assets", f"viz_{current_idx}", f"{new_image_name}"
                        )
                else:
                    new_episode[key] = value
            new_episodes.append(new_episode)

        len_new_episodes = len(new_episodes)
        print(f"For Sample: {list_idx}")
        print("Total Episode:", len_new_episodes)
        print("Unique scenes:", len(np.unique([ep["scene_id"] for ep in new_episodes])))
        print("-" * 30)
        files_to_copy = [
            ("scripts/prediviz/interface/interface.html", "interface.html"),
            ("scripts/prediviz/interface/server.py", "server.py"),
            ("scripts/prediviz/interface/README.md", "README.md"),
            (f"{README_ASSETS_DIR}/annotation_tool.png", "annotation_tool.png"),
            (f"{README_ASSETS_DIR}/receptacle_collage.png", "receptacle_collage.png"),
        ]

        for src, dst in files_to_copy:
            shutil.copy(src, os.path.join(sample_dir, dst))

        new_json_file = os.path.join(sample_dir, "sample_episodes.json")
        with open(new_json_file, "w") as f:
            json.dump({"episodes": new_episodes}, f, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create annotation trials")
    parser.add_argument(
        "--viz-dir",
        required=True,
        type=str,
        help="Path to the PrediViz visualization directory",
    )
    parser.add_argument(
        "--sample-size",
        required=False,
        type=int,
        default=0,
        help="If only a random subset of all the episodes is to be visualized, the sample size.",
    )
    parser.add_argument(
        "--num-directories",
        required=False,
        type=int,
        default=1,
        help="How many annotation directories to produce",
    )
    parser.add_argument(
        "--overlap-samples",
        required=False,
        type=int,
        default=0,
        help="How many samples should be shared across annotation directories",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    create_samples(
        args.viz_dir, args.sample_size, args.num_directories, args.overlap_samples
    )
