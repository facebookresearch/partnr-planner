# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import argparse
import json
import os
import re

from tqdm import tqdm


def extract_assistant_text_end(text):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>.*?<\|eot_id\|>"
    matches = re.finditer(pattern, text, re.DOTALL)
    end_indices = [m.end() for m in matches]
    return end_indices


# New code to process all text files in the specified directory
def process_directory(directory, output_directory, pc_filter=0.75):
    stats_directory = os.path.join(directory, "stats")
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
                output_file = os.path.join(output_directory, ep_id, f"sample_{i}.txt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(sample)

    print(
        f"Processed traces saved to: {output_file}. {good_episodes} good episodes out of {total_episodes} total episodes. {good_episodes/total_episodes*100:.2f}%"
    )
    print(f"Total traces: {total_traces}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files in a directory.")
    parser.add_argument("directory", help="Path to the directory containing text files")
    parser.add_argument(
        "output_directory", help="Path to the directory to save processed traces"
    )
    parser.add_argument(
        "--pc_filter",
        type=float,
        default=1.0,
        help="Filter out traces below this percent complete",
    )
    args = parser.parse_args()

    process_directory(args.directory, args.output_directory, args.pc_filter)
