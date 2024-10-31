#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import glob
import json
import os

import hydra
import omegaconf
import regex as re
from omegaconf import OmegaConf

goal_strings = [
    "help me clean up",
    "help me tidy",
    "help me prepare",
    "help me create",
    "help me set",
    "help me make",
    "help me organize",
    "let's prepare",
    "let's decorate",
    "let's create",
    "let's tidy",
    "let's clean",
    "let's set",
    "let's organize",
    "let's make",
    "put away",
]

invalid_actions = [
    "squeeze",
    "answer",
    "plug",
    "unplug",
    "connect",
    "call",
    "hang",
    "read",
    "charge",
    "feed",
    "bake",
    "peel",
    "light",
    "stir",
    "fix",
    "eat",
    "use",
    "assemble",
    "human",
    "robot",
    "wrist",
    "wrap",
    "draw",
    "old",
    "open",
    "closed",
]

invalid_ordering_terms = ["Then,", "After that,", "Next,", "Finally,"]


def remove_chars(my_string):
    # Remove underscores
    my_string = my_string.replace("_", " ")

    # Remove numbers
    remove_digits = str.maketrans("", "", "0123456789")
    my_string = my_string.translate(remove_digits)

    return my_string


def filter_dataset(samples_dict):
    # iterate through the dataset and validate each instruction
    valid_episodes = []
    existing_instructions = []
    valid_instructions = []
    valid_instructions_lower = []
    total_invalid_count = 0
    repeat_instruction = 0

    for sample in samples_dict:
        if "ignore" in sample:
            continue
        instruction = sample["instruction"]
        existing_instructions.append(instruction)
        valid, new_instruction = validate_instruction(instruction)
        # check for pure/exact matching duplicate instructions too
        if valid and new_instruction.lower() not in valid_instructions_lower:
            sample["instruction"] = new_instruction
            valid_episodes.append(sample)
            valid_instructions.append(new_instruction)
            valid_instructions_lower.append(new_instruction.lower())
        elif valid:
            repeat_instruction += 1
        else:
            valid_instructions.append("invalid instruction!")
            total_invalid_count += 1

    total_valid_count = len(existing_instructions)
    print(
        "Found",
        total_invalid_count,
        "invalid instructions out of total:",
        total_valid_count,
        "Found",
        repeat_instruction,
        "repeated instructions.",
    )
    if total_valid_count > 0:
        print(
            "Total percent of valid instructions:",
            (total_valid_count - total_invalid_count - repeat_instruction)
            / total_valid_count,
        )

    return valid_episodes


def validate_instruction(instruction):
    # check each instruction
    # if it is valid, add it back to the dataset

    # remove "_" and numbers
    instruction = remove_chars(instruction)

    # Remove instructions that have invalid actions such as "squeeze"
    # defined in the list above.
    bad_chars = r"[!]"
    instruction = re.sub(bad_chars, ". ", instruction)
    # all_steps = instruction.split(".")
    all_steps = re.split(r"[.?]", instruction)
    found_valid_instruction = False
    valid_steps_0 = [
        step
        for step in all_steps
        if not bool(set(invalid_actions) & set(step.lower().split(" ")))
        or ("spray" in step)
        and ("spray bottle" in step)
        and (step.lower().index("spray") == step.lower().index("spray bottle"))
    ]

    # check if "bring/fetch/pass/get" is used without ambiguity
    post_specifier = ["put", "place", "to"]
    pre_specifier = ["pack"]
    specifiers = post_specifier + pre_specifier
    getter = ["bring", "fetch", "pass", "get", "give"]
    valid_steps = valid_steps_0.copy()

    for stp in valid_steps_0:
        for get_spec in getter:
            # bring, fetch, pass, get must be followed by put, place, to
            # or preceeded by pack
            if get_spec in stp.lower():
                if not any(spec in stp.lower() for spec in specifiers):  # noqa: SIM114
                    valid_steps.remove(stp)
                    break

                if not any(
                    stp.lower().index(spec) > stp.lower().index(get_spec)
                    for spec in post_specifier
                    if spec in stp.lower() and get_spec in stp.lower()
                ) and not any(
                    stp.lower().index(spec) < stp.lower().index(get_spec)
                    for spec in pre_specifier
                    if spec in stp.lower()
                    and get_spec in stp.lower()
                    and "me" not in stp.lower()
                ):
                    valid_steps.remove(stp)
                    break

    # check if the dropped steps led to incorrect ordering in the overall instruction
    goal_present = False
    for goal in goal_strings:
        if re.search(r"\b" + goal + r"\b", valid_steps[0].lower()):
            goal_present = True
            break
    if len(valid_steps) >= 1:
        for order in invalid_ordering_terms:
            if order in valid_steps[0]:
                valid_steps[0] = str.capitalize(
                    valid_steps[0].replace(order, "").lstrip()
                )
            if len(valid_steps) > 1 and order in valid_steps[1] and goal_present:
                valid_steps[1] = str.capitalize(
                    valid_steps[1].replace(order, "").lstrip()
                )

        # check if the instruction is too abstract
        # abstract_specifier = ["prepare", "make", "organize", "create", "decorate", "put away"]
        # ensure there is at least one more step explaining what make/prepare entails
        # Note: last step is usually '' in the valid_steps
        required_steps_len = 1
        if valid_steps[-1] == "":
            required_steps_len += 1
        if goal_present:
            required_steps_len += 1
        if len(valid_steps) < required_steps_len:
            valid_steps = [""]

    ##super special case#1 to handle "put away"
    # allow put away in goal description (first step of the instruction)
    # put away otherwise should be followed by "to" or "in"
    put_away_specifier = ["to", "in"]
    for ind, stp in enumerate(valid_steps):
        if (
            "away" in stp
            and ind != 0
            and not any(
                stp.lower().index(spec) > stp.lower().index("away")
                for spec in put_away_specifier
                if re.search(r"\b" + spec + r"\b", stp.lower())
            )
        ):
            found_valid_instruction = False
            return found_valid_instruction, ""

    # add question mark back if the instruction was a question
    ques_specifiers = ["can", "could", "would", "will"]
    for stp_id, stp in enumerate(valid_steps):
        if any(spec in stp.lower() for spec in ques_specifiers):
            valid_steps[stp_id] += "?"
        elif stp == "":
            continue
        else:
            valid_steps[stp_id] += "."

    validated_instruction = " ".join(valid_steps)
    if len(validated_instruction) > 1:
        found_valid_instruction = True

        ##super special case#2 -- handle "respective" and "respectively"
        if (
            "respective" in validated_instruction
            and "respectively" not in validated_instruction
        ):
            found_valid_instruction = False

        print("validated instruction! ", validated_instruction)
    else:
        print("invalid instruction: ", instruction)

    return found_valid_instruction, validated_instruction


@hydra.main(
    version_base=None,
    config_path="../conf/",
    config_name="benchmark_gen.yaml",
)
def main(cfg: omegaconf.DictConfig):
    inst_gen_config = OmegaConf.create(cfg)
    output_path = inst_gen_config.generator.output_path

    # loop over all scenes in the output path
    scenes = [
        scene
        for scene in glob.glob(f"{output_path}/*")
        if "yaml" not in scene and "json" not in scene and "csv" not in scene
    ]

    for scene in scenes:
        # loop over all parsed files in folder
        init_state_dicts = []
        parsed_folder = f"{scene}/output_parsed/"
        for file in os.listdir(parsed_folder):
            if file.endswith(".json"):
                with open(os.path.join(parsed_folder, file), "r") as f:
                    json_data = json.load(f)
                    init_state_dicts.append(json_data)

        # obtain filtered dataset, append scene and episode id
        filtered_dataset = []
        filtered_dicts = filter_dataset(init_state_dicts)
        for eid, init_state in enumerate(filtered_dicts):
            init_state["scene_id"] = scene.split("/")[-1]
            init_state["episode_id"] = eid
            filtered_dataset.append(init_state)

        # save filtered dataset
        output_folder = f"{scene}/output_filtered/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(f"{output_folder}/filtered_dicts.json", "w") as f:
            json.dump(filtered_dataset, f, indent=4)


if __name__ == "__main__":
    main()
