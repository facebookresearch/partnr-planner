#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import pandas as pd
import regex as re

#### Following characteristics are considered in categorizing the instructions:
## avg. number of actions per task
## spatial constraints
## temporal constraints
## heterogeneous actions/human actions
## goal-oriented instructions

temporal_terms = ["Then", "After that", "Next", "Finally", "Before", "After", "When"]
spatial_terms = [
    "next to",
    "left",
    "right",
    "beside",
    "near",
    "front",
    "each side",
    "besides",
]
heterogeneous_terms = ["turn", "fill"]
goal_strings = [
    "help me clean up",
    "help me tidy",
    "help me prepare",
    "help me create",
    "help me set",
    "help me make",
    "help me organize",
    "let's decorate",
    "let's create",
    "let's tidy",
    "let's clean",
    "let's set",
    "let's organize",
]


def categorize_dataset(csv_instructions):
    df = pd.read_csv(csv_instructions)
    valid_instructions = df["new instructions"]

    total_spatial = 0
    total_temporal = 0
    total_heterogeneous = 0
    total_goalbased = 0
    total_inst = 0

    categorized_instructions = []
    for inst in valid_instructions:
        if "invalid" not in inst:
            total_inst += 1

            temporal_present = False
            spatial_present = False
            heterogeneous_present = False
            if any(spec.lower() in inst.lower() for spec in temporal_terms):
                temporal_present = True
                total_temporal += 1
            if any(
                re.search(r"\b" + spec + r"\b", inst.lower()) for spec in spatial_terms
            ):
                spatial_present = True
                total_spatial += 1
            if any(spec.lower() in inst.lower() for spec in heterogeneous_terms):
                heterogeneous_present = True

            # special handling of the word clean
            if re.search(r"\b" + "clean up" + r"\b", inst.lower()):
                if len(re.findall(r"\b" + "clean" + r"\b", inst.lower())) > 1:
                    heterogeneous_present = True
            elif re.search(r"\b" + "clean" + r"\b", inst.lower()):
                heterogeneous_present = True

            if heterogeneous_present:
                total_heterogeneous += 1

            goal_present = False
            bad_chars = r"[!\?]"
            inst_clean = re.sub(bad_chars, "", inst)
            for goal in goal_strings:
                if re.search(r"\b" + goal + r"\b", inst_clean.lower()):
                    goal_present = True
                    total_goalbased += 1
                    break

            classified_inst = {
                "instruction": inst,
                "temporal": temporal_present,
                "spatial": spatial_present,
                "heterogeneous": heterogeneous_present,
                "goal-oriented": goal_present,
            }

            print("==classified instruction:==\n")
            print(classified_inst)
            # breakpoint()
            categorized_instructions.append(classified_inst)

    print("\nTotal instructions:", total_inst)
    print("\nTotal spatial:", total_spatial)
    print("\nTotal temporal:", total_temporal)
    print("\nTotal heterogeneous:", total_heterogeneous)
    print("\nTotal goal-based:", total_goalbased)

    return categorized_instructions


if __name__ == "__main__":
    csv_with_filtered_instructions = "modified_instructions_genlarge106_temporal.csv"
    cat_instructions = categorize_dataset(csv_with_filtered_instructions)
    df_cat = pd.DataFrame.from_records(cat_instructions)
    df_cat.to_csv("categorized_" + csv_with_filtered_instructions, index=False)
