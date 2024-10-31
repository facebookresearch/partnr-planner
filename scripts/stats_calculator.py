#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from collections import defaultdict

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


def extract_episode_id(filename):
    # Split the filename by underscores
    parts = filename.split("_")
    # The episode ID should be the second element after the split
    if len(parts) > 1:
        return int(parts[1])
    else:
        return None


def get_value(df, filename, prop):
    ep_id = extract_episode_id(filename)

    # Find the row with the matching Episode_id
    row = df[df["episode_id"] == ep_id]
    if not row.empty:
        # Retrieve the 'success' value
        return row[prop].values[0]
    else:
        return None


def get_skill_histogram(folder_path, filename):
    """
    This method returns a histogram of skill counts
    """
    valid_actions = {
        "rearrange",
        "pick",
        "place",
        "navigate",
        "open",
        "close",
        "explore",
        "findobjecttool",
        "findreceptacletool",
        "findroomtool",
        "findagentactiontool",
        "wait",
    }
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
        steps = data.get("steps", [])
        skill_dict = defaultdict(int)
        for step in steps:
            action = step.get("high_level_actions", {})
            replan_required = step.get("replan_required", False)
            for agent_id in ["0"]:
                agent_action = action.get(agent_id, [])
                # Check for action name errors and construct the action string
                if (
                    replan_required
                    and len(agent_action) > 0
                    and agent_action[0]
                    and agent_action[0].lower() in valid_actions
                ):
                    action_name = agent_action[0].lower()
                    skill_dict[action_name] += 1

    return skill_dict


def get_specific_skill_count(hist, skills):
    """
    This method returns total number of assignments for given skills
    """
    if len(skills) == 0:
        raise ValueError("Variable skills cannot be empty list")

    count = 0
    for skill in skills:
        count += hist[skill]

    return count


def get_motor_skill_count(hist):
    motor_actions = {
        "rearrange",
        "pick",
        "place",
        "navigate",
        "open",
        "close",
        "explore",
    }

    return get_specific_skill_count(hist, motor_actions)


def get_all_skill_count(hist):
    all_actions = {
        "rearrange",
        "pick",
        "place",
        "navigate",
        "open",
        "close",
        "explore",
        "findobjecttool",
        "findreceptacletool",
        "findroomtool",
        "findagentactiontool",
        "wait",
    }

    return get_specific_skill_count(hist, all_actions)


def calculate_column_averages(csv_file):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Initialize an empty dictionary to store column averages
    column_averages = {}

    # Iterate through columns and calculate average for numeric columns
    for column in df.columns:
        # Skip unwanted columns
        if column in ["episode_id", "run_id"]:
            continue
        # Calculate averages
        if pd.api.types.is_numeric_dtype(df[column]):
            column_averages[column] = df[column].mean()

    return column_averages, df


def count_successful_actions(folder_path):
    # Initialize counts for both agents
    successful_rearranges = {"0": 0, "1": 0}
    successful_places = {"0": 0, "1": 0}

    # Iterate over all files in the given folder
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            steps = data.get("steps", [])

            # Track ongoing actions for each agent
            action_in_progress = {"0": None, "1": None}

            for step in steps:
                action = step.get("high_level_actions", {})
                response = step.get("responses", {})

                for agent_id in ["0", "1"]:
                    agent_action = action.get(agent_id, [])

                    # Skip if the list is empty
                    if len(agent_action) == 0:
                        continue

                    # Skip if the action is None
                    if agent_action[0] is None:
                        continue

                    # Check for rearrange or place action
                    if "rearrange" in agent_action[0].lower():
                        action_in_progress[agent_id] = "rearrange"
                    elif "place" in agent_action[0].lower():
                        action_in_progress[agent_id] = "place"

                    agent_response = response.get(agent_id, "")

                    # Check for successful execution if an action is in progress
                    if (
                        action_in_progress[agent_id]
                        and "successful execution" in agent_response.lower()
                    ):
                        if action_in_progress[agent_id] == "rearrange":
                            successful_rearranges[agent_id] += 1
                        elif action_in_progress[agent_id] == "place":
                            successful_places[agent_id] += 1
                        action_in_progress[
                            agent_id
                        ] = None  # Reset after successful execution

    return successful_rearranges, successful_places


def count_common_successful_actions(folder_path):
    # Initialize sets to track successful action descriptions for both agents
    successful_actions_agent_0 = set()
    successful_actions_agent_1 = set()

    # Iterate over all files in the given folder
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            steps = data.get("steps", [])

            # Track ongoing actions for each agent
            action_in_progress = {"0": None, "1": None}

            for step in steps:
                action = step.get("high_level_actions", {})
                response = step.get("responses", {})

                for agent_id in ["0", "1"]:
                    agent_action = action.get(agent_id, [])

                    # Skip if the list is empty or action is None
                    if len(agent_action) < 2 or agent_action[0] is None:
                        continue

                    # Check for rearrange or place action
                    if (
                        "rearrange" in agent_action[0].lower()
                        or "place" in agent_action[0].lower()
                    ):
                        action_description = f"{agent_action[0]} {agent_action[1]}"
                        action_in_progress[agent_id] = action_description

                    agent_response = response.get(agent_id, "")

                    # Check for successful execution if an action is in progress
                    if (
                        action_in_progress[agent_id]
                        and "successful execution" in agent_response.lower()
                    ):
                        if agent_id == "0":
                            successful_actions_agent_0.add(action_in_progress[agent_id])
                        elif agent_id == "1":
                            successful_actions_agent_1.add(action_in_progress[agent_id])
                        action_in_progress[
                            agent_id
                        ] = None  # Reset after successful execution

    # Find common successful actions between both agents
    common_successful_actions = successful_actions_agent_0.intersection(
        successful_actions_agent_1
    )

    return len(common_successful_actions), common_successful_actions


def count_agent_collisions(folder_path):
    total_collisions = 0
    total_file_count = 0

    # Iterate over all files in the given folder
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            total_file_count += 1
            steps = data.get("steps", [])
            previous_collision_state = {"0": False, "1": False}
            current_collision_state = {"0": False, "1": False}

            for step in steps:
                agent_collisions = step.get(
                    "agent_collisions", {"0": False, "1": False}
                )

                # Update current collision states
                current_collision_state["0"] = agent_collisions.get("0", False)
                current_collision_state["1"] = agent_collisions.get("1", False)

                # Check for transition from non-contact to contact
                if (
                    not previous_collision_state["0"] and current_collision_state["0"]
                ) or (
                    not previous_collision_state["1"] and current_collision_state["1"]
                ):
                    total_collisions += 1

                # Update previous collision states for the next iteration
                previous_collision_state["0"] = current_collision_state["0"]
                previous_collision_state["1"] = current_collision_state["1"]

    return total_collisions / total_file_count


def count_hallucination_errors(df, folder_path, k=5):
    valid_actions = {
        "rearrange",
        "pick",
        "place",
        "navigate",
        "open",
        "close",
        "explore",
        "findobjecttool",
        "findreceptacletool",
        "findroomtool",
        "findagentactiontool",
        "wait",
    }
    object_name_errors = [
        "not present in the graph",
        "Use the appropriate tool to get a valid name",
        "This may not be the correct node name, try using appropriate tool to get the exact name. If that doesnt work, this node may just not exist yet, explore the house to discover.",
        "The entity name may be wrong or the entity may not exist in the house",
    ]

    action_directive_syntax_errors = [
        "Invalid Agent ID in Action directive. Only valid Agent IDs are",
        "SyntaxError in Action directive",
        "Wrong use of API for rearrange tool",
        "Incorrect syntax for place/rearrange skill",
        "Wrong use of API for place or rearrange",
    ]

    error_counts = {
        "object_name_errors": 0,
        "action_name_errors": 0,
        "action_directive_syntax_errors": 0,
    }

    files_with_more_than_k_hallucinations = 0
    files_with_more_than_k_hallucinations_and_success = 0
    total_file_count = 0

    # Iterate over all files in the given folder
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            steps = data.get("steps", [])

            file_hallucination_count = 0
            total_file_count += 1

            # Get how many llm calls were made in this episode
            hist = get_skill_histogram(folder_path, filename)
            total_skill_count = get_all_skill_count(hist)

            for step in steps:
                action = step.get("high_level_actions", {})
                response = step.get("responses", {})

                for agent_id in ["0"]:
                    agent_action = action.get(agent_id, [])
                    agent_response = response.get(agent_id, "")

                    # Check for action name errors
                    if (
                        len(agent_action) > 0
                        and agent_action[0]
                        and agent_action[0].lower() not in valid_actions
                    ):
                        error_counts["action_name_errors"] += 1 / total_skill_count
                        # print(agent_action[0])
                        file_hallucination_count += 1

                    # Check for object name errors
                    for error_message in object_name_errors:
                        if error_message.lower() in agent_response.lower():
                            error_counts["object_name_errors"] += 1 / total_skill_count
                            file_hallucination_count += 1
                            break

                    # Check for action directive errors
                    for error_message in action_directive_syntax_errors:
                        if error_message.lower() in agent_response.lower():
                            error_counts["action_directive_syntax_errors"] += (
                                1 / total_skill_count
                            )
                            file_hallucination_count += 1
                            break

        if file_hallucination_count > k:
            files_with_more_than_k_hallucinations += 1
            if get_value(df, filename, "task_state_success"):
                files_with_more_than_k_hallucinations_and_success += 1

    hall_fraction = files_with_more_than_k_hallucinations / total_file_count
    hall_fraction_with_success = (
        files_with_more_than_k_hallucinations_and_success
        / files_with_more_than_k_hallucinations
    )

    # Calculate average errors per episode
    for key in error_counts:
        error_counts[key] /= total_file_count

    return error_counts, hall_fraction, hall_fraction_with_success


def count_embodiment_errors(df, folder_path, k=10):
    embodiment_errors = [
        "The arm is currently grasping",
        "The arm is already grasping",
        "agent is not holding any object",
    ]

    reachability_errors = [
        "not close enough",
        "Furniture is closed, you need to open it first",
        "Object is in a closed furniture, you need to open the furniture first",
        "This furniture is occluded or too far from agent to",
        "is with another agent",
    ]

    affordance_errors = [
        "This is not a movable object",
        "Place receptacle is not furniture or floor",
        "You can't open",
        "You can't close",
        "is not articulated - and cannot be opened.",
        "is not articulated - and cannot be closed.",
        "No valid placements found for entity",
    ]

    error_counts = {
        # "failed_to_perform_errors": 0,
        "embodiment_errors": 0,
        "reachability_errors": 0,
        "affordance_errors": 0,
    }

    files_with_more_than_k_errors = 0
    files_with_more_than_k_errors_and_success = 0
    total_file_count = 0

    # Iterate over all files in the given folder
    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            steps = data.get("steps", [])

            file_error_count = 0
            total_file_count += 1

            # Get how many motor skill calls were made in this episode
            hist = get_skill_histogram(folder_path, filename)
            motor_skill_count = get_motor_skill_count(hist)

            for step in steps:
                response = step.get("responses", {})

                for agent_id in ["0"]:
                    agent_response = response.get(agent_id, "")

                    # Check for embodiment_errors
                    for error_message in embodiment_errors:
                        if error_message.lower() in agent_response.lower():
                            error_counts["embodiment_errors"] += 1 / motor_skill_count
                            file_error_count += 1
                            break

                    # Check for reachability ettors
                    for error_message in reachability_errors:
                        if error_message.lower() in agent_response.lower():
                            error_counts["reachability_errors"] += 1 / motor_skill_count
                            file_error_count += 1
                            break

                    # Check for affordance_errors
                    for error_message in affordance_errors:
                        if error_message.lower() in agent_response.lower():
                            error_counts["affordance_errors"] += 1 / motor_skill_count
                            file_error_count += 1
                            break

            if file_error_count > k:
                files_with_more_than_k_errors += 1
                if get_value(df, filename, "task_state_success"):
                    files_with_more_than_k_errors_and_success += 1

    error_fraction = files_with_more_than_k_errors / total_file_count
    error_fraction_with_success = (
        files_with_more_than_k_errors_and_success / files_with_more_than_k_errors
    )
    # Calculate average errors per episode
    for key in error_counts:
        error_counts[key] /= total_file_count

    return error_counts, error_fraction, error_fraction_with_success


def main():
    # Load arguments
    csv_file = sys.argv[1]
    log_folder = sys.argv[2]

    # calculate overall averages
    averages, df = calculate_column_averages(csv_file)
    total_episodes = len(pd.read_csv(csv_file))

    # Prepare data for overall stats
    overall_stats = [["Total number of episodes", total_episodes]]
    overall_stats.extend(
        [
            [f"Average of {column}", f"{average:.2f}"]
            for column, average in averages.items()
        ]
    )
    print("\n\n---------------- Overall Stats -------------------------")
    print(tabulate(overall_stats, headers=["Metric", "Value"], tablefmt="grid"))

    # calculate work division
    successful_rearranges, successful_places = count_successful_actions(log_folder)
    number_of_common_successful_actions, _ = count_common_successful_actions(log_folder)
    work_division = [
        [
            "Agent 0 successful moves",
            successful_rearranges["0"] + successful_places["0"],
        ],
        [
            "Agent 1 successful moves",
            successful_rearranges["1"] + successful_places["1"],
        ],
        [
            "Total number of successful repeated moves",
            number_of_common_successful_actions,
        ],
    ]
    print("\n\n----------------- Successful Rearrange / Place Actions ----------")
    print(tabulate(work_division, headers=["Metric", "Value"], tablefmt="grid"))

    # Print collisions
    number_of_collisions = count_agent_collisions(log_folder)
    print("\n\n----------------- Collisions ----------")
    print(
        tabulate(
            [
                [
                    "Average number of collisions per episode",
                    f"{number_of_collisions:.2f}",
                ]
            ],
            headers=["Metric", "Value"],
            tablefmt="grid",
        )
    )

    # Print hallucinations
    k = 3
    (
        hallu_counts,
        episodes_with_hallu,
        successful_episodes_with_hallu,
    ) = count_hallucination_errors(df, log_folder, k)
    hallucination_stats = [
        [
            "Average % object name errors",
            f"{hallu_counts['object_name_errors']*100:.2f} %",
        ],
        [
            "Average % action name errors",
            f"{hallu_counts['action_name_errors']*100:.2f} %",
        ],
        [
            "Average % syntax errors in action directives",
            f"{hallu_counts['action_directive_syntax_errors']*100:.2f} %",
        ],
        [
            f"Percent of episodes with at least {k} hallucinations",
            f"{episodes_with_hallu*100:.2f} %",
        ],
        [
            f"Percent of episodes with at least {k} hallucinations and successful",
            f"{successful_episodes_with_hallu*100:.2f} %",
        ],
    ]
    print("\n\n----------------- Hallucinations ----------")
    print(tabulate(hallucination_stats, headers=["Metric", "Value"], tablefmt="grid"))

    # Print embodiment errors
    (
        error_counts,
        episodes_with_errors,
        successful_episodes_with_errors,
    ) = count_embodiment_errors(df, log_folder, k)
    embodiment_errors = [
        [
            "Average % embodiment errors",
            f"{error_counts['embodiment_errors']*100:.2f} %",
        ],
        [
            "Average % reachability errors",
            f"{error_counts['reachability_errors']*100:.2f} %",
        ],
        [
            "Average % affordance errors",
            f"{error_counts['affordance_errors']*100:.2f} %",
        ],
        [
            f"Percent of episodes with at least {k} embodiment errors",
            f"{episodes_with_errors*100:.2f} %",
        ],
        [
            f"Percent of episodes with at least {k} embodiment errors and successful",
            f"{successful_episodes_with_errors*100:.2f} %",
        ],
    ]
    print("\n\n----------------- Embodiment Errors ----------")
    print(tabulate(embodiment_errors, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    main()
