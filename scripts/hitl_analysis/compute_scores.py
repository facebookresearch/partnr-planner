# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import stats


def print_statistics(
    mean: float, std_err: float, confidence_interval: Tuple[float, float]
) -> None:
    """
    Print the mean, standard deviation, and 95% confidence interval in a nice format.

    Args:
        mean (float): The mean value.
        std_dev (float): The standard deviation value.
        confidence_interval (tuple): A tuple containing the lower and upper bounds of the 95% confidence interval.
    """

    # Extract the lower and upper bounds of the confidence interval
    lower_bound, upper_bound = confidence_interval

    # Print the statistics
    print(f"Mean: {mean:.4f}")
    print(f"Standard Error: {std_err:.4f}")
    print(f"95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")


def compute_statistics(
    data: List[Union[float, int]]
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate the mean, standard deviation, and 95% confidence interval for a given list of numbers.

    Args:
        data (list): A list of numbers.

    Returns:
        tuple: A tuple containing the mean, standard deviation, and 95% confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    std_err = std_dev / np.sqrt(len(data))
    if std_err == 0.0:
        confidence_interval = (mean, mean)
    else:
        confidence_interval = stats.t.interval(
            0.95, len(data) - 1, loc=mean, scale=std_err
        )

    print_statistics(mean, std_err, confidence_interval)

    return mean, std_err, confidence_interval


def compute_scores(metrics_file: str) -> None:
    with open(metrics_file) as f:
        metrics = json.load(f)

    success_at_round: Dict[int, int] = defaultdict(int)
    for eid, r in metrics["eid_to_round_needed"].items():
        # add up a round-based success count. later divide by n_eps to get accuracy
        if r == -1:
            continue
        success_at_round[r] += metrics["eid_to_success"][eid]

    print(f"Total episodes: {len(metrics['eid_to_success'])}")

    print("\n\n---Success Rate---")
    success = list(metrics["eid_to_success"].values())
    compute_statistics(success)
    print("After round:")
    prev = 0.0
    for i in range(1, max(success_at_round.keys())):
        prev += success_at_round[i] / len(metrics["eid_to_success"])
        print(f" {i}: {prev:.4f}")

    print("\n\n---Percent Complete---")
    pc = list(metrics["eid_to_pc"].values())
    compute_statistics(pc)

    print("\n\n---Portion of tasks by robot (offloading metric)---")
    pc = list(metrics["ratio_agent_0"].values())
    compute_statistics(pc)

    print("\n\n---Portion of tasks by human---")
    pc = list(metrics["ratio_agent_1"].values())
    compute_statistics(pc)

    print("\n\n---Total Steps---")
    pc = list(metrics["remaining_num"].values())
    compute_statistics(pc)

    print("\n\n---Exploration Efficiency---")
    pc = list(metrics["explore_steps"].values())
    compute_statistics(pc)

    print("\n\n---Extraneous Actions---")
    pc = list(metrics["ratio_extraneous_actions"].values())
    compute_statistics(pc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-path",
        type=str,
        required=False,
        default="data/hitl_data/2024-10-02-object-states/p5_single_train_10k",
    )
    args = parser.parse_args()

    metrics_file = os.path.join(
        args.collection_path, "processed", "processed_metrics.json"
    )
    compute_scores(metrics_file)
