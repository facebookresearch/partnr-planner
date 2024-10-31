#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Tuple


def calculate_scores(
    data: Dict[str, Any]
) -> Tuple[
    int,
    int,
    float,
    float,
    float,
    float,
    List[Tuple[Tuple[Any, ...], int]],
    List[Tuple[Tuple[Any, ...], int]],
]:
    """
    Compute task and evaluation scores for a collection of PrediViz annotations.
    Returns:
        total_episodes (int): total number of annotated episodes in the data
        total_frames (int): total number of annotated temporal frames in the data
        task_score_frame (float): the percent score for frame-based task correctness
        eval_score_frame (float): the percent score for frame-based eval correctness
        task_score_ep (float): the percent score for episode-based task correctness
        eval_score_ep (float): the percent score for frame-based eval correctness
        top_incorrect_task_remarks (List[Tuple[Tuple[Any, ...], int]]): description
            and frequency of task failure modes.
        top_incorrect_eval_remarks (List[Tuple[Tuple[Any, ...], int]]): description
            and frequency of evaluation failure modes.
    """
    tc_frame, ec_frame = [], []  # tc: task correctness, ec: eval correctness
    tc_ep, ec_ep = [], []
    task_remarks, eval_remarks = [], []
    for episode_annotation in data.values():
        task_ep, eval_ep, has_annotation = True, True, False
        for step_annotation in episode_annotation.values():
            has_task_ann = "task_correctness" in step_annotation and step_annotation[
                "task_correctness"
            ] in {"yes", "no"}
            has_eval_ann = "eval_correctness" in step_annotation and step_annotation[
                "eval_correctness"
            ] in {"yes", "no"}
            if not has_task_ann or not has_eval_ann:
                continue

            correct_task = step_annotation["task_correctness"] == "yes"
            correct_eval = step_annotation["eval_correctness"] == "yes"
            tc_frame.append(correct_task)
            if not correct_task:
                task_ep = False
            ec_frame.append(correct_eval)
            if not correct_task and len(step_annotation["task_remarks"]):
                task_remarks.append(tuple(step_annotation["task_remarks"]))
            if not correct_eval:
                eval_ep = False
            if (
                correct_task
                and not correct_eval
                and ",".join(step_annotation["eval_remarks"]).strip()
            ):
                eval_remarks.append(tuple(step_annotation["eval_remarks"]))
            has_annotation = True
        if has_annotation:
            tc_ep.append(task_ep)
            ec_ep.append(eval_ep)

    total_frames = len(tc_frame)
    total_episodes = len(tc_ep)

    # Evaluation correctness is based on only the correct tasks
    evals_correct = sum(tc and ec for tc, ec in zip(tc_frame, ec_frame))
    evals_correct = sum(tc and ec for tc, ec in zip(tc_ep, ec_ep))
    return (
        total_episodes,
        total_frames,
        100 * sum(tc_frame) / total_frames,
        100 * evals_correct / sum(tc_frame),
        100 * sum(tc_ep) / total_episodes,
        100 * evals_correct / sum(tc_ep),
        Counter(task_remarks).most_common(3),
        Counter(eval_remarks).most_common(3),
    )


def display_scores(
    total_tasks: int,
    total_frames: int,
    tc_frame: float,
    ec_frame: float,
    tc_episode: float,
    ec_episode: float,
    top_incorrect_task_remarks: List[Tuple[Tuple[Any, ...], int]],
    top_incorrect_eval_remarks: List[Tuple[Tuple[Any, ...], int]],
) -> None:
    msg = f"""Completed Annotations
Episodes:    {total_tasks}
Frames:      {total_frames}

Task Correctness
Per Episode: {tc_episode:.2f}%
Per Frame:   {tc_frame:.2f}%

Eval Correctness
Per Episode: {ec_episode:.2f}%
Per Frame:   {ec_frame:.2f}%
"""
    print(msg)
    print("Task Failures")
    if len(top_incorrect_task_remarks) == 0:
        print("No remarks")
    for remark, count in top_incorrect_task_remarks:
        print(f"- {remark}: {count}")
    print("\nEval Failures")
    if len(top_incorrect_eval_remarks) == 0:
        print("No remarks")
    for remark, count in top_incorrect_eval_remarks:
        print(f"- {remark}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate annotation scores of a single file."
    )
    parser.add_argument(
        "--annotation-file",
        required=True,
        type=str,
        help="Path to the annotation file",
    )
    args = parser.parse_args()
    with open(args.annotation_file, "r") as f:
        data = json.load(f)
    display_scores(*calculate_scores(data))


if __name__ == "__main__":
    main()
