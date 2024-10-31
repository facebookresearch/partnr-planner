#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree


import glob
import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def get_max_token_length(dataset: Dataset, tokenizer: AutoTokenizer):
    """
    Obtain the maximum length of a dataset, based on a given tokenizer.
    :param dataset: the dataset to consider
    :param tokenizer: the tokenizer we will use
    """
    lengths = [len(tokenizer.encode(example["text"])) for example in dataset]
    split_max_length = max(lengths)
    return split_max_length


def clip_and_filter_datasets(datasets: DatasetDict, tokenizer: AutoTokenizer, config):
    """
    Makes sure all the datasets have less tokens than the max allowed length
    :param datasets: the dataset to consider
    :param tokenizer: the tokenizer we will use
    :param config: the dataset config
    """
    # Find the max length across all datasets
    max_length = 0

    for split, dataset in datasets.items():
        split_max_length = get_max_token_length(dataset, tokenizer)
        print(f"Max length in {split}: {split_max_length}")
        max_length = max(max_length, split_max_length)
    print(f"Overall max length across all datasets: {max_length}")

    # Update max_tokens if necessary
    if not config.should_clip_dataset:
        if max_length > config.llm_config.generation_params.max_tokens:
            print(
                f"Updating max_tokens from {config.llm_config.generation_params.max_tokens} to {max_length}"
            )
            config.llm_config.generation_params.max_tokens = max_length
    else:

        def filter_length(example):
            output = tokenizer.encode(example["text"])
            return len(output) <= config.llm_config.generation_params.max_tokens

        for split in ["train", "validation", "train_subset"]:
            if split in datasets:
                original_count = len(datasets[split])
                datasets[split] = datasets[split].filter(filter_length)
                filtered_count = len(datasets[split])
                percentage = (filtered_count / original_count) * 100
                print(f"{split}: {percentage:.2f}% below max length")

    return datasets


def build_dataset_simple(
    path, val=None, train=None, max_val_size=1000, max_train_size=-1
):
    """
    Build a dataset for LLM completion. The dataset is composed of a set of txt
    files under the path: {path}/{val/train}, every text file is a separate training sample.
    :param val: list of paths of the validation set
    :param train: list of paths of the train set
    """
    random.seed(0)
    PERCENT_VAL = 0.1
    SIZE_TRAIN_SUBSET = 100

    if val is None or train is None:
        # If we only provide a path, we will use the full path to build the dataset and make
        # a split
        files_all = glob.glob(f"{path}/*/*/*.txt")
        dataset = load_dataset("text", data_files=files_all, sample_by="document")[
            "train"
        ]

        # Split the dataset into train and validation sets
        train_test_split = dataset.train_test_split(test_size=PERCENT_VAL, seed=42)
        dataset_train = train_test_split["train"]
        dataset_val = train_test_split["test"]

    else:
        # Otherwise, we build the train and test set with the splits available.
        files_train = []
        files_val = []
        for file_train in train:
            path_data = f"{path}/{file_train}/*/*.txt"
            c_files = glob.glob(path_data)
            files_train += c_files
            print(f"{len(c_files)} files from: {path_data}")

        if max_train_size != -1:
            episode_names = list(
                {os.path.dirname(episode_name) for episode_name in files_train}
            )
            random.shuffle(episode_names)
            if max_train_size < len(episode_names):
                episode_names = episode_names[:max_train_size]
            files_train = [
                file_name
                for file_name in files_train
                if os.path.dirname(file_name) in episode_names
            ]
            print(f"Reducing training set to {len(episode_names)} episodes.")

        for file_val in val:
            path_data = f"{path}/{file_val}/*/*.txt"
            c_files = glob.glob(path_data)
            files_val += c_files
            print(f"{len(c_files)} files from: {path_data}")

        dataset_val = load_dataset("text", data_files=files_val, sample_by="document")[
            "train"
        ]

        dataset_train = load_dataset(
            "text", data_files=files_train, sample_by="document"
        )["train"]

    # Create a train subset
    train_subset = dataset_train.shuffle(seed=42).select(range(SIZE_TRAIN_SUBSET))

    if max_val_size != -1 and max_val_size < len(dataset_val):
        print(f"Reducing validation set to {max_val_size} samples")
        dataset_val = dataset_val.shuffle(seed=42).select(range(max_val_size))

    datasets = {
        "train": dataset_train,
        "validation": dataset_val,
        "train_subset": train_subset,
    }

    return datasets
