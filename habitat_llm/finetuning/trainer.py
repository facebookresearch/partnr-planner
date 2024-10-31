#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
from typing import Any, Dict, List
import hydra
import numpy as np
import torch
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from datasets import DatasetDict
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from habitat_llm.finetuning.dataset import clip_and_filter_datasets
from habitat_llm.finetuning.ft_utils import init_distrib

IGNORE_INDEX = -100


# Maybe set eval_do_concat_batches=False then we don't need to use batch_eval_metrics
class MetricsAccumulator:
    def __init__(self, tokenizer: AutoTokenizer, instruct=False):
        """
        Initialize the class to accumulate metrics
        :param tokenizer: the tokenizer used for this language model
        :param instruct: boolean indicating whether the model follows the instruct format
        """
        self.tokenizer = tokenizer
        self.instruct = instruct
        self.all_per_token_accuracies: List[Any] = []
        self.all_completion_accuracies: List[Any] = []
        self.extra_metrics: Dict[str, Any] = {
            "thought_accuracy": [],
            "action_accuracy": [],
        }

    def compute_metrics(self, eval_preds, compute_result=False):
        """
        Compute the accuracy metrics for a set of predictions.
        :param eval_preds: list of predictions from the model
        :param compute_result: whether the result should be aggregated since it was the last element of the batch
        """
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        preds = preds.argmax(-1)

        # Mask to ignore padding tokens
        mask = labels != IGNORE_INDEX
        if mask.sum().item() == 0:
            raise Exception

        # Shift the predictions right by one to line up with labels
        shifted_preds = torch.roll(preds, shifts=1, dims=-1)

        # Calculate per-token accuracy for the batch
        correct = (shifted_preds == labels) & mask
        per_token_accuracy = correct.sum().item() / mask.sum().item()
        self.all_per_token_accuracies.append(per_token_accuracy)

        # Calculate completion accuracy for the batch

        if self.instruct:
            completion_results = []
            # Find contiguous regions of the mask and process individually
            for i in range(len(labels)):
                mask_np = mask[i].cpu().numpy()
                regions = np.ma.clump_masked(np.ma.masked_array(mask_np, mask=mask_np))
                split_mask = np.zeros((len(regions), mask_np.shape[0]), dtype=bool)
                for rid, region in enumerate(regions):
                    split_mask[rid, region.start : region.stop] = True
                for sub_mask in split_mask:
                    # these shifted by one token but the search is still finding the right thing
                    label_text = self.tokenizer.decode(labels[i][sub_mask])
                    pred_text = self.tokenizer.decode(preds[i][sub_mask])

                    # regex to find the predicted action
                    pattern = r"Thought:.*?\n(.*?)\nAssigned!"

                    label_match = re.search(pattern, label_text, re.DOTALL)
                    pred_match = re.search(pattern, pred_text, re.DOTALL)

                    if label_match and pred_match:
                        label_capture = label_match.group(1)
                        pred_capture = pred_match.group(1)
                        completion_results.append(label_capture == pred_capture)
                    else:
                        completion_results.append(False)
            self.all_completion_accuracies.append(np.mean(completion_results))

            correct_completions = correct.sum(axis=-1) == mask.sum(axis=-1)
            completion_accuracy = correct_completions.float().mean().item()
            self.all_completion_accuracies.append(completion_accuracy)
        else:
            correct_completions = correct.sum(axis=-1) == mask.sum(axis=-1)
            completion_accuracy = correct_completions.float().mean().item()
            self.all_completion_accuracies.append(completion_accuracy)

        if compute_result:
            # Compute the average metrics across all batches
            avg_per_token_accuracy = np.mean(self.all_per_token_accuracies)
            avg_completion_accuracy = np.mean(self.all_completion_accuracies)
            # Reset the accumulators
            self.all_per_token_accuracies = []
            self.all_completion_accuracies = []

            return {
                "per_token_accuracy": avg_per_token_accuracy,
                "completion_accuracy": avg_completion_accuracy,
            }
        else:
            return {}


class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer
        :param config: configuration of the trainer
        """
        self.config = config

        device_string = PartialState().process_index
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_config.name,
            device_map={"": device_string},
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_config.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reserved_string = "<|reserved_special_token_0|>"
        self.reserved_token = self.tokenizer.encode(
            self.reserved_string, add_special_tokens=False
        )
        self.device_string = device_string

    def tokenize_for_generation(self, example: Dict[str, str]):
        """
        Given an input text, splits it into input and labels and tokenizes them. Note that
        While DataCollatorForCompletionLLM already does this, we need to explicitly tokenize it here
        so that we can left pad the input and right pad the labels for autoregressive generation.
        :param example: a data example, containing the text we want to tokenizer
        """
        tokenized_result = self.tokenizer(example["text"])
        index_t = np.where(
            np.array(tokenized_result["input_ids"]) == self.reserved_token
        )[0].item()
        result_dict = {
            "input_ids": tokenized_result["input_ids"][: index_t + 1],
            "attention_mask": tokenized_result["attention_mask"][: index_t + 1],
            "labels": tokenized_result["input_ids"][(index_t + 1) :],
        }
        return result_dict

    def evaluate(self, datasets: DatasetDict):
        """
        Evaluate the trainer on a set of datasets.
        :param datasets: the datasets we want to evaluate
        """
        self.tokenizer.padding_side = "left"
        val_dataset = datasets["validation"]

        tokenized_dataset = val_dataset.map(self.tokenize_for_generation)
        tokenized_dataset = tokenized_dataset.remove_columns("text")

        data_completion_llm = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors="pt",
        )

        val_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=data_completion_llm,
        )

        peft_model = PeftModel.from_pretrained(
            self.model,
            self.config.eval_checkpoint_dir,
            torch_dtype=torch.bfloat16,
        )
        model = peft_model.merge_and_unload()
        token_accuracy = 0
        action_accuracy = 0
        count = 0
        for batch in tqdm(val_dataloader):
            with torch.inference_mode():
                print("Running inference")
                num_new_tokens = batch["labels"].shape[1]
                output_ids = (
                    model.generate(
                        batch["input_ids"].to(model.device),
                        do_sample=False,
                        max_new_tokens=num_new_tokens,
                        use_cache=True,
                        attention_mask=batch["attention_mask"].to(model.device),
                    )
                    .cpu()
                    .numpy()
                )
                ind_reserved = batch["input_ids"].shape[-1] - 1
                gt_tokens = batch["labels"].cpu().numpy()
                # Labels are left padded, we need to right pad them
                pad_sizes = (gt_tokens == IGNORE_INDEX).sum(-1)
                for ind in range(gt_tokens.shape[0]):
                    pad_size = pad_sizes[ind]
                    if pad_size > 0:
                        # Change padding, content from the right goes to the left side, and we
                        # pad the right side
                        gt_tokens[ind, :(-pad_size)] = gt_tokens[ind, pad_size:]
                        gt_tokens[ind, -pad_size:] = IGNORE_INDEX
                init_t = ind_reserved + 1
                end_t = init_t + gt_tokens.shape[-1]
                pred_tokens = output_ids[:, init_t:end_t]
                mask = gt_tokens != self.tokenizer.pad_token_id
                correct_tokens = ((pred_tokens == gt_tokens) * mask).sum(-1) / mask.sum(
                    -1
                )
                correct_actions = correct_tokens == 1.0
                count += output_ids.shape[0]
                action_accuracy += correct_actions.sum()
                token_accuracy += correct_tokens.sum()
                if self.config.eval.print_output:
                    i = 0
                    pred_str = self.tokenizer.decode(output_ids[i][ind_reserved:])
                    input_str = self.tokenizer.decode(output_ids[i][:ind_reserved])
                    ind_end = pad_sizes[i]
                    gt_str = self.tokenizer.decode(gt_tokens[i][:-ind_end])

                    print(input_str)
                    print("Prediction:", pred_str.split("<end_act>")[0])
                    print("GT:", gt_str.split("<end_act>")[0])
                    print("-----")

            print(f"Action Accuracy: {action_accuracy/count:.2f}")
            print(f"Token Accuracy: {token_accuracy/count:.2f}")

    def train(self, datasets: DatasetDict):
        """
        Train the model
        :param datasets: the datasets we want to train on
        """
        config = self.config
        target_modules = list(config.llm_config.finetune.lora.target_modules)
        lora_config = LoraConfig(
            r=config.llm_config.finetune.lora.rank,
            target_modules=target_modules,
            lora_alpha=config.llm_config.finetune.lora.alpha,
            lora_dropout=config.llm_config.finetune.lora.dropout,
            use_rslora=True,
            bias="none",
        )
        model = get_peft_model(self.model, lora_config)

        if config.training_arguments.instruct:
            collator = DataCollatorForCompletionOnlyLM(
                response_template="<|start_header_id|>assistant<|end_header_id|>",
                tokenizer=self.tokenizer,
            )
        else:
            response_template_ids = self.tokenizer.encode(
                "<|reserved_special_token_0|>",
                add_special_tokens=False,
            )
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids, tokenizer=self.tokenizer
            )

        generation_params = config.llm_config.generation_params
        training_args = SFTConfig(
            dataset_text_field="text",
            output_dir=config.training_arguments.checkpoint_dir,
            run_name=config.wandb.name,
            report_to="wandb",
            save_strategy="steps",
            num_train_epochs=config.training_arguments.epochs,
            max_steps=config.training_arguments.num_steps,
            logging_steps=10,
            save_steps=config.training_arguments.save_steps,
            evaluation_strategy="steps",
            eval_steps=config.training_arguments.eval_steps,
            per_device_train_batch_size=config.training_arguments.batch_size,
            per_device_eval_batch_size=config.training_arguments.batch_size,
            include_inputs_for_metrics=False,
            include_num_input_tokens_seen=True,
            label_names=["labels"],
            batch_eval_metrics=True,
        )
        metrics_accumulator = MetricsAccumulator(
            self.tokenizer, instruct=config.training_arguments.instruct
        )
        trainer = SFTTrainer(
            model,
            train_dataset=datasets["train"],
            eval_dataset={
                "val": datasets["validation"],
                "train_subset": datasets["train_subset"],
            },
            compute_metrics=metrics_accumulator.compute_metrics,
            max_seq_length=generation_params.max_tokens,
            args=training_args,
            data_collator=collator,
        )

        print("Eval")
        trainer.evaluate()
        print(f"Train for {config.training_arguments.epochs} epochs...")
        trainer.train()
        print("End")


@hydra.main(
    version_base=None,
    config_path="../conf/finetuning",
    config_name="finetuning",
)
def main(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.llm_config.name)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = hydra.utils.instantiate(config.dataset)
    datasets = clip_and_filter_datasets(datasets, tokenizer, config)

    print("Dataset Splits")
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)}")

    if not config.evaluate:
        config_dict = {"train_config": OmegaConf.to_container(config)}

    if config.init_distrib:
        init_distrib()

    if not config.evaluate and PartialState().process_index == 0:
        wandb.init(
            name=config.wandb.name, project=config.wandb.project, config=config_dict
        )

    trainer = Trainer(config)

    if config.evaluate:
        trainer.evaluate(datasets)
    else:
        trainer.train(datasets)


if __name__ == "__main__":
    main()
