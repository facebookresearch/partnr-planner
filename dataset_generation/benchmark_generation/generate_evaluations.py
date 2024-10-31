#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import habitat
import hydra
import omegaconf
import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataset_generation.benchmark_generation.evaluation_generation.evaluation_generator import (
    LLMEvaluationGenerator,
)
from dataset_generation.benchmark_generation.evaluation_generation.parsing import (
    metadata_to_state_string,
    proposition_to_llm_output_str,
)
from dataset_generation.benchmark_generation.evaluation_generation.utils import (
    display_packing_stats,
    scene_dirs_from_dir,
    set_config_scene_ids,
)
from dataset_generation.benchmark_generation.verify_dataset import (
    verify_dataset_parallel,
)


def setup_config(
    config: omegaconf.DictConfig,
) -> Tuple[omegaconf.DictConfig, List[str], Dict[int, str]]:
    """Initializes the job config."""
    config = set_config_scene_ids(config)
    config = config.eval_gen

    prompt_template_strs = (
        load_template_prompt_examples(config) if config.generate else {}
    )
    if config.generate and config.llm.inference_mode == "rlm":
        # setup the RLM connection info
        running_servers = os.listdir(config.llm.rlm_path)
        try:
            server = running_servers[config.llm.rlm_index]
        except IndexError as e:
            print(
                f"Error: expected running server in {config.llm.rlm_path}"
                f" at RLM index {config.llm.rlm_index}."
            )
            raise e

        with habitat.config.read_write(config):
            config.llm.host = server.split(":")[0]
            config.llm.port = int(server.split(":")[1])

    scene_ids_to_run = (
        config.scene_ids
        if config.scene_index == -1
        else [config.scene_ids[config.scene_index]]
    )
    if len(scene_ids_to_run) == 0:
        raise AssertionError("No scenes selected.")

    if config.is_object_state_gen:
        with habitat.config.read_write(config):
            config.proposition_prompt_file = config.os_proposition_prompt_file

    print("--- Config --- ")
    print(OmegaConf.to_yaml(config))
    print("selected scenes:", scene_ids_to_run)
    return config, scene_ids_to_run, prompt_template_strs


def run_single_scene(
    scene_id: str, config: omegaconf.DictConfig, prompt_template_strs: Dict[int, str]
) -> None:
    outputs_dir = os.path.join(config.output_path, str(config.run_name), scene_id)
    eval_generator = LLMEvaluationGenerator(
        dataset_file_in=os.path.join(
            config.path_to_dataset_in, scene_id, "dataset.json.gz"
        ),
        dataset_file_out=os.path.join(outputs_dir, f"{config.run_name}.json.gz"),
        plain_text_eval_dir=os.path.join(outputs_dir, "plaintext_evals"),
        plain_text_eval_dir_copy=os.path.join(outputs_dir, "plaintext_evals_orig"),
        metadata_dir=os.path.join(outputs_dir, "metadata"),
        log_dir=os.path.join(outputs_dir, "logs"),
        scene_info_file=os.path.join(
            config.path_to_dataset_in, scene_id, "scene_info.json"
        ),
        scene_metadata_dir=config.scene_metadata_dir,
        proposition_prompt_file=config.proposition_prompt_file,
        temporal_prompt_file=config.temporal_prompt_file,
        use_spatial_temporal_correction_heuristic=config.use_spatial_temporal_correction_heuristic,
        tie_prompt_file=config.tie_prompt_file,
        predicate_vocabulary_file=config.predicate_vocabulary_file,
        llm=instantiate(config.llm.llm)(conf=config.llm),
        max_tokens_proposition_call=config.llm.max_tokens_proposition_call,
        max_tokens_dag_call=config.llm.max_tokens_dag_call,
        max_tokens_tie_call=config.llm.max_tokens_tie_call,
        skip_temporal_prediction=config.skip_temporal_prediction,
        skip_tie_prediction=config.skip_tie_prediction,
        skip_episode_default=config.skip_episode_default,
        is_from_templates=config.is_from_templates,
        filter_file_dir=config.filter_file_dir,
        is_object_state_gen=config.is_object_state_gen,
        use_resolved_coreferences=config.use_resolved_coreferences,
        prompt_template_strs=prompt_template_strs,
    )

    if config.generate:
        eval_generator.generate_plain_text_evaluations()
    if config.pack:
        eval_generator.plaintext_evals_to_dataset()


def merge_datasets(source_path: str) -> None:
    """
    Merge scene-specific datasets into a single dataset and scene-specific
    metadata into a single metadata folder. Assigns new episode IDs.
    Args:
        - source_path: path to the directory containing scene directories of generated and packed episodes.
    Produces:
        - [source_path]/[run_name].json.gz
        - [source_path]/metadata/[episode_id].json
    """
    run_name = source_path.split("/")[-1]

    datasets_to_merge = []
    for scene_dir in scene_dirs_from_dir(source_path):
        dset_path = os.path.join(scene_dir, f"{run_name}.json.gz")
        metadata_path = os.path.join(scene_dir, "metadata/episode_{eid}.json")
        if not os.path.exists(dset_path):
            continue
        assert os.path.exists(
            os.path.dirname(metadata_path)
        ), f"{dset_path} missing {metadata_path}"
        datasets_to_merge.append((dset_path, metadata_path))

    assert len(datasets_to_merge), "no datasets to merge."

    new_metadata_dir = os.path.join(source_path, "metadata")
    os.makedirs(new_metadata_dir, exist_ok=True)

    new_episodes = []
    new_eid = 0
    for dset, metadata_template in tqdm.tqdm(datasets_to_merge):
        with gzip.open(dset, "rt") as f:
            eps = json.load(f)["episodes"]
        new_episodes.extend(eps)
        for ep in eps:
            # copy the episode metadata to src_path/metadata/episode_{eid}.json
            eid = str(ep["episode_id"])
            if not eid.isdigit():
                eid = eid.split("|")[-1]
            metadata_f = metadata_template.format(eid=eid)

            ep["info"]["episode_id"] = ep["episode_id"]
            ep["episode_id"] = str(new_eid)
            shutil.copy(
                metadata_f, os.path.join(new_metadata_dir, f"episode_{new_eid}.json")
            )
            new_eid += 1

    display_packing_stats(source_path, len(new_episodes))

    with gzip.open(os.path.join(source_path, f"{run_name}.json.gz"), "wt") as f:
        s = json.dumps({"config": None, "episodes": new_episodes})
        f.write(s)


def verify_inference_in_sim(
    output_path: str,
    run_name: str,
    scene_index: Optional[int] = -1,
    scene_ids: Optional[List[str]] = None,
    verification_num_proc: Optional[int] = 1,
) -> None:
    """
    Verifies that each episode can load and evaluation doesn't crash.
    Which dataset is verified? One of:
        (1) a scene dataset (if scene_index is provided)
        (2) the merged dataset (if it exists)
        (3) all scene dataset files
    """
    merged_dataset_path = os.path.join(output_path, run_name, f"{run_name}.json.gz")
    results_dir = os.path.join(output_path, run_name, "verification")

    to_verify: List[Tuple[str, str]] = []  # (dataset_path, results_path)
    if scene_index == -1:
        if os.path.exists(merged_dataset_path):
            to_verify = [(merged_dataset_path, results_dir)]
        else:
            for scene in scene_ids:
                dset = os.path.join(output_path, run_name, scene, f"{run_name}.json.gz")
                to_verify.append((dset, os.path.join(results_dir, scene)))
    else:
        dset = os.path.join(
            output_path, run_name, scene_ids[scene_index], f"{run_name}.json.gz"
        )
        to_verify = [(dset, results_dir)]

    for dataset_path, results_path in to_verify:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        verify_dataset_parallel(dataset_path, results_path, verification_num_proc)


def trim_failed_episodes(output_path: str, run_name: str) -> None:
    """
    Trims the merged dataset to episodes which passed simulator verification.
    Saves these episodes to a new file `[run_name]_verified.json.gz`.
    """
    summary_file = os.path.join(output_path, run_name, "verification", "summary.json")
    if not os.path.exists(summary_file):
        raise AssertionError(
            f"Summary file not found: {summary_file}. Run `verify` first."
        )

    with open(summary_file) as f:
        summary = json.load(f)

    dataset_file = os.path.join(output_path, run_name, f"{run_name}.json.gz")
    with gzip.open(dataset_file, "rt") as f:
        dataset = json.load(f)
    eps_before = len(dataset["episodes"])

    failed_eids = {
        int(k.removesuffix(".json"))
        for k, v in summary.items()
        if not v["success_init"]
    }
    dataset["episodes"] = list(
        filter(lambda e: int(e["episode_id"]) not in failed_eids, dataset["episodes"])
    )
    eps_after = len(dataset["episodes"])

    dataset_file_out = os.path.join(
        output_path, run_name, f"{run_name}_verified.json.gz"
    )
    with gzip.open(dataset_file_out, "wt") as f:
        s = json.dumps(dataset)
        f.write(s)

    perc = round(100 * eps_after / eps_before, 2)
    print()
    print(f"Episodes in {run_name} (before): {eps_before}")
    print(f"Episodes in {run_name}  (after): {eps_after} ({perc}%)")
    print(f"Saved trimmed dataset to: {dataset_file_out}")
    print()


def load_template_prompt_examples(config: omegaconf.DictConfig) -> Dict[int, str]:
    """
    Produce a mapping of template task number to string of the proposition prompt example.
    """
    if not config.is_from_templates:
        return {}

    with gzip.open(config.template_dataset, "rt") as f:
        template_episodes = json.load(f)["episodes"]

    with open(config.predicate_vocabulary_file) as f:
        object_state_negations = json.load(f)["object_state_negations"]

    prompt_example_strs = {}  # episode index (NOT eid) to prompt example string

    for i, ep in enumerate(template_episodes):
        eid = str(ep["info"]["episode_id"])
        if not eid.isdigit():
            eid = eid.split("|")[-1]

        metadata_f = os.path.join(
            config.template_dataset_dir,
            ep["scene_id"],
            "metadata",
            f"episode_{eid}.json",
        )
        with open(metadata_f) as f:
            metadata = json.load(f)

        metadata_str = metadata_to_state_string(metadata, object_state_negations)
        propositions_str = "".join(
            proposition_to_llm_output_str(p, metadata)
            for p in ep["evaluation_propositions"]
        ).lstrip("\n")
        ex_str = f"""<step> Source: user

The initial state is:
{metadata_str}

Instruction: "{ep["instruction"]}"

<step> Source: assistant
{propositions_str}
[END]
"""
        prompt_example_strs[i] = ex_str

    return prompt_example_strs


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="evaluation_gen.yaml",
)
def main(config: omegaconf.DictConfig) -> None:
    config, scene_ids_to_run, prompt_template_strs = setup_config(config)

    if config.generate or config.pack:
        for scene_id in scene_ids_to_run:
            run_single_scene(str(scene_id), config, prompt_template_strs)
    if config.merge:
        merge_datasets(os.path.join(config.output_path, config.run_name))
    if config.verify:
        verify_inference_in_sim(
            config.output_path,
            config.run_name,
            config.scene_index,
            config.scene_ids,
            config.verification_num_proc,
        )
        trim_failed_episodes(
            config.output_path,
            config.run_name,
        )


if __name__ == "__main__":
    main()
