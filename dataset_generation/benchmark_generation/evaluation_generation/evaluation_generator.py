#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import tqdm

from dataset_generation.benchmark_generation.evaluation_generation.attach_auto_dependencies import (
    infer_and_attach_dependencies,
)
from dataset_generation.benchmark_generation.evaluation_generation.heuristics import (
    filter_generated_ties_heuristic,
    spatial_temporal_correction_heuristic,
)
from dataset_generation.benchmark_generation.evaluation_generation.metadata_mapping import (
    generate_hash_to_text,
    generate_metadata_mappings,
)
from dataset_generation.benchmark_generation.evaluation_generation.parsing import (
    DependencyParser,
    InstructionParser,
    LLMGenerationError,
    PropositionParser,
    SkipParser,
    TemporalParser,
    TerminalSatisfactionParser,
    TieParser,
    metadata_to_state_string,
    split_into_temporal_sub_instructions,
    temporal_words_in_str,
    trim_template_to_fit,
)
from dataset_generation.benchmark_generation.evaluation_generation.utils import (
    extract_template_task_number,
    get_scene_to_within_receps,
    object_initializations_from_name_to_recep,
)
from dataset_generation.benchmark_generation.evaluation_generation.within_set_verification import (
    verify_and_correct_within_set_propositions,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0, CollaborationEpisode
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    EvaluationConstraint,
    EvaluationProposition,
    EvaluationPropositionDependency,
    SameArgConstraint,
    TemporalConstraint,
)


class LLMEvaluationGenerator:
    def __init__(
        self,
        dataset_file_in: str,
        dataset_file_out: str,
        plain_text_eval_dir: str,
        metadata_dir: str,
        log_dir: str,
        scene_info_file: str,
        scene_metadata_dir: str,
        proposition_prompt_file: str,
        temporal_prompt_file: str,
        use_spatial_temporal_correction_heuristic: bool,
        tie_prompt_file: str,
        predicate_vocabulary_file: str,
        llm: Any,
        max_tokens_proposition_call: int,
        max_tokens_dag_call: int,
        max_tokens_tie_call: int,
        skip_temporal_prediction: bool,
        skip_tie_prediction: bool,
        skip_episode_default: bool,
        is_from_templates: bool,
        filter_file_dir: str,
        is_object_state_gen: bool,
        use_resolved_coreferences: bool,
        prompt_template_strs: Optional[Dict[int, str]] = None,
        plain_text_eval_dir_copy: str = "",
    ) -> None:
        if not os.path.exists(dataset_file_in):
            raise IOError(f"Dataset file `{dataset_file_in}` does not exist.")
        os.makedirs(plain_text_eval_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        if plain_text_eval_dir_copy != "":
            os.makedirs(plain_text_eval_dir_copy, exist_ok=True)

        with gzip.open(dataset_file_in, "rt") as f:
            self.dataset_in = json.load(f)

        self.plain_text_eval_dir = plain_text_eval_dir
        self.plain_text_eval_dir_copy = plain_text_eval_dir_copy
        self.metadata_dir = metadata_dir
        self.log_dir = log_dir
        self.dataset_file_out = dataset_file_out
        self.scene_info_file = scene_info_file
        with open(scene_info_file, "r") as f:
            self.scene_info_metadata = json.load(f)
        self.recep_to_description = generate_hash_to_text(
            os.path.join(scene_metadata_dir, "fpmodels-with-decomposed.csv"),
            self.scene_info_metadata["receptacle_to_handle"],
        )
        self.use_spatial_temporal_correction_heuristic = (
            use_spatial_temporal_correction_heuristic
        )
        with open(predicate_vocabulary_file) as f:
            self.predicate_vocabulary: Dict[str, Any] = json.load(f)
        self.affordances = self._load_affordances_dict(
            os.path.join(scene_metadata_dir, "affordance_objects.csv")
        )

        with open(proposition_prompt_file, "r") as f:
            self.proposition_prompt_template = f.read()

        with open(temporal_prompt_file, "r") as f:
            self.dag_prompt_template = f.read()

        with open(tie_prompt_file, "r") as f:
            self.tie_prompt_template = f.read()

        self.llm = llm
        self.max_tokens_proposition_call = max_tokens_proposition_call
        self.max_tokens_dag_call = max_tokens_dag_call
        self.max_tokens_tie_call = max_tokens_tie_call
        self.skip_temporal_prediction = skip_temporal_prediction
        self.skip_tie_prediction = skip_tie_prediction
        self.skip_episode_default = skip_episode_default
        self.is_from_templates = is_from_templates
        self.filter_file_dir = filter_file_dir
        self.is_object_state_gen = is_object_state_gen
        self.use_resolved_coreferences = use_resolved_coreferences
        self.coref_file = os.path.join(
            os.path.dirname(dataset_file_in), "resolved_coref.json"
        )
        self.prompt_template_strs = prompt_template_strs
        if is_from_templates and prompt_template_strs is None:
            raise AssertionError(
                "`prompt_template_strs` must be provided if `is_from_templates==True`."
            )

        predicates: Dict[str, List[Dict[str, Any]]] = self.predicate_vocabulary[
            "predicates"
        ]
        predicates_to_skip = (
            set(self.predicate_vocabulary["object_state_negations"])
            if not is_object_state_gen
            else set()
        )
        self.prop_parser = PropositionParser(
            predicates, self.affordances, predicates_to_skip
        )
        self.dag_parser = TemporalParser(predicates)
        self.tie_parser = TieParser(predicates)
        self.dep_parser = DependencyParser(predicates)

    def generate_plain_text_evaluations(self):
        """Produces evaluation functions for episodes in self.dataset_in. The entities in
        evaluation functions are references via semantic names (table_1) instead of sim
        handles. Metadata is saved that maps from semantic names to handles.

        For each episode, saves:
          - plaintext_evals/episode_{i}.py       * plaintext evaluation data
          - plaintext_evals_orig/episode_{i}.py  * plaintext evaluation data (copy)
          - logs/episode_{i}.log                 * prompts, raw+parsed outputs, failures
          - metadata/episode_{i}.json            * all relevant episode+scene metadata
        """
        if self.is_object_state_gen and self.use_resolved_coreferences:
            if not os.path.exists(self.coref_file):
                raise AssertionError(
                    f"resolved coreferences missing from {self.coref_file}."
                )
            with open(self.coref_file) as f:
                coref_resolved_instructions: Dict[int, str] = {
                    int(k): v for k, v in json.load(f).items()
                }
        else:
            coref_resolved_instructions = None

        for episode in tqdm.tqdm(self.dataset_in["episodes"]):
            eid = self._eid_from_episode(episode)

            prop_prompt_template_ex = ""
            if self.is_from_templates:
                ttn = extract_template_task_number(episode)
                prop_prompt_template_ex = self.prompt_template_strs[ttn]

            eval_function_file = os.path.join(
                self.plain_text_eval_dir, f"episode_{eid}.py"
            )
            if os.path.exists(eval_function_file):
                continue

            metadata = generate_metadata_mappings(
                episode,
                self.scene_info_metadata,
                self.recep_to_description,
            )
            plaintext_eval_str = self.generate_plaintext_evaluation(
                metadata, eid, prop_prompt_template_ex, coref_resolved_instructions
            )

            # save results to files
            with open(eval_function_file, "w") as f:
                f.write(plaintext_eval_str)
            with open(os.path.join(self.metadata_dir, f"episode_{eid}.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            if self.plain_text_eval_dir_copy == "":
                return

            eval_function_file_copy = os.path.join(
                self.plain_text_eval_dir_copy, f"episode_{eid}.py"
            )
            with open(eval_function_file_copy, "w") as f:
                f.write(plaintext_eval_str)

    def compile_plaintext(
        self,
        instruction: str,
        propositions: List[EvaluationProposition],
        tc_constraint: TemporalConstraint,
        tie_constraints: List[Union[SameArgConstraint, DifferentArgConstraint]],
        dependencies: List[EvaluationPropositionDependency],
    ) -> str:
        """
        Produces a string for the plain text file containing all generated evaluation data.
        This string is later saved for manual verification and correction.
        """
        return (
            "# type: ignore\n"
            + InstructionParser.to_plaintext(instruction)
            + self.prop_parser.to_plaintext(propositions)
            + self.dag_parser.to_plaintext(tc_constraint)
            + self.tie_parser.to_plaintext(tie_constraints, propositions)
            + self.dep_parser.to_plaintext(dependencies)
            + TerminalSatisfactionParser.to_plaintext()
            + SkipParser.to_plaintext(self.skip_episode_default)
        )

    def generate_plaintext_evaluation(
        self,
        metadata: Dict,
        eid: int,
        prop_prompt_template_ex: str = "",
        coref_resolved_instructions: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Generates a plaintext evaluation for a single episode consisting of a list
        of propositions, a temporal constraint, and a list of tie constraints. Each
        is generated from a separate LLM call. If an error occurs during generation,
        the plaintext file is produced with empty components.

        prop_prompt_template_ex, if provided, will inject a final example into the
        proposition generation prompt. See the default prompt for the required format.

        coref_resolved_instructions, if provided, maps the episode ID to a resolved
        instruction. The original instruction is replaced with the resolved one.
        """
        state_str = metadata_to_state_string(
            metadata, self.predicate_vocabulary["object_state_negations"]
        )
        inst = metadata["instruction"]

        if self.is_object_state_gen:
            coref_inst = (
                inst
                if coref_resolved_instructions is None
                else coref_resolved_instructions[int(eid)]
            )
            try:
                (
                    propositions,
                    tc_constraint,
                ) = self.generate_os_propositions_and_temporal(
                    eid, coref_inst, state_str, metadata, prop_prompt_template_ex
                )
            except LLMGenerationError as e:
                self._log_results(
                    eid, f"Failure in Proposition Generation. Reason:\n{e}"
                )
                return self.compile_plaintext(inst, [], None, [], [])
        else:
            # generate propositions
            try:
                propositions = self.generate_propositions(
                    eid, inst, state_str, metadata, prop_prompt_template_ex
                )
            except LLMGenerationError as e:
                self._log_results(
                    eid, f"Failure in Proposition Generation. Reason:\n{e}"
                )
                return self.compile_plaintext(inst, [], None, [], [])

            # generate the temporal constraint
            try:
                tc_constraint = self.generate_temporal_constraint(
                    eid, inst, propositions
                )
            except LLMGenerationError as e:
                self._log_results(eid, f"Failure in DAG Generation. Reason:\n{e}")
                return self.compile_plaintext(inst, propositions, None, [], [])

        # generate the tied constraints
        try:
            tie_constraints = self.generate_ties(eid, inst, propositions)
        except LLMGenerationError as e:
            self._log_results(eid, f"Failure in Tie Generation. Reason:\n{e}")
            return self.compile_plaintext(inst, propositions, tc_constraint, [], [])

        self._log_results(eid, "Success.")
        return self.compile_plaintext(
            inst, propositions, tc_constraint, tie_constraints, []
        )

    def generate_os_propositions_and_temporal(
        self,
        eid: int,
        instruction: str,
        state_str: str,
        metadata: Dict,
        prop_prompt_template_ex: str = "",
    ) -> Tuple[List[EvaluationProposition], TemporalConstraint]:
        """
        Eval gen of object state episodes has lower accuracy with the standard
        proposition prediction -> temporal prediction. Here, we first split the
        instruction into temporal sub-instructions. Then we produce propositions
        independently for each sub-instruction.
        """
        temporal_splits = split_into_temporal_sub_instructions(instruction)
        self._log_results(eid, f"[OS prediction] instruction before: \n{instruction}")
        self._log_results(eid, "[OS prediction] instruction after: \n")
        for s in temporal_splits:
            self._log_results(eid, f"\t{s}", end="\n")
        self._log_results(eid, "", end="\n")

        propositions, temporal_groups, prop_idx = [], [], 0
        for instruction_single in temporal_splits:
            new_props = self.generate_propositions(
                eid, instruction_single, state_str, metadata, prop_prompt_template_ex
            )
            propositions.extend(new_props)
            temporal_groups.append([prop_idx + i for i in range(len(new_props))])
            prop_idx += len(new_props)

        tc = self.dag_parser.constraint_from_groups(temporal_groups, len(propositions))
        return propositions, tc

    def generate_propositions(
        self,
        eid: int,
        instruction: str,
        state_str: str,
        metadata: Dict,
        prop_prompt_template_ex: str = "",
    ) -> List[EvaluationProposition]:
        """
        Prompts an LLM to produce evaluation propositions.
        Parses the output into List[EvaluationProposition].
        """
        prompt = self.proposition_prompt_template
        prompt = prompt.replace("{INSTRUCTION}", instruction)
        prompt = prompt.replace("{INIT_STATE}", state_str)
        template_key = "{TEMPLATE_EXAMPLE}"
        if prop_prompt_template_ex == "":
            # trim whitespace to match the other few-shot examples
            template_key += "\n\n"
        else:
            prop_prompt_template_ex = trim_template_to_fit(prop_prompt_template_ex)

        prompt = prompt.replace(template_key, prop_prompt_template_ex)

        self._log_results(eid, f"[prop call] LLM Prompt: \n{prompt}")

        propositions_str = self.llm.generate(
            prompt=prompt,
            stop="[END]",
            max_length=self.max_tokens_proposition_call,
        )
        self._log_results(eid, f"[prop call] Raw LLM Output: \n{propositions_str}")

        propositions = self.prop_parser.from_llm(propositions_str, metadata)
        self._log_results(
            eid,
            f"[prop call] Parsed LLM Output: \n{self.prop_parser.to_plaintext(propositions)}",
        )
        return propositions

    def generate_temporal_constraint(
        self,
        eid: int,
        instruction: str,
        propositions: List[EvaluationProposition],
    ) -> TemporalConstraint:
        """
        Infers the temporal order of a list propositions using LLM.
        Parses this result into DAG proposition groups.
        """
        empty_constraint = TemporalConstraint([], len(propositions))
        if self.skip_temporal_prediction:
            self._log_results(eid, "[dag call] Skipping.")
            return empty_constraint
        if not temporal_words_in_str(instruction):
            self._log_results(
                eid, "[dag call] Skipping: temporal words not in the instruction."
            )
            return empty_constraint

        dag_prompt = self.dag_prompt_template
        dag_prompt = dag_prompt.replace("{INSTRUCTION}", instruction)
        dag_prompt = dag_prompt.replace(
            "{PROPOSITIONS}", self.prop_parser.to_plaintext(propositions)
        )
        self._log_results(eid, f"[dag call] LLM Prompt: \n{dag_prompt}")

        dag_str = self.llm.generate(
            prompt=dag_prompt, stop="\n\n", max_length=self.max_tokens_dag_call
        )
        self._log_results(eid, f"[dag call] Raw LLM Output: \n{dag_str}")

        tc_constraint = self.dag_parser.from_llm(dag_str, n_props=len(propositions))
        self._log_results(
            eid,
            f"[dag call] Parsed LLM Output: \n{self.dag_parser.to_plaintext(tc_constraint)}",
        )
        if self.use_spatial_temporal_correction_heuristic:
            try:
                tc_constraint = spatial_temporal_correction_heuristic(
                    tc_constraint, propositions
                )
            except KeyError as e:
                raise LLMGenerationError(f"Failure in temporal heuristic. Reason:\n{e}")

            self._log_results(
                eid,
                f"[dag call] Parsed LLM Output: \n{self.dag_parser.to_plaintext(tc_constraint)}",
            )
        return tc_constraint

    def generate_ties(
        self,
        eid: int,
        instruction: str,
        propositions: List[EvaluationProposition],
    ) -> List[Union[SameArgConstraint, DifferentArgConstraint]]:
        if self.skip_tie_prediction:
            self._log_results(eid, "[tie call] Skipping.")
            return []

        tie_prompt = self.tie_prompt_template
        tie_prompt = tie_prompt.replace("{INSTRUCTION}", instruction)
        tie_prompt = tie_prompt.replace(
            "{PROPOSITIONS}", self.prop_parser.to_plaintext(propositions)
        )
        self._log_results(eid, f"[tie call] LLM Prompt: \n{tie_prompt}")

        tie_str = self.llm.generate(
            prompt=tie_prompt, stop="\n\n", max_length=self.max_tokens_dag_call
        )
        self._log_results(eid, f"[tie call] Raw LLM Output: \n{tie_str}")

        try:
            constraints = self.tie_parser.from_llm(tie_str, propositions)
        except Exception as e:
            # don't crash, this call isn't critical
            self._log_results(eid, f"[tie call] Unexpected error: {str(e)}")
            constraints = []

        self._log_results(
            eid,
            f"[tie call] Parsed LLM Output: \n{self.tie_parser.to_plaintext(constraints, propositions)}",
        )

        filtered_constraints = filter_generated_ties_heuristic(
            eid, constraints, propositions, self._log_results
        )
        n_before = len(constraints)
        n_after = len(filtered_constraints)
        if n_before != n_after:
            self._log_results(
                eid,
                f"[tie call] Filtered ties from {n_before} to {n_after} ties.",
            )
        return filtered_constraints

    def parse_plaintext_eval(
        self, plaintext_str: str, metadata: Dict
    ) -> Tuple[
        str,
        List[EvaluationProposition],
        List[EvaluationConstraint],
        List[EvaluationPropositionDependency],
    ]:
        """
        Takes a string of the plain text eval for an episode and converts it to
        propositions and constraints.
        """

        # check if the annotator marked skip
        skipped, reason = SkipParser.from_plaintext(plaintext_str)
        if skipped:
            raise LLMGenerationError("Episode skipped. " + reason)

        instruction = InstructionParser.from_plaintext(plaintext_str)
        propositions = self.prop_parser.from_plaintext(plaintext_str, metadata)
        tc_constraint = self.dag_parser.from_plaintext(plaintext_str, len(propositions))
        tie_constraints = self.tie_parser.from_plaintext(plaintext_str, propositions)
        ts_constraint = TerminalSatisfactionParser.from_plaintext(
            plaintext_str, propositions
        )
        all_constraints: List[EvaluationConstraint] = [
            tc_constraint,
            *tie_constraints,
            ts_constraint,
        ]
        dependencies = self.dep_parser.from_plaintext(plaintext_str)
        return instruction, propositions, all_constraints, dependencies

    def plaintext_evals_to_dataset(
        self, eids_to_skip: Optional[Set[int]] = None
    ) -> None:
        """
        Converts semantic names in evaluation functions to handles, resulting in
        evaluation functions that can be loaded into Hab-LLM.
        """
        if eids_to_skip is None:
            eids_to_skip = set()

        def eid_from_fname(fname):
            return int(fname.split(".")[0].split("_")[-1])

        # regenerate metadata in case handles changed.
        new_metadata = {}  # eid to metadata
        for ep in self.dataset_in["episodes"]:
            eid = self._eid_from_episode(ep)
            new_metadata[eid] = generate_metadata_mappings(
                ep,
                self.scene_info_metadata,
                self.recep_to_description,
            )

        new_data = {}  # maps eid to dict containing propositions, constraints
        failure_modes = defaultdict(list)
        for fname in sorted(
            os.listdir(self.plain_text_eval_dir), key=lambda s: eid_from_fname(s)
        ):
            eid = eid_from_fname(fname)
            if eid in eids_to_skip:
                continue

            with open(os.path.join(self.plain_text_eval_dir, fname)) as f:
                eval_fn_plain_text_str = f.read()
            try:
                metadata = new_metadata[eid]
                (
                    instruction,
                    propositions,
                    constraints,
                    dependencies,
                ) = self.parse_plaintext_eval(eval_fn_plain_text_str, metadata)
            except LLMGenerationError as e:
                print(f"Failed to pack EID {eid} of file {fname}.\n\tReason: {str(e)}")
                failure_modes[str(e)].append(eid)
                continue
            except KeyError:
                print(f"eid {eid} no longer in the orig dataset?")
                failure_modes["missing eid"].append(eid)
                continue

            new_data[eid] = {
                "instruction": instruction,
                "propositions": propositions,
                "constraints": constraints,
                "dependencies": dependencies,
            }

        # log packing failures for future analysis and summary stats
        failure_summary = {
            "n_episodes_packed": len(new_data),
            "eids_packed": list(new_data.keys()),
            "failure_modes": failure_modes,
        }
        failure_file = os.path.join(
            os.path.dirname(self.dataset_file_out), "packing_failures.json"
        )
        with open(failure_file, "w") as f:
            json.dump(failure_summary, f, indent=2)

        self.save_new_dataset(new_data)

    def save_new_dataset(self, new_data: Dict[int, Any]) -> None:
        """Compile a CollaborationDataset and save it to disk."""
        if not self.dataset_file_out.endswith(".json.gz"):
            raise AssertionError(
                f"Dataset file out should end with .json.gz. Found: `{self.dataset_file_out}`"
            )

        scene_to_within_receps = get_scene_to_within_receps(self.filter_file_dir)

        dataset = CollaborationDatasetV0()
        for ep in self.dataset_in["episodes"]:
            eid = self._eid_from_episode(ep)

            # skip if we don't have evaluation data for this episode
            if eid not in new_data:
                continue

            instruction: str = new_data[eid]["instruction"]

            episode = CollaborationEpisode(  # type: ignore
                episode_id=ep["info"]["extra_info"]["episode_id"],
                scene_id=ep["scene_id"],
                scene_dataset_config=ep["scene_dataset_config"],
                additional_obj_config_paths=ep["additional_obj_config_paths"],
                start_position=ep["start_position"],
                start_rotation=ep["start_rotation"],
                ao_states=ep["ao_states"],
                rigid_objs=ep["rigid_objs"],
                targets=ep["targets"],
                markers=ep["markers"],
                name_to_receptacle=ep["name_to_receptacle"],
                instruction=instruction,
                info={
                    "object_labels": ep["info"]["object_labels"],
                    "initial_state": ep["info"]["extra_info"]["initial_state"],
                    "object_initializations": object_initializations_from_name_to_recep(
                        ep["name_to_receptacle"], ep["scene_id"], scene_to_within_receps
                    ),
                },
                evaluation_propositions=new_data[eid]["propositions"],
                evaluation_proposition_dependencies=new_data[eid]["dependencies"],
                evaluation_constraints=new_data[eid]["constraints"],
                object_states=ep["object_states"] if "object_states" in ep else {},
            )
            dataset.episodes.append(episode)

        dataset = infer_and_attach_dependencies(dataset, override_existing=False)
        dataset = verify_and_correct_within_set_propositions(
            dataset, self.filter_file_dir
        )

        with gzip.open(self.dataset_file_out, "wt") as f:
            s = dataset.to_json()
            f.write(s)
        print(
            f"packed {len(dataset.episodes)} episodes"
            f" out of {len(self.dataset_in['episodes'])}."
        )

    def _log_results(self, eid: int, msg: str, end: str = "\n\n") -> None:
        """Appends the log message to the episode log."""
        with open(os.path.join(self.log_dir, f"episode_{eid}.log"), "a") as f:
            f.write(msg + end)

    def _load_affordances_dict(self, affordances_csv: str) -> Dict[str, Set[str]]:
        """Maps object state predicates to a set of object/furniture classes that have an associated affordance"""
        with open(affordances_csv, "r") as f:
            reader = csv.reader(f)

            k_map = self.predicate_vocabulary["affordance_to_predicates"]
            affordances: Dict[str, Set[str]] = {
                p: set() for p in sorted({p for ps in k_map.values() for p in ps})
            }
            for row in reader:
                try:
                    k_map[row[0]]
                except KeyError as e:
                    print(
                        f"Affordance key `{row[0]}` missing from key map {k_map.keys()}."
                    )
                    raise e
                for predicate in k_map[row[0]]:
                    for aff in row[2:]:
                        aff = aff.lstrip().removeprefix("'").removeprefix("['")
                        aff = aff.removesuffix("'").removesuffix("']")
                        affordances[predicate].add(aff)
        return affordances

    @staticmethod
    def _eid_from_episode(episode: CollaborationEpisode) -> int:
        """Extracts the episode ID integer from an episode object."""
        eid = episode["info"]["extra_info"]["episode_id"]
        if isinstance(eid, int):
            return eid
        if eid.isdigit():
            return int(eid)
        return int(eid.split("|")[-1].split(".")[0])
