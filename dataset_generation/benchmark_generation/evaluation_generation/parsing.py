#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import itertools
import json
import re
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import nltk

from dataset_generation.benchmark_generation.evaluation_generation.utils import (
    self_next_to_self_in_proposition,
)
from habitat_llm.agent.env.evaluation import evaluation_functions
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    EvaluationProposition,
    EvaluationPropositionDependency,
    SameArgConstraint,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)

TEMPORAL_WORDS = {
    "after",
    "afterward",
    "afterwards",
    "before",
    "finally",
    "finish",
    "following",
    "last",
    "lastly",
    "once",
    "preceding",
    "second",
    "secondly",
    "succeeding",
    "then",
    "upon",
    "now",
}
TEMPORAL_WORDS_COMMA = {"next,"}


class LLMGenerationError(Exception):
    pass


class EvalGenParser(ABC):
    """
    Defines the abstract methods used during eval generation for a single component
    call to an LLM. Examples include generating propositions, a temporal constraint,
    or tie constraints.
    """

    def __init__(self, predicate_vocab) -> None:
        self.predicate_vocab = predicate_vocab
        super().__init__()

    @abstractmethod
    def from_llm(self, *args, **kwargs) -> Any:
        """Parses the LLM output to an evaluation object (propositions, constraints, etc)"""

    @abstractmethod
    def to_plaintext(self, *args, **kwargs) -> str:
        """Saves the evaluation object as plaintext for verification and correction"""

    @abstractmethod
    def from_plaintext(self, *args, **kwargs) -> Any:
        """Parses the plaintext representation to an evaluation object"""


class PropositionParser(EvalGenParser):
    def __init__(
        self,
        predicate_vocab: Dict[str, List[Dict[str, Any]]],
        affordances: Dict[str, Set[str]],
        predicates_to_skip: Optional[Set[str]] = None,
    ) -> None:
        self.affordances = affordances
        self.predicates_to_skip: Set[str] = (
            set() if predicates_to_skip is None else predicates_to_skip
        )
        super().__init__(predicate_vocab)

    def from_llm(
        self, llm_str_output: str, metadata: dict
    ) -> List[EvaluationProposition]:
        """
        Parse the raw LLM output string into a list of eval propositions.
        The LLM output string takes the form:
            [FN] is_on_top([...], [...]) [/FN]
            [FN] is_inside([...], [...]) [/FN]
            ...
        """
        llm_str_output = "[FN] " + llm_str_output

        # find the propositions wrapped by [FN] tags.
        pattern = r"\[FN\](.*?)\[/FN\]"
        propositions_str_list = [m.strip() for m in re.findall(pattern, llm_str_output)]
        if len(propositions_str_list) == 0:
            raise LLMGenerationError("The LLM produced no parsable propositions.")

        propositions = []
        for l in propositions_str_list:
            try:
                tree = ast.parse(l).body[0]
            except Exception as e:
                raise LLMGenerationError(
                    f"Invalid proposition string: `{l}`. \nError: {str(e)}."
                )

            args = {}
            fn_name = tree.value.func.id  # type: ignore
            if fn_name == "is_close_to":
                fn_name = "is_next_to"
            if fn_name not in self.predicate_vocab:
                raise LLMGenerationError(
                    f"Predicate name `{fn_name}` not in predicate vocab."
                )
            if fn_name in self.predicates_to_skip:
                continue

            for k, v in zip(self.predicate_vocab[fn_name], tree.value.args):  # type: ignore
                try:
                    args[k["name"]] = self._parse_single_arg_llm(
                        arg_str=v, arg_name=k["name"], metadata=metadata
                    )
                except Exception as e:
                    raise LLMGenerationError(
                        f"Error parsing arg `{v}`. Error: {str(e)}"
                    )

            prop = EvaluationProposition(function_name=fn_name, args=args)

            if self_next_to_self_in_proposition(prop):
                raise LLMGenerationError("An object cannot be next to itself.")

            propositions.append(prop)

        if len(propositions) == 0:
            raise LLMGenerationError("Empty proposiitons list.")

        return propositions

    def _parse_single_arg_llm(self, arg_str: str, arg_name: str, metadata: Dict) -> Any:
        arg = ast.literal_eval(arg_str)
        if arg_name in ["receptacle_handles", "room_ids"]:
            if isinstance(arg, list):
                arg = arg[0]
            if not isinstance(arg, str):
                raise LLMGenerationError(
                    f"Expected arg for {arg_name} to be a string. Got: {arg}"
                )
            if arg_name == "receptacle_handles":
                parsed_arg = self._convert_to_furniture_instances(arg, metadata)
            else:
                parsed_arg = self._convert_to_room_instances(arg, metadata)
        elif arg_name == "object_handles" or "entity_handles" in arg_name:
            # usually objects, but allow furniture.
            if isinstance(arg, list):
                if not len(arg) or not isinstance(arg[0], str):
                    raise LLMGenerationError(
                        f"Expected arg for entity_handles to be a str or list[str]. Got: {arg}"
                    )
                arg = " or ".join(arg)
            if not isinstance(arg, str):
                raise LLMGenerationError(
                    f"Expected arg for entity_handles to be a str. Got: {arg}"
                )
            # now we have a string of ORs.
            parsed_arg = []
            for entity in arg.split(" or "):
                entity = entity.strip()
                if entity.split("_")[-1].isdigit():
                    parsed_arg.append(entity)
                else:
                    parsed_arg.extend(
                        self._convert_to_furniture_instances(entity, metadata)
                    )
        else:
            parsed_arg = arg
        return parsed_arg

    def to_plaintext(self, propositions: List[EvaluationProposition]) -> str:
        """Generate a string representation of the eval propositions for annotation."""
        s = """
# ----------------------------------------
# PROPOSITIONS
#    is_on_top(objects, receptacles, number=1, is_same_receptacle=False)
#    is_inside(objects, receptacles, number=1, is_same_receptacle=False)
#    is_in_room(objects, rooms, number=1, is_same_room=False)
#    is_next_to(entities_a, entities_b, number=1, is_same_b=False, l2_threshold=0.5)
#    is_on_floor(objects, number=1)
#    object states: args=(objects, number=1)
#      is_clean, is_dirty. default: dirty
#      is_filled, is_empty. default: empty
#      is_powered_on, is_powered_off. default: off
#    Args:
#        objects/receptacles/entities_*: OR of a list
#        number: n objects/entities_a must satisfy
#        is_same_*: the same receptacle/entities_b must satisfy all n objects/entities_a
# ----------------------------------------
propositions = [
"""

        for prop in propositions:
            s += "    " + self.to_plaintext_single(prop) + ",\n"
        s += "]\n"
        return s

    def from_plaintext(
        self, plaintext_str: str, metadata: Dict
    ) -> List[EvaluationProposition]:
        """Loads the proposition list from the plaintext evaluation str (possibly after
        manual human annotation). Maps semantic names (e.g. table_1) to handles and
        produces a list of evaluation propositions that can be loaded into Hab-LLM.
        """

        propositions_str_list = extract_lines_between(
            plaintext_str, "propositions = [", "]"
        )

        propositions: List[EvaluationProposition] = []
        for proposition_str in propositions_str_list:
            if proposition_str == "":
                continue
            try:
                tree = ast.parse(proposition_str).body[0]
                fn_name = tree.value.func.id  # type: ignore
            except Exception as e:
                raise LLMGenerationError(
                    f"Invalid Python code. Proposition string: `{proposition_str}`."
                    f"\nError: {str(e)}."
                )

            if fn_name not in self.predicate_vocab:
                raise LLMGenerationError(f"`{fn_name}` is not a valid predicate.")

            # initialize defaults
            args = {}
            for arg_template in self.predicate_vocab[fn_name]:
                args[arg_template["name"]] = arg_template["default"]

            # insert actual args

            # variable number of args in *args
            if self.predicate_vocab[fn_name][0]["name"] == "*args":
                entity_type = self.predicate_vocab[fn_name][0]["entity_type"]
                for arg in tree.value.args:  # type: ignore
                    try:
                        arg_template = self.predicate_vocab[fn_name][0]
                    except IndexError as e:
                        raise LLMGenerationError(
                            f"Invalid *args of predicate `{fn_name}`. Arg value: `{arg.value}`."
                            f" Error: {str(e)}"
                        )
                    args["*args"].append(
                        self._parse_single_arg(
                            "*args", arg, entity_type, fn_name, metadata
                        )
                    )
            else:
                for i, arg in enumerate(tree.value.args):  # type: ignore
                    try:
                        arg_template = self.predicate_vocab[fn_name][i]
                    except IndexError as e:
                        raise LLMGenerationError(
                            f"Invalid arg index `{i}` of predicate `{fn_name}`. Arg value: `{arg.value}`."
                            f" Error: {str(e)}"
                        )

                    arg_name = arg_template["name"]
                    args[arg_name] = self._parse_single_arg(
                        arg_name, arg, arg_template["entity_type"], fn_name, metadata
                    )

            # insert actual kwargs
            for keyword in tree.value.keywords:  # type: ignore
                arg_entity_type = None
                for arg_template in self.predicate_vocab[fn_name]:
                    if keyword.arg == arg_template["name"]:
                        arg_entity_type = arg_template["entity_type"]
                        break
                else:
                    raise LLMGenerationError(
                        f"Keyword `{keyword.arg}` not in predicate `{fn_name}`."
                    )
                try:
                    args[keyword.arg] = self._parse_single_arg(
                        keyword.arg, keyword.value, arg_entity_type, fn_name, metadata
                    )
                except Exception as e:
                    raise LLMGenerationError(
                        f"The kwarg `{keyword.arg}={arg}` cannot evaluate in Python."
                        f" Error: {str(e)}"
                    )
            propositions.append(EvaluationProposition(function_name=fn_name, args=args))

        return propositions

    def to_plaintext_single(self, prop: EvaluationProposition) -> str:
        s = f"{prop.function_name}("
        for arg_name, arg_value in prop.args.items():
            if self._is_kwarg(prop.function_name, arg_name):
                s += f"{arg_name}={arg_value}, "
            else:
                s += f"{arg_value}, "
        s = s.removesuffix(", ")
        s += ")"
        return s

    def _strip_id(self, s: str) -> str:
        """removes the suffix _[int]"""
        return "_".join(s.split("_")[:-1])

    def _convert_to_room_instances(self, s: str, metadata: Dict) -> List[str]:
        """Convert a room string reference to a list of room instances."""
        return [r for r in metadata["rooms"] if s in self._strip_id(r)]

    def _convert_to_furniture_instances(self, s: str, metadata: Dict) -> List[str]:
        """Convert a furniture string reference to a list of furniture instances."""
        x = s.split(" in ")
        furniture_cat = x[0]
        rooms = metadata["rooms"]

        if len(x) > 1:
            # limits rooms to the specified category. Allow all rooms if no match.
            matching_rooms = self._convert_to_room_instances(x[1], metadata)
            if len(matching_rooms):
                rooms = matching_rooms

        # get all furn IDs of furn cat
        furniture_of_cat = [
            f
            for f in metadata["recep_to_description"]
            if self._strip_id(f) == furniture_cat
        ]

        # keep only those that belong to a valid room
        keep = [f for f in furniture_of_cat if metadata["recep_to_room"][f] in rooms]

        # if there are none, keep them all.
        return keep if len(keep) else furniture_of_cat

    def _is_kwarg(self, fn_name: str, arg_name: str) -> bool:
        """Returns True if arg_name is a keyword argument of function fn_name"""
        if fn_name not in self.predicate_vocab:
            raise LLMGenerationError(
                f"Predicate name `{fn_name}` not in predicate vocab."
            )

        for arg in self.predicate_vocab[fn_name]:
            if arg["name"] == arg_name:
                return arg["is_kwarg"]

        raise LLMGenerationError(
            f"Arg `{arg_name}` not found in predicate vocabulary under `{fn_name}`"
        )

    def _parse_single_arg(self, arg_name, arg, entity_type, fn_name, metadata):
        def _parse_arg_literal_list(arg_literal):
            arg_literal_verified = []
            for e in arg_literal:
                self._assert_valid_affordances(fn_name, arg_name, e)
                arg_literal_verified.append(
                    self.map_to_handle(e, entity_type, metadata)
                )
            return arg_literal_verified

        arg_literal = ast.literal_eval(arg)
        if entity_type is not None:
            if isinstance(arg_literal, list):
                if len(arg_literal) == 0:
                    raise LLMGenerationError("Argument list is empty.")
                if isinstance(arg_literal[0], list):
                    arg_literal = [_parse_arg_literal_list(x) for x in arg_literal]
                else:
                    arg_literal = _parse_arg_literal_list(arg_literal)
            else:
                self._assert_valid_affordances(fn_name, arg_name, arg_literal)
                arg_literal = self.map_to_handle(arg_literal, entity_type, metadata)
        return arg_literal

    def _assert_valid_affordances(
        self, fn_name: str, arg_name: str, entity_str: str
    ) -> None:
        """entity_str is an object/furniture class or instance."""
        if fn_name not in self.affordances or arg_name != "object_handles":
            return

        if entity_str.split("_")[-1].isdigit():
            entity_str = "_".join(entity_str.split("_")[:-1])
        if entity_str in self.affordances[fn_name]:
            return
        raise LLMGenerationError(
            f"Predicate `{fn_name}` is not afforded to {entity_str}"
        )

    @staticmethod
    def map_to_handle(name: str, entity_type: str, metadata: Dict[str, Any]) -> str:
        """Maps the name of an entity to its sim handle, guided by entity_type."""
        map_key = {
            "object": "object_to_handle",
            "receptacle": "recep_to_handle",
            "room": "room_to_id",
        }[entity_type]

        valid_entities = metadata[map_key]

        # allow objects to act as receptacles
        if entity_type == "receptacle":
            valid_entities = {**metadata["object_to_handle"], **valid_entities}

        if name not in valid_entities:
            raise LLMGenerationError(f"`{name}` is not a valid {entity_type}.")
        return valid_entities[name]


class TemporalParser(EvalGenParser):
    def from_llm(self, llm_str_output: str, n_props: int) -> TemporalConstraint:
        """Parses the LLM output to an evaluation object (propositions or constraints)"""
        if llm_str_output == "":
            raise LLMGenerationError("[dag call] The LLM produced an empty output.")

        # remove the trailing text
        lines = llm_str_output.split("\n")
        end_idx = -1
        for i, l in enumerate(lines):
            if l == "]":
                end_idx = i
        if end_idx == -1:
            tc_groups_str = llm_str_output.split("\n\n")[0]
        else:
            tc_groups_str = "\n".join(lines[: end_idx + 1])

        # start the str with the right list opener
        if not tc_groups_str.startswith("[["):
            tc_groups_str = "[" + tc_groups_str

        # load the List[List[int]] from str
        tc_groups_str = tc_groups_str.replace("],\n]", "]]")
        tc_groups_str = tc_groups_str.replace("]\n ", "],\n ")
        if not tc_groups_str.endswith("]"):
            tc_groups_str += "]"
        try:
            tc_groups_str = (
                tc_groups_str.split("\n]")[0].replace("\n", "").replace(" ", "") + "]"
            )
            groups = json.loads(tc_groups_str)
        except Exception as e:
            raise LLMGenerationError(
                f"Error parsing DAG groups string into groups. Error: {str(e)}"
            )

        if len(groups) == 0:
            raise LLMGenerationError("Proposition groups empty.")

        # check all indices are unique
        flattened = [x for xs in groups for x in xs]
        if len(flattened) != len(set(flattened)):
            raise LLMGenerationError("DAG groups contain duplicate indices.")

        # check all indices are valid
        for idx in flattened:
            if idx < 0 or idx >= n_props:
                raise LLMGenerationError("DAG groups contain invalid indices.")

        return self.constraint_from_groups(groups, n_props)

    def to_plaintext(self, tc_constraint: TemporalConstraint) -> str:
        """Generate a string representation of the temporal constraint for annotation."""
        start = """
# ----------------------------------------
# TEMPORAL GROUPS
#    Place propositions in groups s.t. one group must be satisfied before the next.
#    Example:
#        [ [0, 1], [2, 3] ] means props 0 & 1 must be satisfied before props 2 & 3.
# ----------------------------------------
temporal_groups = [
"""
        end = "]\n"
        if tc_constraint is None:
            return start + end

        for group in self.groups_from_constraint(tc_constraint):
            start += f"    {group},\n"
        return start + end

    def from_plaintext(self, plaintext_str: str, n_props: int) -> TemporalConstraint:
        """
        Loads temporal groups from the plaintext evaluation str (possibly after manual
        human annotation). Returns a TemporalConstraint that can be loaded into Hab-LLM.
        """
        dag_lines = extract_lines_between(plaintext_str, "temporal_groups = [", "]")
        dag_lines = [l for l in dag_lines if l.strip() != ""]
        dag_str = "[" + ",".join(dag_lines) + "]"
        try:
            prop_groups = json.loads(dag_str)
        except Exception as e:
            raise LLMGenerationError(
                f"Error parsing DAG groups string into groups. Error: {str(e)}"
            )
        return self.constraint_from_groups(prop_groups, n_props)

    @staticmethod
    def groups_from_constraint(tc_constraint: TemporalConstraint) -> List[List[int]]:
        groups = tc_constraint.get_topological_generations()
        if len(groups) == 0:
            n_props = tc_constraint.n_propositions
            if n_props is None:
                raise AssertionError(
                    "n_propositions cannot be None in TemporalConstraint"
                )
            return [list(range(n_props))]
        return groups

    @staticmethod
    def constraint_from_groups(
        prop_groups: List[List[int]], n_props: int
    ) -> TemporalConstraint:
        dag_edges: List[Tuple[int, int]] = []
        for gen_idx in range(1, len(prop_groups)):
            prev_gen = prop_groups[gen_idx - 1]
            cur_gen = prop_groups[gen_idx]
            for i, j in itertools.product(prev_gen, cur_gen):
                dag_edges.append((i, j))
        try:
            return TemporalConstraint(dag_edges, n_props)
        except Exception as e:
            raise LLMGenerationError(
                f"Error producing TemporalConstraint. Error: {str(e)}"
            )


class TieParser(EvalGenParser):
    def from_llm(
        self, llm_str_output: str, propositions: List[EvaluationProposition]
    ) -> List[Union[SameArgConstraint, DifferentArgConstraint]]:
        """Parses the LLM output to an evaluation object (propositions or constraints)"""
        if llm_str_output == "":
            raise LLMGenerationError("[tie call] The LLM produced an empty output.")

        # remove the trailing text
        lines = llm_str_output.split("\n")
        end_idx = -1
        for i, l in enumerate(lines):
            if l == "]":
                end_idx = i
        if end_idx == -1:
            ties_str = llm_str_output.split("\n\n")[0]
        else:
            ties_str = "\n".join(lines[: end_idx + 1])

        # start the str with the right list opener
        if not ties_str.startswith("[["):
            ties_str = "[" + ties_str

        return self.str_to_constraints(ties_str, propositions, True)

    def to_plaintext(
        self,
        constraints: List[Union[SameArgConstraint, DifferentArgConstraint]],
        propositions: List[EvaluationProposition],
    ) -> str:
        """Generate a string representation of the tie constraints for annotation."""
        s = """
# ----------------------------------------
# TIE CONSTRAINTS
#    options: SameArgConstraint, DifferentArgConstraint
#    Args:
#        proposition_indices: List[int]
#        arg_indices: List[int]
#    Example:
#        SameArgConstraint([0, 2], [1, 1]). Means: Propositions 0 & 2 must
#        match values on the argument at argument index 1 and 1, respectively.
# ----------------------------------------
tie_constraints = [
"""
        same_arg_constraints = 0
        different_arg_constraints = 0
        for constraint in constraints:
            same_arg_constraints += int(isinstance(constraint, SameArgConstraint))
            different_arg_constraints += int(
                isinstance(constraint, DifferentArgConstraint)
            )

            # map the argument name to argument index for easier manual annotation
            arg_idxs = []
            for prop_idx, arg_name in zip(
                constraint.proposition_indices, constraint.args["arg_names"]
            ):
                predicate_name = propositions[prop_idx].function_name
                for i, arg in enumerate(self.predicate_vocab[predicate_name]):
                    if arg["name"] == arg_name:
                        arg_idxs.append(i)
                        break
                else:
                    raise LLMGenerationError("Error parsing arg_names to indices.")

            c_str = f"{constraint.__class__.__name__}({constraint.proposition_indices}, {arg_idxs})"
            s += f"    {c_str},\n"

        return s + "]\n"

    def from_plaintext(
        self, plaintext_str: str, propositions: List[EvaluationProposition]
    ) -> List[Union[SameArgConstraint, DifferentArgConstraint]]:
        """Loads the list of tie constraints from the plaintext evaluation str (possibly
        after manual human annotation). Converts indexed arguments to argument names and
        returns a list of tie constraints that can be loaded into Hab-LLM.
        """
        tie_lines = extract_lines_between(plaintext_str, "tie_constraints = [", "]")
        return self.str_to_constraints("\n".join(tie_lines), propositions, True)

    def str_to_constraints(
        self,
        ties_str: str,
        propositions: List[EvaluationProposition],
        link_is_indexed: bool,
    ) -> List[Union[SameArgConstraint, DifferentArgConstraint]]:
        ties_str_list = ties_str.removeprefix("[").removesuffix("]").split("\n")
        ties_str_list = [s for s in ties_str_list if s != ""]

        constraints: List[Union[SameArgConstraint, DifferentArgConstraint]] = []
        for l in ties_str_list:
            try:
                tree = ast.parse(l.removesuffix(",")).body[0]
                fn_name = tree.value.func.id  # type: ignore
            except Exception as e:
                raise LLMGenerationError(
                    f"Invalid constraint string: `{l}`. \nError: {str(e)}."
                )

            if fn_name not in ["SameArgConstraint", "DifferentArgConstraint"]:
                raise LLMGenerationError(
                    f"Constraint name `{fn_name}` not a supported tie constraint."
                )
            constraint_cls: Union[
                Type[SameArgConstraint], Type[DifferentArgConstraint]
            ] = getattr(evaluation_functions, fn_name)

            args = tree.value.args + [kw.value for kw in tree.value.keywords]  # type: ignore
            if len(args) != 2:
                raise LLMGenerationError("Improper number of constraint args.")

            try:
                prop_idxs: List[int] = ast.literal_eval(args[0])
                if not isinstance(prop_idxs, list):
                    raise ValueError("invalid type.")
                if len(prop_idxs) > 0 and not isinstance(prop_idxs[0], int):
                    raise ValueError("invalid type.")
                if len(prop_idxs) != len(set(prop_idxs)):
                    raise ValueError("duplicate indices.")
                for idx in prop_idxs:
                    if idx < 0 or idx >= len(propositions):
                        raise LLMGenerationError("invalid indices.")
            except Exception as e:
                raise LLMGenerationError(
                    f"Error parsing first constraint argument. Error: {e}"
                )

            arg_names: List[str] = []

            if link_is_indexed:
                # convert arg 2 from an argument index to argument names
                try:
                    arg_idxs = ast.literal_eval(args[1])
                    if isinstance(arg_idxs, int):
                        arg_idxs = [arg_idxs for _ in range(len(prop_idxs))]
                    elif not isinstance(arg_idxs, list) or (
                        len(arg_idxs) > 0 and not isinstance(arg_idxs[0], int)
                    ):
                        raise ValueError("invalid type.")
                except Exception as e:
                    raise LLMGenerationError(
                        f"Error parsing second constraint argument. Error: {e}"
                    )

                for prop_idx, arg_idx in zip(prop_idxs, arg_idxs):
                    prop_name = propositions[prop_idx].function_name
                    arg_names.append(self.predicate_vocab[prop_name][arg_idx]["name"])
            else:
                try:
                    arg_names = ast.literal_eval(args[1])
                    if not isinstance(arg_names, list):
                        raise ValueError("invalid type.")
                    if len(arg_names) > 0 and not isinstance(arg_names[0], str):
                        raise ValueError("invalid type.")
                except Exception as e:
                    raise LLMGenerationError(
                        f"Error parsing second constraint argument. Error: {e}"
                    )

            try:
                constraint = constraint_cls(
                    proposition_indices=prop_idxs, arg_names=arg_names
                )
            except AssertionError as e:
                raise LLMGenerationError(f"Same/Diff arg constraint error: {str(e)}")

            constraints.append(constraint)

        return constraints


class DependencyParser(EvalGenParser):
    def from_llm(
        self, llm_str_output: str, propositions: List[EvaluationProposition]
    ) -> List[Union[EvaluationPropositionDependency]]:
        """Parses the LLM output to an evaluation object (propositions or constraints)"""
        raise NotImplementedError

    def to_plaintext(self, dependencies: List[EvaluationPropositionDependency]) -> str:
        """Generate a string representation of the tie constraints for annotation."""
        s = """
# ----------------------------------------
# PROPOSITION DEPENDENCIES
#    types:
#        WhileSatisfied: evaluate propositions when depends_on propositions are True.
#        AfterSatisfied: evaluate propositions when depends_on propositions have each been satisfied at some point in the past.
#        AfterUnsatisfied: evaluate propositions when depends_on propositions were at some point satisfied and no longer are.
#        BeforeUnsatisfied: evaluate propositions when depends_on propositions have yet to be satisfied.
#    Args:
#        proposition_indices: List[int]
#        depends_on: List[int]
#    Example:
#        WhileSatisfied([1], [0])     Means: Proposition 1 will only be queried when Proposition 0 is True.
# ----------------------------------------
dependencies = [
"""
        dep_type_names = {
            "while_satisfied": "WhileSatisfied",
            "after_satisfied": "AfterSatisfied",
            "after_unsatisfied": "AfterUnsatisfied",
            "before_satisfied": "BeforeSatisfied",
        }
        for dep in dependencies:
            if dep.relation_type not in dep_type_names:
                raise LLMGenerationError(
                    f"Dependency type {dep.relation_type} does not exist."
                )

            s += f"    {dep_type_names[dep.relation_type]}({dep.proposition_indices}, {dep.depends_on}),\n"
        return s + "]\n"

    def from_plaintext(
        self, plaintext_str: str
    ) -> List[EvaluationPropositionDependency]:
        """Loads the list of proposition dependencies from the plaintext evaluation str
        (possibly after manual human annotation). See function `to_plaintext()` for the
        expected input string format.
        """
        try:
            dep_lines = extract_lines_between(plaintext_str, "dependencies = [", "]")
        except LLMGenerationError:
            return []

        dep_name_types = {
            "WhileSatisfied": "while_satisfied",
            "AfterSatisfied": "after_satisfied",
            "AfterUnsatisfied": "after_unsatisfied",
            "BeforeSatisfied": "before_satisfied",
        }
        deps: List[EvaluationPropositionDependency] = []
        for dep_str in dep_lines:
            dep_str = dep_str.rstrip().removesuffix(",")
            if dep_str == "":
                continue
            try:
                tree = ast.parse(dep_str).body[0]
            except Exception as e:
                raise LLMGenerationError(
                    f"Invalid dependency string: `{dep_str}`. \nError: {str(e)}."
                )
            fn_name = tree.value.func.id  # type: ignore
            if fn_name not in dep_name_types:
                raise LLMGenerationError(f"Dependency type {fn_name} does not exist.")
            relation_type = dep_name_types[fn_name]

            mode = "all"
            try:
                prop_idxs = ast.literal_eval(tree.value.args[0])  # type: ignore
                dep_idxs = ast.literal_eval(tree.value.args[1])  # type: ignore
                if len(tree.value.args) == 3:  # type: ignore
                    mode = tree.value.args[2]  # type: ignore
                    assert mode in {"any", "all"}
            except Exception as e:
                raise LLMGenerationError(
                    f"Invalid dependency arguments: `{dep_str}`. \nError: {str(e)}."
                )

            deps.append(
                EvaluationPropositionDependency(
                    proposition_indices=prop_idxs,
                    depends_on=dep_idxs,
                    relation_type=relation_type,
                    dependency_mode=mode,
                )
            )

        return deps


class SkipParser:
    @staticmethod
    def to_plaintext(default_to_skip: bool) -> str:
        """
        Generate a string representation of the episode skip option for annotation.
        Default to True for progress tracking.
        """
        return f"""
# ----------------------------------------
# mark True if the task has a fatal issue
# ----------------------------------------
skip_episode = {default_to_skip}
reason = ""
"""

    @staticmethod
    def from_plaintext(plaintext_str: str) -> Tuple[bool, str]:
        """
        Loads whether or not the episode should be skipped from the plaintext evaluation
        str (possibly after manual human annotation). Also returns the provided skip reason.
        """
        # returns skip_episode (bool) and the reason why (str)
        skipped, reason = False, ""
        for line in plaintext_str.split("\n"):
            line = line.lstrip()
            if line.startswith("skip_episode") and "True" in line:
                skipped = True
                continue
            if skipped and line.startswith("reason"):
                reason = line
        return skipped, reason


class InstructionParser:
    @staticmethod
    def to_plaintext(instruction: str) -> str:
        """Generate a string representation of the instruction for annotation."""
        instruction_str = textwrap.fill(instruction, 100, subsequent_indent="")
        header = """
# ----------------------------------------
# INSTRUCTION
#    modify as necessary, but keep in mind the scene is fixed.
# ----------------------------------------
"""
        return f'{header}instruction = """\n{instruction_str}\n"""\n'

    @staticmethod
    def from_plaintext(plaintext_str: str) -> str:
        """
        Loads the (potentially modified) instruction from the plaintext evaluation str.
        """
        lines = extract_lines_between(plaintext_str, 'instruction = """', '"""')
        instruction = ""
        for line in lines:
            line = line.lstrip().rstrip()
            if line.lower().startswith("instruction:"):
                continue
            instruction += line + " "

        # fix nonuniform spacing introduced by instruction filtering
        instruction = instruction.lstrip().rstrip()
        instruction = re.sub(r"\s+", " ", instruction)
        instruction = instruction.replace(" .", ".")
        return instruction


class TerminalSatisfactionParser:
    @staticmethod
    def to_plaintext(
        terminal_constraint: Optional[TerminalSatisfactionConstraint] = None,
    ) -> str:
        """Generate a string representation of constraint exclusion for annotation."""
        if terminal_constraint is None:
            excludes = []
        else:
            n_props = terminal_constraint.args["n_propositions"]
            prop_idxs = set(terminal_constraint.args["proposition_indices"])
            excludes = [i for i in range(n_props) if i not in prop_idxs]
        return f"""
# ----------------------------------------
# FINAL SATISFACTION CONSTRAINT:
#    We assume all propositions must remain satisfied to the end of the episode.
#    if a proposition *should* become unsatisfied, add its index here.
# ----------------------------------------
exclude_final_constraint = {excludes}
"""

    @staticmethod
    def from_plaintext(
        plaintext_str: str, propositions: List[EvaluationProposition]
    ) -> TerminalSatisfactionConstraint:
        """
        Loads the annotated exclusions to the terminal satisfaction constraint from the
        plaintext evaluation str. Returns a TerminalSatisfactionConstraint.
        """
        exclusions = set()
        for line in plaintext_str.split("\n"):
            line = line.lstrip().rstrip()
            if line.startswith("exclude_final_constraint"):
                try:
                    exclusions_parsed: List[int] = ast.literal_eval(
                        ast.parse(line).body[0].value  # type: ignore
                    )
                    if len(exclusions_parsed) and not all(
                        isinstance(i, int) for i in exclusions_parsed
                    ):
                        raise ValueError(f"{exclusions_parsed} should all be ints.")
                except Exception as e:
                    raise LLMGenerationError(
                        f"Invalid final satisfaction exclusion: `{line}`. \nError: {str(e)}."
                    )
                exclusions = set(exclusions_parsed)
                break

        return TerminalSatisfactionConstraint(
            proposition_indices=[
                i for i in range(len(propositions)) if i not in exclusions
            ]
        )


def temporal_words_in_str(instruction) -> bool:
    """Returns True if any temporal words are found in the instruction."""
    try:
        nltk.word_tokenize("")
    except LookupError:
        nltk.download("punkt")
    instruction_words = set(w.lower() for w in nltk.word_tokenize(instruction))
    if len(instruction_words & TEMPORAL_WORDS) > 0:
        return True

    # these words count as temporal only if followed by a comma.
    instruction_lowered = instruction.lower()
    return any(cw in instruction_lowered for cw in TEMPORAL_WORDS_COMMA)


def metadata_to_state_string(
    metadata: Dict[str, Any], state_negations_map: Dict[str, str]
) -> str:
    """
    Make a textual scene representation for LLM to understand the scene.
    Note: objects are referred to at the instance level. Rooms are referred to
    at the category level (eg bedroom_1 -> bedroom). Furniture are referred to
    at the category level first (eg table), and then later the room-level
    (eg table in livingroom). Outline:

    Objects:
        * [object-name] (list of states)
        ...
    Furniture:
        * [furniture-cat]
        ...
    Rooms:
        * [room-cat]
        ...
    Object-Furniture-Room Relations:
        * [object-name] on [furniture-cat] in [room-cat]
        ...
    Furniture-Room Relations:
        * [furniture-cat] in [room-cat]
        ...
    """

    def strip_id(s: str) -> str:
        """removes the suffix _[int]"""
        if s.split("_")[-1].isdigit():
            return "_".join(s.split("_")[:-1])
        return s

    def object_states_to_str(obj_states_dict: Dict[str, bool]) -> str:
        if len(obj_states_dict) == 0:
            return ""
        s = " ("
        for k, v in obj_states_dict.items():
            state = (k if v else state_negations_map[k]).removeprefix("is_")
            s += f"{state}, "
        return s.removesuffix(", ") + ")"

    state_str = "Objects:\n"
    for obj in metadata["objects"]:
        state_str += f"    * {obj}"
        if "object_to_states" in metadata and obj in metadata["object_to_states"]:
            state_str += object_states_to_str(metadata["object_to_states"][obj])
        state_str += "\n"

    state_str += "Furniture:\n"
    for furn in sorted({strip_id(f) for f in metadata["recep_to_description"]}):
        state_str += f"    * {furn}\n"

    state_str += "Rooms:\n"
    for room in sorted({strip_id(r) for r in metadata["rooms"]}):
        state_str += f"    * {room}\n"

    state_str += "Object-Furniture-Room Relations:\n"
    for obj, room in metadata["object_to_room"].items():
        recep = strip_id(metadata["object_to_recep"][obj])
        room = strip_id(room)
        if recep == "":
            state_str += f"    * {obj} in {room}\n"
        else:
            state_str += f"    * {obj} on {recep} in {room}\n"

    state_str += "Furniture-Room Relations:\n"
    lines = set()
    for recep, room in metadata["recep_to_room"].items():
        line = f"    * {strip_id(recep)} in {strip_id(room)}\n"
        if line not in lines:
            state_str += line
            lines.add(line)

    return state_str


def extract_lines_between(
    s: str, starting_str: str, ending_str: Optional[str] = None
) -> List[str]:
    """
    Return a list of lines that lie between two specified lines in a multi-line string.
    """
    lines = [l.lstrip().rstrip().rstrip(",") for l in s.split("\n")]
    for i, l in enumerate(lines):
        if l == starting_str:
            start_idx = i
            break
    else:
        raise LLMGenerationError("parse error: starting_str not found")

    if ending_str is None:
        return lines[start_idx + 1 :]

    for i, l in enumerate(lines[start_idx:]):
        if l == ending_str:
            end_idx = start_idx + i
            break
    else:
        raise LLMGenerationError("parse error: ending_str not found")
    return lines[start_idx + 1 : end_idx]


def proposition_to_llm_output_str(prop, metadata: Dict[str, Any]) -> str:
    """
    Formats a proposition as the LLM output that generated this proposition.
    Use case: dynamic insertion of relevant templated examples to eval gen.
    """

    def objects_to_macro(objs):
        """Object instances in a single string joined by `or`"""
        x = " or ".join(objs)
        return f'"{x}"'

    def furn_to_macro(furn, metadata):
        """furniture class, not instance. Adds `in [room class]` if all in same room"""
        furn_class = ""
        room_classes = set()
        for f in furn:
            if f.split("_")[-1].isdigit():
                furn_class = "_".join(f.split("_")[:-1])
            else:
                furn_class = f
            room = (
                metadata["recep_to_room"][f]
                if f in metadata["recep_to_room"]
                else metadata["object_to_room"][f]
            )
            room = "_".join(room.split("_")[:-1])
            room_classes.add(room)

        if len(room_classes) > 1:
            return f'"{furn_class}"'
        return f'"{furn_class} in {next(iter(room_classes))}"'

    def rooms_to_macro(rooms):
        """room class, not room instance."""
        rooms = ["_".join(r.split("_")[:-1]) for r in rooms]
        return f'"{rooms[0]}"'

    handle_to_semantic_name = {}
    for k, v in metadata["object_to_handle"].items():
        handle_to_semantic_name[v] = k
    for k, v in metadata["recep_to_handle"].items():
        handle_to_semantic_name[v] = k
    for k, v in metadata["room_to_id"].items():
        handle_to_semantic_name[v] = k

    fn_name = prop["function_name"]
    if fn_name in ["is_on_top", "is_inside"]:
        obj_macro = objects_to_macro(
            [handle_to_semantic_name[o] for o in prop["args"]["object_handles"]]
        )
        furn_macro = furn_to_macro(
            [handle_to_semantic_name[r] for r in prop["args"]["receptacle_handles"]],
            metadata,
        )
        return f"\n[FN] {prop['function_name']}({obj_macro}, {furn_macro}) [/FN]"
    elif fn_name in ["is_in_room"]:
        obj_macro = objects_to_macro(
            [handle_to_semantic_name[o] for o in prop["args"]["object_handles"]]
        )
        room_macro = rooms_to_macro(
            [handle_to_semantic_name[r] for r in prop["args"]["room_ids"]]
        )
        return f"\n[FN] {prop['function_name']}({obj_macro}, {room_macro}) [/FN]"
    elif fn_name in ["is_next_to"]:
        macro_a = objects_to_macro(
            [handle_to_semantic_name[o] for o in prop["args"]["entity_handles_a"]]
        )
        ent_b = [handle_to_semantic_name[o] for o in prop["args"]["entity_handles_b"]]
        if ent_b[0] in metadata["recep_to_description"]:
            macro_b = furn_to_macro(ent_b, metadata)
        else:
            macro_b = objects_to_macro(ent_b)
        return f"\n[FN] is_next_to({macro_a}, {macro_b}) [/FN]"
    elif fn_name in [
        "is_on_floor",
        "is_clean",
        "is_dirty",
        "is_filled",
        "is_empty",
        "is_powered_on",
        "is_powered_off",
    ]:
        obj_macro = objects_to_macro(
            [handle_to_semantic_name[o] for o in prop["args"]["object_handles"]]
        )
        return f"\n[FN] {prop['function_name']}({obj_macro}) [/FN]"
    else:
        raise NotImplementedError


def trim_template_to_fit(template_str: str, max_str_len: int = 2600) -> str:
    """
    If the template string is longer than max_str_len, remove the furniture-room
    relations. Helps avoid OOM errors with LLM.
    """
    if len(template_str) <= max_str_len:
        return template_str
    # otherwise, remove furniture-room relations
    remove = False
    new_lines = []
    for line in template_str.split("\n"):
        if "Furniture-Room Relations:" in line:
            remove = True

        if not remove:
            new_lines.append(line)

        if remove and line.strip() == "":
            remove = False

    template_str = "\n".join(new_lines)
    if len(template_str) <= max_str_len:
        return template_str

    raise LLMGenerationError("template example contains too many tokens. OOM expected.")


def split_into_temporal_sub_instructions(instruction: str) -> List[str]:
    """
    Split an instruction into a list of temporal sub-instructions
    using the temporal word lists as delimiters.
    """
    inst = instruction.lower()
    delimiters = TEMPORAL_WORDS | TEMPORAL_WORDS_COMMA
    regex_pattern = "|".join(map(re.escape, delimiters))
    return [s.lstrip(",").strip() for s in re.split(regex_pattern, inst)]
