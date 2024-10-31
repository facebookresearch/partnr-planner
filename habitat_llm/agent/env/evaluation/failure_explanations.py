# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import numpy as np
from habitat.sims.habitat_simulator import sim_utilities as sutils

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    DifferentArgConstraint,
    EvaluationConstraint,
    EvaluationProposition,
    SameArgConstraint,
    TemporalConstraint,
    TerminalSatisfactionConstraint,
)
from habitat_llm.sims.metadata_interface import MetadataInterface


def handles_to_class_str(handles: List[str], mi: MetadataInterface) -> str:
    """Converts the list of handles to semantic classes joined by `or`"""
    names = [sutils.object_shortname_from_handle(h) for h in handles]
    cats = {f'"{" ".join(mi.get_object_category(n).split("_"))}"' for n in names}

    if None in cats:
        raise AssertionError(f"handles {handles} failed to map to categories.")
    return " or ".join(cats)


def get_object_class(proposition, mi: MetadataInterface) -> str:
    """Converts the object handles of the proposition to natural language"""
    return handles_to_class_str(proposition.args["object_handles"], mi)


def get_furniture_class(proposition, mi: MetadataInterface) -> str:
    """Converts the receptacle handles of the proposition to natural language"""
    return handles_to_class_str(proposition.args["receptacle_handles"], mi)


def get_room_class(proposition, mi: MetadataInterface) -> str:
    """Converts the room IDs of the is_in_room proposition to natural language"""
    if proposition.function_name != "is_in_room":
        raise AssertionError("proposition must be is_in_room")

    room_cats = {f'"{r.split(".")[0]}"' for r in proposition.args["room_ids"]}
    return " or ".join(room_cats)


def get_clustered_object_classes(
    proposition: EvaluationProposition, mi: MetadataInterface
) -> str:
    """
    Converts the list of clustered object handles in is_clustered to natural language.
    """
    if proposition.function_name != "is_clustered":
        raise AssertionError("proposition must be is_clustered")

    handle_strs = [handles_to_class_str(hs, mi) for hs in proposition.args["*args"]]
    for i in range(len(handle_strs)):
        if " or " in handle_strs[i]:
            handle_strs[i] = f"({handle_strs[i]})"  # TODO: test
    return " and ".join(handle_strs)


def derive_proposition_str(
    prop: EvaluationProposition, mi: MetadataInterface, should: bool = False
) -> str:
    """
    A natural language version of the evaluation proposition.
    Names each entity uses semantic class labels.
    """
    should_txt = " should have been" if should else ""
    if prop.function_name == "is_in_room":
        obj = get_object_class(prop, mi)
        room = get_room_class(prop, mi)
        return f"{obj}{should_txt} moved to the {room}"

    if prop.function_name == "is_inside":
        obj = get_object_class(prop, mi)
        furn = get_furniture_class(prop, mi)
        return f"{obj}{should_txt} placed inside the {furn}"

    if prop.function_name == "is_on_top":
        obj = get_object_class(prop, mi)
        furn = get_furniture_class(prop, mi)
        return f"{obj}{should_txt} placed on top of the {furn}"

    if prop.function_name == "is_on_floor":
        obj = get_object_class(prop, mi)
        return f"{obj}{should_txt} placed on the floor"

    if prop.function_name == "is_next_to":
        entity_a = handles_to_class_str(prop.args["entity_handles_a"], mi)
        entity_b = handles_to_class_str(prop.args["entity_handles_b"], mi)
        return f"{entity_a}{should_txt} placed next to {entity_b}"

    if prop.function_name == "is_clustered":
        objs_str = get_clustered_object_classes(prop, mi)
        return f"{objs_str}{should_txt} clustered"

    # object states
    obj = get_object_class(prop, mi)
    state_str = " ".join(prop.function_name.split("_")[1:])
    return f"Object {obj}{should_txt} {state_str}"


def derive_failed_proposition_message(
    propositions: List[EvaluationProposition],
    proposition_satisfied_at: List[int],
    mi: MetadataInterface,
) -> str:
    """Explain in natural language which task propositions were not satisfied"""
    msg = ""
    for i in range(len(propositions)):
        if proposition_satisfied_at[i] != -1:
            continue
        msg += "\n - "
        prop = propositions[i]
        msg += derive_proposition_str(prop, mi, should=True)

    return "Episode failed. Missing steps:" + msg if msg != "" else ""


def derive_failed_constraint_message(
    propositions: List[EvaluationProposition],
    constraints: List[EvaluationConstraint],
    constraint_satisfaction: np.ndarray,
    mi: MetadataInterface,
) -> str:
    """Explain in natural language which task constraints were not satisfied"""

    def temporal_msg(c: TemporalConstraint, constraint_satisfied: np.ndarray) -> str:
        msg = "\nSteps were completed out of order:"
        for i in range(constraint_satisfied.shape[0]):
            if constraint_satisfied[i]:
                continue
            msg += "\n - "
            msg += derive_proposition_str(propositions[i], mi)
            msg += " should have been completed after:"
            for j in sorted(e[0] for e in c.dag.in_edges(i)):
                msg += "\n   - "
                msg += derive_proposition_str(propositions[j], mi)

        return msg

    def terminal_msg(constraint_satisfied: np.ndarray) -> str:
        msg = "\nCompleted steps were later undone:"
        for i in range(constraint_satisfied.shape[0]):
            if constraint_satisfied[i]:
                continue
            msg += "\n - "
            msg += derive_proposition_str(propositions[i], mi)
        return msg

    def arg_msg(
        c: Union[SameArgConstraint, DifferentArgConstraint], is_same: bool
    ) -> str:
        is_object = (
            "object_handles" in c.args["arg_names"]
            or "entity_handles_a" in c.args["arg_names"]
        )
        is_room = "room_ids" in c.args["arg_names"]
        if is_object:
            if is_same:
                msg = "\nThe same object should have been used for the following placements:"
            else:
                msg = "\nDifferent objects should have been used for the following placements:"
        else:
            receiver = "room" if is_room else "furniture"
            if is_same:
                msg = f"\nPlacements should have been made with the same {receiver}:"
            else:
                receiver = "rooms" if receiver == "room" else receiver
                msg = f"\nPlacements should have been made with different {receiver}:"

        for i in c.proposition_indices:
            msg += "\n - "
            msg += derive_proposition_str(propositions[i], mi)
        return msg

    msg = ""
    for i, c in enumerate(constraints):
        constraint_satisfied = constraint_satisfaction[i]
        if constraint_satisfied.all():
            continue

        if isinstance(c, TemporalConstraint):
            msg += temporal_msg(c, constraint_satisfied)
        if isinstance(c, TerminalSatisfactionConstraint):
            msg += terminal_msg(constraint_satisfied)
        if isinstance(c, SameArgConstraint):
            msg += arg_msg(c, is_same=True)
        if isinstance(c, DifferentArgConstraint):
            msg += arg_msg(c, is_same=False)

    prefix = "Episode failed. All steps were completed, but constraints were broken:"
    return prefix + msg if msg != "" else ""


def derive_evaluation_explanation(
    propositions: List[EvaluationProposition],
    constraints: List[EvaluationConstraint],
    proposition_satisfied_at: List[int],
    constraint_satisfaction: np.ndarray,
    mi: MetadataInterface,
) -> str:
    """
    Compiles a natural language text description of the status of the task in relation to
    the evaluation function. The objects, furniture, and room instances are parsed from
    their respective handle/id into a semantic class name using the metadata interface.
    This level of instance disambiguation isn't perfect, but may be helpful for HITL
    applications.

    Flow:
    1. check if all propositions have been satisfied. If not, return a message explaining
        which propositions were failed.
    2. check if all constraints have been satisfied. If not, return a message explaining
        which constraints were failed.
    3. Return the empty string.
    """
    has_failed_proposition = any(x == -1 for x in proposition_satisfied_at)
    if has_failed_proposition:
        return derive_failed_proposition_message(
            propositions, proposition_satisfied_at, mi
        )
    if not constraint_satisfaction.all():
        return derive_failed_constraint_message(
            propositions, constraints, constraint_satisfaction, mi
        )
    return ""
