# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from habitat_llm.agent.env.evaluation.predicate_wrappers import PropositionResult


@dataclass
class EvaluationProposition:
    function_name: str
    args: Dict[str, Any]


@dataclass
class EvaluationPropositionDependency:
    """
    Supported relation_types:
        - while_satisfied
        - after_satisfied
        - after_unsatisfied
        - before_satisfied
    Supported dependency_modes:
        - all
        - any
    """

    proposition_indices: List[int]
    depends_on: List[int]
    relation_type: str
    dependency_mode: str = "all"


class EvaluationConstraint(ABC):
    """Defines an abstract constraint over propositions"""

    star_args = "*args"

    def __init__(self, **kwargs) -> None:
        self.args = deepcopy(kwargs)  # save initialization args for serialization.
        super().__init__()

    def __getstate__(self):
        """Return the state of the constraint for serialization."""
        return {"type": self.__class__.__name__, "args": self.args}

    def __setstate__(self, d):
        """Re-initialize the constraint for deserialization."""
        self.__init__(**d["args"])

    @staticmethod
    def assert_proposition_indices_valid(
        proposition_indices: Iterable[int], n_propositions: int
    ) -> None:
        if not all(i < n_propositions and i >= 0 for i in proposition_indices):
            raise AssertionError(
                f"Proposition indices {proposition_indices} invalid"
                f" for n_propositions={n_propositions}"
            )

    def _assert_arg_exists(
        self, arg_name: str, state: List[PropositionResult], prop_idx: int
    ) -> None:
        """Asserts that arg_name exists in the PropositionResult at state index `prop_idx`"""
        prop_result = state[prop_idx]
        missing_star_args = (
            arg_name.startswith(self.star_args)
            and self.star_args not in prop_result.info
        )
        missing_args = arg_name not in prop_result.info
        if missing_star_args or missing_args:
            raise ValueError(
                f"Arg `{arg_name}` missing from proposition {prop_idx} info dict."
            )

    def _satisfying_prop_value(
        self, prop_result: PropositionResult, arg_name: str
    ) -> Any:
        """
        Return the satisfying value of the argument `arg_name`.
        If the argument is variable-number (`*args_[i]`),
        return the satisfying value of prop_result["*args"][i].
        """
        if not arg_name.startswith(self.star_args):
            return prop_result.info[arg_name]
        arg_idx = int(arg_name.split("_")[-1])
        return prop_result.info[self.star_args][arg_idx]

    def _satisfying_values_match(self, v1: Any, v2: Any) -> bool:
        """
        Checks that two proposition-satisfying argument values match.
        If a value is iterable, then check that at least one element matches.
        """
        if isinstance(v1, Iterable) and isinstance(v2, Iterable):
            return bool(len(set(v1) & set(v2)))
        if isinstance(v1, Iterable) and not isinstance(v2, Iterable):
            return v2 in v1
        if not isinstance(v1, Iterable) and isinstance(v2, Iterable):
            return v1 in v2
        return v1 == v2

    def update_unrolled_proposition(
        self,
        propositions: List[EvaluationProposition],
        idx_orig: int,
        idx_new: int,
    ) -> List["EvaluationConstraint"]:
        """
        Propositions with argument number>1 are unrolled to n propositions. If an init
        param of a constraint relies on these propositions, this is where you should
        update it. If an update requires a new constraint, return it.
        Args:
            propositions: the list of propositions including the unrolled proposition
            idx_orig: the index of the original proposition being unrolled
            idx_new: the index of the new proposition with a lowered `number` arg.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        state_sequence: List[List[PropositionResult]],
        proposition_satisfied_at: List[int],
    ) -> List[bool]:
        """
        Args:
            state_sequence: a sequence of states. A state is List[PropositionResult].
            proposition_satisfied_at: encodes the timestep proposition i was satisfied.
                -1 indicates not satisfied.

        Returns:
            A list of booleans where the ith element is False if proposition i is
                invalidated by the constraint.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class SameArgConstraint(EvaluationConstraint):
    """
    Requires the arg used to satisfy a proposition to be the same
    across a pre-determined set of propositions.
    """

    def __init__(
        self,
        proposition_indices: List[int],
        arg_names: List[str],
        n_propositions: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            proposition_indices=proposition_indices,
            arg_names=arg_names,
            n_propositions=n_propositions,
        )

        self.proposition_indices = proposition_indices
        if len(self.proposition_indices) != len(arg_names):
            raise AssertionError("proposition_indices has different len than arg_names")
        self.arg_names: Dict[int, str] = {}
        for prop_idx, arg_name in zip(proposition_indices, arg_names):
            self.arg_names[prop_idx] = arg_name

        if n_propositions is None:
            return

        self.assert_proposition_indices_valid(self.proposition_indices, n_propositions)

    def update_unrolled_proposition(
        self, propositions: List[EvaluationProposition], idx_orig: int, idx_new: int
    ) -> List["EvaluationConstraint"]:
        """Updates proposition_indices"""
        if idx_orig in self.proposition_indices:
            self.proposition_indices.append(idx_new)
            self.arg_names[idx_new] = self.arg_names[idx_orig]
        return []

    def __call__(
        self,
        state_sequence: List[List[PropositionResult]],
        proposition_satisfied_at: List[int],
    ) -> List[bool]:
        constraints_valid = [True for _ in range(len(proposition_satisfied_at))]
        if len(state_sequence) == 0:
            return constraints_valid

        idxs = self.proposition_indices
        idxs_to_check = [idx for idx in idxs if proposition_satisfied_at[idx] != -1]
        idxs_satisfied_at = [
            # (proposition index, when satisfied, argument name)
            (idx, proposition_satisfied_at[idx], self.arg_names[idx])
            for idx in idxs_to_check
        ]

        if len(idxs_satisfied_at) < 2:
            # no ties to check.
            return constraints_valid

        # sort by when satisfied to find which argument value is "ground truth"
        idxs_satisfied_at.sort(key=lambda t: t[1])

        # get the arg value that satisfies the first proposition satisfied.
        first_prop_idx, first_prop_satisfied_at, arg_name = idxs_satisfied_at[0]
        state = state_sequence[first_prop_satisfied_at]
        self._assert_arg_exists(arg_name, state, first_prop_idx)
        satisfying_value = self._satisfying_prop_value(state[first_prop_idx], arg_name)

        # check that each subsequent prop is satisfied by the same value. if not, invalidate it.
        for prop_idx, prop_satisfied_at, arg_name in idxs_satisfied_at[1:]:
            state = state_sequence[prop_satisfied_at]
            self._assert_arg_exists(arg_name, state, prop_idx)
            if not self._satisfying_values_match(
                self._satisfying_prop_value(state[prop_idx], arg_name), satisfying_value
            ):
                constraints_valid[prop_idx] = False

        return constraints_valid

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"proposition_indices={self.proposition_indices},"
            f" arg_names={self.arg_names})"
        )


class DifferentArgConstraint(EvaluationConstraint):
    """
    Requires the arg used to satisfy a proposition to be unique
    within a pre-determined set of propositions.
    """

    def __init__(
        self,
        proposition_indices: List[int],
        arg_names: List[str],
        n_propositions: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            proposition_indices=proposition_indices,
            arg_names=arg_names,
            n_propositions=n_propositions,
        )

        self.proposition_indices = proposition_indices
        if len(self.proposition_indices) != len(arg_names):
            raise AssertionError("proposition_indices has different len than arg_names")
        self.arg_names: Dict[int, str] = {}
        for prop_idx, arg_name in zip(proposition_indices, arg_names):
            self.arg_names[prop_idx] = arg_name

        if n_propositions is None:
            return

        self.assert_proposition_indices_valid(self.proposition_indices, n_propositions)

    def update_unrolled_proposition(
        self,
        propositions: List[EvaluationProposition],
        idx_orig: int,
        idx_new: int,
    ) -> List["EvaluationConstraint"]:
        """
        This creates a new DifferentArgConstraint where the proposition_indices are the
        same but with idx_new instead of idx_orig.
        """
        if idx_orig not in self.proposition_indices:
            return []

        new_proposition_indices = [idx_new]
        new_arg_names = [self.arg_names[idx_orig]]
        for idx in self.proposition_indices:
            if idx != idx_orig:
                new_proposition_indices.append(idx)
                new_arg_names.append(self.arg_names[idx])

        return [
            DifferentArgConstraint(
                proposition_indices=new_proposition_indices,
                arg_names=new_arg_names,
                n_propositions=len(propositions),
            )
        ]

    def __call__(
        self,
        state_sequence: List[List[PropositionResult]],
        proposition_satisfied_at: List[int],
    ) -> List[bool]:
        constraints_valid = [True for _ in range(len(proposition_satisfied_at))]
        if len(state_sequence) == 0:
            return constraints_valid

        idxs = self.proposition_indices
        idxs_to_check = [idx for idx in idxs if proposition_satisfied_at[idx] != -1]
        idxs_satisfied_at = [
            # (proposition index, when satisfied, argument name)
            (idx, proposition_satisfied_at[idx], self.arg_names[idx])
            for idx in idxs_to_check
        ]

        if len(idxs_satisfied_at) < 2:
            # no ties to check.
            return constraints_valid

        # sort by when satisfied to find which argument value is "ground truth"
        idxs_satisfied_at.sort(key=lambda t: t[1])

        # get the arg value that satisfies the first proposition satisfied.
        first_prop_idx, first_prop_satisfied_at, arg_name = idxs_satisfied_at[0]
        state = state_sequence[first_prop_satisfied_at]
        self._assert_arg_exists(arg_name, state, first_prop_idx)
        pivot_arg_values = {
            self._satisfying_prop_value(state[first_prop_idx], arg_name)
        }

        # check each subsequent prop is satisfied by a unique value. if not, invalidate it.
        for prop_idx, prop_satisfied_at, arg_name in idxs_satisfied_at[1:]:
            state = state_sequence[prop_satisfied_at]
            self._assert_arg_exists(arg_name, state, prop_idx)
            satisfying_value = self._satisfying_prop_value(state[prop_idx], arg_name)
            if any(
                self._satisfying_values_match(satisfying_value, sv)
                for sv in pivot_arg_values
            ):
                constraints_valid[prop_idx] = False
            pivot_arg_values.add(satisfying_value)

        return constraints_valid

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"proposition_indices={self.proposition_indices},"
            f" arg_names={self.arg_names})"
        )


class TemporalConstraint(EvaluationConstraint):
    """
    Enforces temporal constraints over propositions (eg, proposition i must be satisfied
    before proposition j). This information is encoded in a directed acyclic graph (DAG).
    In the example

        dag_edges = [(0, 1), (1, 2)]

    prop 1 must be satisfied after prop 0, and prop 2 must be satisfied after prop 1.
    """

    dag: nx.DiGraph
    dependencies: Dict[int, Set[int]]

    def __init__(
        self,
        dag_edges: List[Tuple[int, int]],
        n_propositions: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(dag_edges=dag_edges, n_propositions=n_propositions)

        self.dag = nx.DiGraph(dag_edges)
        if not nx.is_directed_acyclic_graph(self.dag):
            raise AssertionError("Graph is not a DAG")

        self.n_propositions = n_propositions
        self.dependencies = {node: nx.ancestors(self.dag, node) for node in self.dag}

        if n_propositions is None:
            return

        idxs_to_check = set(self.dependencies.keys())
        for deps in self.dependencies.values():
            idxs_to_check |= deps
        self.assert_proposition_indices_valid(idxs_to_check, n_propositions)

    def update_unrolled_proposition(
        self, propositions: List[EvaluationProposition], idx_orig: int, idx_new: int
    ) -> List["EvaluationConstraint"]:
        """This updates the DAG."""
        if idx_orig not in self.dag:
            return []

        for neighbor in self.dag.successors(idx_orig):
            self.dag.add_edge(idx_new, neighbor)
        for neighbor in self.dag.predecessors(idx_orig):
            self.dag.add_edge(neighbor, idx_new)
        self.dependencies[idx_new] = deepcopy(self.dependencies[idx_orig])
        return []

    def get_topological_generations(self) -> List[List[int]]:
        """Extract topological generations from the DAG (proposition groups).
        Note: there can be many such stratifications in a DAG. This returns one of them.
        """
        return list(nx.topological_generations(self.dag))

    def __call__(
        self,
        state_sequence: List[List[PropositionResult]],
        proposition_satisfied_at: List[int],
    ) -> List[bool]:
        """
        Apply the temporal constraints encoded in self.dependencies to the propositions.
        The indices are aligned across self.dependencies, proposition_satisfied_at, and
        constraint_valid. Returns a list of booleans where the ith element is True if
        proposition i is still valid after applying the constraint.
        """
        constraint_valid = [True for _ in range(len(proposition_satisfied_at))]

        for prop_idx, dependencies in self.dependencies.items():
            # if no dependencies, skip
            if len(dependencies) == 0:
                continue

            dependencies_satisfied_at = {
                proposition_satisfied_at[idx] for idx in dependencies
            }

            # all dependencies must be satisfied
            if -1 in dependencies_satisfied_at:
                constraint_valid[prop_idx] = False
                continue

            # all dependencies must be satisfied before the prop at prop_idx.
            if proposition_satisfied_at[prop_idx] <= max(dependencies_satisfied_at):
                constraint_valid[prop_idx] = False

        return constraint_valid

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"dag_edges={self.dag.edges},"
            f" dependencies={self.dependencies})"
        )


class TerminalSatisfactionConstraint(EvaluationConstraint):
    """
    Requires a satisfied proposition to remain satisfied in the final state of the task.
    """

    def __init__(
        self,
        proposition_indices: List[int],
        n_propositions: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            proposition_indices=proposition_indices, n_propositions=n_propositions
        )

        self.proposition_indices = proposition_indices

        if n_propositions is None:
            return

        self.assert_proposition_indices_valid(self.proposition_indices, n_propositions)

    def update_unrolled_proposition(
        self, propositions: List[EvaluationProposition], idx_orig: int, idx_new: int
    ) -> List["EvaluationConstraint"]:
        """Updates proposition_indices"""
        if idx_orig in self.proposition_indices:
            self.proposition_indices.append(idx_new)
        return []

    def __call__(
        self,
        state_sequence: List[List[PropositionResult]],
        proposition_satisfied_at: List[int],
    ) -> List[bool]:
        constraints_valid = [True for _ in range(len(proposition_satisfied_at))]
        if len(state_sequence) == 0:
            return constraints_valid

        idxs = self.proposition_indices
        idxs_to_check = [idx for idx in idxs if proposition_satisfied_at[idx] != -1]
        final_state = state_sequence[-1]
        for idx in idxs_to_check:
            constraints_valid[idx] = final_state[idx].is_satisfied

        return constraints_valid

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"proposition_indices={self.proposition_indices})"
        )


def unroll_propositions_with_number(
    propositions: List[EvaluationProposition],
    dependencies: List[EvaluationPropositionDependency],
    constraints: List[EvaluationConstraint],
) -> Tuple[
    List[EvaluationProposition],
    List[EvaluationPropositionDependency],
    List[EvaluationConstraint],
]:
    """
    Propositions with a number argument should be considered as n different
    propositions when evaluating % Completion. This function unrolls number=n
    down to number=1 adding a new proposition for each i.
    """
    for prop_idx in range(len(propositions)):
        prop = propositions[prop_idx]
        if "number" not in prop.args:
            continue
        if isinstance(prop.args["number"], list):
            # what does it mean to unroll number=[n, ..., m]?
            # for now we don't unroll this. we can provide a solution if we need for analysis.
            warnings.warn(
                f"Skipping proposition unroll of proposition {prop.function_name}."
                f" Number {prop.args['number']} is a list.",
                stacklevel=2,
            )
            continue

        # unroll proposition
        for i in range(1, prop.args["number"]):
            # copy the proposition and unroll it one step
            new_prop = deepcopy(prop)
            new_prop.args["number"] = i
            propositions.append(new_prop)
            new_prop_idx = len(propositions) - 1

            # update the constraints
            for j in range(len(constraints)):
                new_constraints = constraints[j].update_unrolled_proposition(
                    propositions, idx_orig=prop_idx, idx_new=new_prop_idx
                )
                constraints.extend(new_constraints)

            # update proposition dependencies
            for j in range(len(dependencies)):
                if prop_idx in dependencies[j].proposition_indices:
                    dependencies[j].proposition_indices.append(i)
                # NOTE: we do not unroll propositions in dependencies[j].depends_on. Reason:
                # if dependency_mode is "all", it would be redundant.
                # if dependency_mode is "any", it would be wrong.

    return propositions, dependencies, constraints


def dependency_is_satisfied(
    dep: EvaluationPropositionDependency, state_sequence: List[List[PropositionResult]]
) -> bool:
    def prop_currently_satisfied(prop_idx):
        """The proposition is currently satisfied."""
        if not len(state_sequence):
            return False
        return state_sequence[-1][prop_idx].is_satisfied

    def prop_has_been_satisfied(prop_idx):
        """The proposition was satisfied at some point in time."""
        return any(s[prop_idx].is_satisfied for s in state_sequence)

    def prop_satisfied_became_unsatisfied(prop_idx):
        """The proposition was satisfied at one point in time and later became unsatisfied."""
        if len(state_sequence) < 2:
            return False
        return any(
            state_sequence[i][prop_idx].is_satisfied
            and not state_sequence[i + 1][prop_idx].is_satisfied
            for i in range(len(state_sequence) - 1)
        )

    if not dep.dependency_mode in ["all", "any"]:
        raise ValueError("Invalid dependency mode encountered:", dep.dependency_mode)
    dep_mode_fn = all if dep.dependency_mode == "all" else any
    if dep.relation_type == "while_satisfied":
        return dep_mode_fn(prop_currently_satisfied(p) for p in dep.depends_on)
    if dep.relation_type == "after_satisfied":
        return dep_mode_fn(prop_has_been_satisfied(p) for p in dep.depends_on)
    if dep.relation_type == "after_unsatisfied":
        return dep_mode_fn(prop_satisfied_became_unsatisfied(p) for p in dep.depends_on)
    if dep.relation_type == "before_satisfied":
        return dep_mode_fn(not prop_has_been_satisfied(p) for p in dep.depends_on)
    raise ValueError(
        f"EvaluationPropositionDependency has unsupported relation_type `{dep.relation_type}.`"
    )


def determine_propositions_to_evaluate(
    state_sequence: List[List[PropositionResult]],
    propositions: List[EvaluationProposition],
    dependencies: List[EvaluationPropositionDependency],
) -> Set[int]:
    """
    Determine the evaluation propositions to evaluate based on
    temporal dependencies and the current state sequence.
    """
    props_to_check = set(range(len(propositions)))
    for dep in dependencies:
        if dependency_is_satisfied(dep, state_sequence):
            continue
        for prop_idx in dep.proposition_indices:
            props_to_check.discard(prop_idx)

    return props_to_check


def apply_constraint_satisfaction(
    constraints: List[EvaluationConstraint],
    state_sequence: List[List[PropositionResult]],
    prop_satisfied_at: List[int],
) -> np.ndarray:
    """Applies each evaluation constraint independently to the state sequence. Returns a
    2d array where rows are constraints, cols are propositions, and the value is False
    if any constraint has invalidated the proposition.

    Example:
        np.array([
            [T, T, T],  # constraint 0: propositions 0,1,2 remain valid
            [T, F, F],  # constraint 1: propositions 1,2 have been invalidated
        ])
        Suppose constraint 1 is a TemporalConstraint(dag_edges=((0, 1), (0, 2))), meaning
        that props 1 & 2 must be satisfied after prop 0. Props 1 & 2 will be invalidated
        (as above) if they are satisfied earlier than prop 0 in the state sequence.
    """
    return np.array(
        [
            constraint(
                state_sequence=state_sequence,
                proposition_satisfied_at=prop_satisfied_at,
            )
            for constraint in constraints
        ]
    )


def compute_percent_complete(
    proposition_satisfied_at: List[int], constraint_data: np.ndarray
) -> float:
    """
    Computes the percent complete task metric. The ratio of propositions satisfied to
    the number of propositions is returned. Constraints are applied.
    """
    prop_satisfied = np.array(proposition_satisfied_at) != -1

    # case: no propositions
    if prop_satisfied.shape[0] == 0:
        return 0.0

    # case: no constraints
    if constraint_data.shape[0] == 0:
        res = prop_satisfied
    else:
        constraints_satisfied = constraint_data.all(0)
        res = np.logical_and(prop_satisfied, constraints_satisfied)

    return np.mean(res)


def aggregate_measures(stats_episodes: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates measures from a collection of episodes.
    Args:
        stats_episodes: maps episode IDs to a dictionary of measures
    Returns:
        dictionary of averaged measures
    """
    aggregated_stats = {}
    all_ks: Set[str] = set()
    for ep in stats_episodes.values():
        all_ks.update(ep.keys())
    for stat_key in all_ks:
        aggregated_stats[stat_key] = np.mean(
            [v[stat_key] for v in stats_episodes.values() if stat_key in v]
        )
    return aggregated_stats
