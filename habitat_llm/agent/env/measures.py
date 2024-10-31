# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import habitat
import numpy as np
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.sims.habitat_simulator import sim_utilities

from habitat_llm.agent.env.dataset import CollaborationEpisode
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationProposition,
    apply_constraint_satisfaction,
    compute_percent_complete,
    determine_propositions_to_evaluate,
    unroll_propositions_with_number,
)
from habitat_llm.agent.env.evaluation.failure_explanations import (
    derive_evaluation_explanation,
)
from habitat_llm.agent.env.evaluation.predicate_wrappers import (
    PropositionResult,
    SimBasedPredicates,
)
from habitat_llm.sims.metadata_interface import (
    MetadataInterface,
    get_metadata_dict_from_config,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


############################################################
# Define custom measures
############################################################


@registry.register_measure
class AutoEvalPropositionTracker(Measure):
    """
    At every time step, the current proposition state is appended to a state sequence.
    """

    cls_uuid: str = "auto_eval_proposition_tracker"

    _sim: Simulator
    _ao_link_map: Dict[int, int]
    _config: "DictConfig"
    _propositions: List[EvaluationProposition]
    _state_sequence: List[List[PropositionResult]]
    _proposition_satisfied_at: List[int]
    _prop_default_tracking = {
        "object_handles",
        "receptacle_handles",
        "entity_handles_a",
        "entity_handles_b",
        "room_ids",
        "*args",
    }

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AutoEvalPropositionTracker.cls_uuid

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def reset_metric(
        self, *args, episode: CollaborationEpisode, task: EmbodiedTask, **kwargs
    ):
        # precompute ao_link_map for fast predicates
        self._ao_link_map = sim_utilities.get_ao_link_id_map(self._sim)

        self._propositions = copy.deepcopy(episode.evaluation_propositions)
        self._dependencies = copy.deepcopy(episode.evaluation_proposition_dependencies)
        self._constraints = copy.deepcopy(episode.evaluation_constraints)
        self._state_sequence = []

        (
            self._propositions,
            self._dependencies,
            self._constraints,
        ) = unroll_propositions_with_number(
            self._propositions, self._dependencies, self._constraints
        )
        self._proposition_satisfied_at = [-1 for _ in range(len(self._propositions))]

        self._metric: Dict[str, List[Any]] = {
            "propositions": self._propositions,
            "dependencies": self._dependencies,
            "constraints": self._constraints,
            "state_sequence": self._state_sequence,
            "proposition_satisfied_at": self._proposition_satisfied_at,
        }
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, **kwargs: Any):
        """Update both `state_sequence` and `proposition_satisfied_at`."""
        object_states_dict = SimBasedPredicates.get_state_snapshot_if_none(self._sim)
        propositions_to_evaluate = determine_propositions_to_evaluate(
            self._metric["state_sequence"],
            self._propositions,
            self._dependencies,
        )

        # state defaults
        state = []
        for prop in self._propositions:
            info = {k: "" for k in prop.args if k in self._prop_default_tracking}
            state.append(PropositionResult(False, info))

        for idx in propositions_to_evaluate:
            prop = self._propositions[idx]
            sim_predicate_fn = getattr(SimBasedPredicates, prop.function_name)
            if "*args" in prop.args:
                # supports a variable number of arguments
                prop_result: PropositionResult = sim_predicate_fn(
                    *prop.args["*args"],
                    sim=self._sim,
                    ao_link_map=self._ao_link_map,
                    object_states_dict=object_states_dict,
                    **{k: v for k, v in prop.args.items() if k != "*args"},
                )
            else:
                prop_result = sim_predicate_fn(
                    sim=self._sim,
                    ao_link_map=self._ao_link_map,
                    object_states_dict=object_states_dict,
                    **prop.args,
                )
            state[idx] = prop_result

            already_satisfied = self._metric["proposition_satisfied_at"][idx] != -1
            if not already_satisfied and prop_result.is_satisfied:
                t_satisfied = len(self._metric["state_sequence"])
                self._metric["proposition_satisfied_at"][idx] = t_satisfied

        self._metric["state_sequence"].append(state)


@registry.register_measure
class TaskConstraintValidation(Measure):
    cls_uuid: str = "task_constraint_validation"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return TaskConstraintValidation.cls_uuid

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def reset_metric(
        self, *args, episode: CollaborationEpisode, task: EmbodiedTask, **kwargs
    ):
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [AutoEvalPropositionTracker.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        """
        Applies each task constraint independently. The resulting metric is an ndarray[i,j]
        where row i is a constraint, col j is a proposition, and ndarray[i,j] is False if
        the application of constraint i invalidates proposition j.
        """
        prop_data = task.measurements.measures[
            AutoEvalPropositionTracker.cls_uuid
        ].get_metric()
        self._metric = apply_constraint_satisfaction(
            prop_data["constraints"],
            prop_data["state_sequence"],
            prop_data["proposition_satisfied_at"],
        )


@registry.register_measure
class TaskPercentComplete(Measure):
    cls_uuid: str = "task_percent_complete"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return TaskPercentComplete.cls_uuid

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        self._config = config
        super().__init__(*args, config=config, **kwargs)

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [AutoEvalPropositionTracker.cls_uuid]
        )
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [TaskConstraintValidation.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        """
        The ratio of propositions satisfied to total number of propositions.
        """
        prop_data = task.measurements.measures[
            AutoEvalPropositionTracker.cls_uuid
        ].get_metric()
        constraint_data: np.ndarray = task.measurements.measures[
            TaskConstraintValidation.cls_uuid
        ].get_metric()
        self._metric = compute_percent_complete(
            prop_data["proposition_satisfied_at"], constraint_data
        )


@registry.register_measure
class TaskStateSuccess(Measure):
    """
    True if all propositions and constraints of the collaboration task are satisfied.
    """

    cls_uuid: str = "task_state_success"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return TaskStateSuccess.cls_uuid

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def reset_metric(self, *args, task: EmbodiedTask, **kwargs):
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [TaskPercentComplete.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        percent_complete = task.measurements.measures[
            TaskPercentComplete.cls_uuid
        ].get_metric()
        self._metric = float(percent_complete == 1.0)


@registry.register_measure
class TaskEvaluationLog(Measure):
    """
    Returns a log of the propositions and state sequence for further analysis and
    potential agent feedback.
    """

    cls_uuid: str = "task_evaluation_log"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return TaskEvaluationLog.cls_uuid

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def reset_metric(
        self, *args, episode: CollaborationEpisode, task: EmbodiedTask, **kwargs
    ):
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [AutoEvalPropositionTracker.cls_uuid]
        )
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [TaskConstraintValidation.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        """Aggregates evaluation data for detailed logging and failure analysis."""

        prop_data = task.measurements.measures[
            AutoEvalPropositionTracker.cls_uuid
        ].get_metric()
        constraint_data: np.ndarray = task.measurements.measures[
            TaskConstraintValidation.cls_uuid
        ].get_metric()
        self._metric = {
            "propositions": prop_data["propositions"],
            "dependencies": prop_data["dependencies"],
            "constraints": prop_data["constraints"],
            "proposition_satisfied_at": prop_data["proposition_satisfied_at"],
            "constraint_satisfaction": constraint_data,
            "state_sequence": prop_data["state_sequence"],
        }


@registry.register_measure
class TaskExplanation(Measure):
    cls_uuid: str = "task_explanation"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return TaskExplanation.cls_uuid

    def __init__(self, *args: Any, config: "DictConfig", **kwargs: Any):
        self.metadata_interface = MetadataInterface(
            get_metadata_dict_from_config(config)
        )
        super().__init__(*args, **kwargs)

    def reset_metric(
        self, *args, episode: CollaborationEpisode, task: EmbodiedTask, **kwargs
    ):
        task.measurements.check_measure_dependencies(
            self.cls_uuid, [TaskEvaluationLog.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self,
        *args: Any,
        episode: CollaborationEpisode,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        log = task.measurements.measures[TaskEvaluationLog.cls_uuid].get_metric()
        self._metric = derive_evaluation_explanation(
            log["propositions"],
            log["constraints"],
            log["proposition_satisfied_at"],
            log["constraint_satisfaction"],
            self.metadata_interface,
        )


############################################################
# Define measure configs
############################################################


@dataclass
class AutoEvalPropositionTrackerConfig(MeasurementConfig):
    """
    A service measurement. Maintains the binary state of a list of propositions.
    Keeps all timesteps for use by downstream evaluation metrics.
    """

    type: str = "AutoEvalPropositionTracker"
    name: str = "auto_eval_proposition_tracker"


@dataclass
class TaskConstraintValidationConfig(MeasurementConfig):
    """
    Applies evaluation constraints to the propositional state sequence. Depends on
    AutoEvalPropositionTracker.
    """

    type: str = "TaskConstraintValidation"
    name: str = "task_constraint_validation"


@dataclass
class TaskPercentCompleteConfig(MeasurementConfig):
    """
    Measurement for the percentage of propositions satisfied in a task.
    Depends on AutoEvalPropositionTracker and TaskConstraintValidation.
    """

    type: str = "TaskPercentComplete"
    name: str = "task_percent_complete"


@dataclass
class TaskStateSuccessConfig(MeasurementConfig):
    """
    Measurement for binary success of a task. Depends on TaskPercentComplete.
    """

    type: str = "TaskStateSuccess"
    name: str = "task_state_success"


@dataclass
class TaskEvaluationLogConfig(MeasurementConfig):
    """
    Measurement for insight into which propositions and constraints were satisfied.
    Depends on AutoEvalPropositionTracker and TaskConstraintValidation.
    """

    type: str = "TaskEvaluationLog"
    name: str = "task_evaluation_log"


@dataclass
class TaskExplanationConfig(MeasurementConfig):
    """
    Measurement for parsing the evaluation log into a natural language explanation
    of task failure.
    """

    type: str = "TaskExplanation"
    name: str = "task_explanation"
    metadata: Any = None


############################################################
# Register measures
############################################################

ALL_MEASURES = [
    AutoEvalPropositionTrackerConfig,
    TaskConstraintValidationConfig,
    TaskPercentCompleteConfig,
    TaskStateSuccessConfig,
    TaskEvaluationLogConfig,
    TaskExplanationConfig,
]


def register_measures(conf):
    with habitat.config.read_write(conf):
        for measure_config in ALL_MEASURES:
            MeasureConfig = measure_config()
            if measure_config == TaskExplanationConfig:
                MeasureConfig.metadata = conf.habitat.dataset.metadata
            conf.habitat.task.measurements[MeasureConfig.name] = MeasureConfig
