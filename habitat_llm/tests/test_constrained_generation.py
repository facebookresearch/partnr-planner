#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import gc

import pytest
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.tests.test_planner import get_config, setup_env
from habitat_llm.utils import setup_config
from habitat_llm.utils.grammar import OBJECT

DATASET_OVERRIDES = [
    "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",  # We test with a specific dataset
    "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
    "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
    "habitat.environment.iterator_options.shuffle=False",
    "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/humanoids/female_0/female_0.urdf",  # We change the config to human 0 since only human 0 in the CI testing dataset
    "habitat.simulator.agents.agent_1.motion_data_path=data/humanoids/female_0/female_0_motion_data_smplx.pkl",  # We change the config to human 0 since only human 0 in the CI testing dataset
]


def test_grammar_generation():
    default_agent_uid = 0
    config = get_config(
        "examples/planner_multi_agent_demo_config.yaml",
        overrides=[
            "evaluation=decentralized_evaluation_runner_multi_agent",
            "planner@evaluation.agents.agent_0.planner=llm_planner",
            "agent@evaluation.agents.agent_0.config=oracle_rearrange_object_states_agent",
            "llm@evaluation.agents.agent_0.planner.plan_config.llm=mock",
            "planner@evaluation.agents.agent_1.planner=llm_planner",
            "llm@evaluation.agents.agent_1.planner.plan_config.llm=mock",
        ]
        + DATASET_OVERRIDES,
    )

    if not CollaborationDatasetV0.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")

    config = setup_config(config, 0)
    env_interface = setup_env(config)
    config.evaluation.agents.agent_0.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }
    config.evaluation.agents.agent_1.planner.plan_config.llm.llm = {
        "_target_": "unittest.mock.Mock"
    }
    # Set the planner
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    planner = eval_runner.planner[0]
    grammar_str = planner.build_tool_grammar(
        env_interface.world_graph[default_agent_uid]
    )
    parsed_grammar = parse_ebnf(grammar_str)
    start_rule_id = parsed_grammar.symbol_table["tool_call"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    accept_cases = [
        "Navigate[living_room_1]",
        "Pick[cushion_0]",
        "Place[cushion_0,on,table_32,none,none]",
        "Open[cabinet_39]",
        "Close[cabinet_39]",
        "Explore[bathroom_1]",
        "FindObjectTool[something edible]",
        "FindObjectTool[phone, watch, credit_card]",
        "FindReceptacleTool[sink]",
        "FindRoomTool[kitchen]",
        "FindAgentActionTool[]",
        "Wait[]",
        "Rearrange[toy_construction_set_1,on,table_32,none,none]",
        "Rearrange[toy_construction_set_1,on,table_32,next_to,cushion_0]",
        "Rearrange[toy_construction_set_1, on, table_32, next_to,   cushion_0]",
        "Place[toy_construction_set_1, within,table_32,next_to, cushion_0]",
        # allows furniture as reference
        "Place[toy_construction_set_1, within,table_32,next_to, table_32]",
        "Rearrange[toy_construction_set_1, within,table_32,next_to, table_32]",
        # allows objects and furniture to be cleaned and powered on
        "Clean[toy_construction_set_1]",
        "Clean[table_32]",
        "PowerOn[toy_construction_set_1]",
        "PowerOn[table_32]",
    ]
    print(grammar_str)
    for test_case in accept_cases:
        result = recognizer._accept_string(test_case)
        assert result, f"Test case failed: {test_case}"

    reject_cases = [
        # invalid room
        "Navigate[bad_room]",
        # invalid relation
        "Place[cushion_0,on_top,table_32,none,none]",
        # invalid object
        "Open[cushion_1]",
        # invalid tool
        "FindCategoryTool[something edible]",
        # too many arguments
        "Wait[argument]",
        # constraint but not reference
        "Rearrange[toy_construction_set_1,on,table_32,next_to,none]",
        "Place[toy_construction_set_1,on,table_32,next_to,none]",
    ]
    for test_case in reject_cases:
        assert not recognizer._accept_string(
            test_case
        ), f"Test case failed: {test_case}"

    # test the grammar when there are no objects
    grammar_str = planner.build_tool_grammar(
        env_interface.perception.get_graph_without_objects()
    )
    # the object rule and object tools should not appear
    assert OBJECT not in grammar_str
    for tool in ["Rearrange", "Pick", "Place"]:
        assert tool not in grammar_str
    parsed_grammar = parse_ebnf(grammar_str)
    # non object skills still accept
    accept_cases = [
        "Navigate[living_room_1]",
        "Open[cabinet_39]",
        "Close[cabinet_39]",
        "Explore[bathroom_1]",
        "FindObjectTool[something edible]",
        "FindReceptacleTool[sink]",
        "FindRoomTool[kitchen]",
        "FindAgentActionTool[]",
        "Wait[]",
    ]
    for test_case in accept_cases:
        result = recognizer._accept_string(test_case)
        assert result, f"Test case failed: {test_case}"
    reject_cases = [
        # there are now no objects
        "Pick[cushion_1]",
        "Pick[]",
        "Rearrange[,on,cabinet_39,none,]",
        "Navigate[cushion_1]",
    ]
    for test_case in reject_cases:
        assert not recognizer._accept_string(
            test_case
        ), f"Test case failed: {test_case}"

    # test the full grammar with thought and assignment
    grammar_str = planner.build_response_grammar(
        env_interface.world_graph[default_agent_uid]
    )
    parsed_grammar = parse_ebnf(grammar_str)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    accept_cases = [
        "I need to pick up the cushion, watch and credit card.\nAgent_0_Action: Pick[cushion_0]\nAssigned!",
        "I've moved all objects!\nFinal Thought: Exit!",
    ]
    for test_case in accept_cases:
        result = recognizer._accept_string(test_case)
        assert result, f"Test case failed: {test_case}"

    # For the response grammar we need the agent to be assigned
    reject_cases = [
        "Pick[cushion_0]",
        "Open[cabinet_39]",
        # cannot only have the thought
        "I will pick up the cushion.",
        "Since the other agent is still seems to be moving the credit card I will proceed to the next parts of the task\nAgent_0_Action:  Rearrange[credit],on,bed_21,next_to,drink_]]\nAssigned!",
    ]
    for test_case in reject_cases:
        result = recognizer._accept_string(test_case)
        assert not result, f"Test case failed: {test_case}"

    # Destroy envs
    env_interface.env.close()
    del env_interface
    gc.collect()
