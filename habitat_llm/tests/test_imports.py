#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import importlib

# ----------------------
#   habitat_llm.agent
# ----------------------


def test_import_agent_env_evaluation():
    try:
        importlib.import_module("habitat_llm.agent.env.evaluation")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_agent_env_actions():
    try:
        importlib.import_module("habitat_llm.agent.env.actions")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_agent_env_environment_interface():
    try:
        importlib.import_module("habitat_llm.agent.env.environment_interface")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_agent_env_sensors():
    try:
        importlib.import_module("habitat_llm.agent.env.sensors")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_agent_agent():
    try:
        importlib.import_module("habitat_llm.agent.agent")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
#   habitat_llm.llm
# ----------------------


def test_import_llm_instruct_utils():
    try:
        importlib.import_module("habitat_llm.llm.instruct.utils")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_llm_llama():
    try:
        importlib.import_module("habitat_llm.llm.llama")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_llm_mock():
    try:
        importlib.import_module("habitat_llm.llm.mock")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_llm_openai_chat():
    try:
        importlib.import_module("habitat_llm.llm.openai_chat")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
# habitat_llm.perception
# ----------------------


def test_import_perception_perception():
    try:
        importlib.import_module("habitat_llm.perception.perception")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_perception_perception_sim():
    try:
        importlib.import_module("habitat_llm.perception.perception_sim")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
# habitat_llm.planner
# ----------------------


def test_import_planner_scripted_centralized_planner():
    try:
        importlib.import_module("habitat_llm.planner.scripted_centralized_planner")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_planner_llm_planner():
    try:
        importlib.import_module("habitat_llm.planner.llm_planner")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_planner_planner():
    try:
        importlib.import_module("habitat_llm.planner.planner")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
#   habitat_llm.tools
# ----------------------


def test_import_tools_motor_skills():
    try:
        importlib.import_module("habitat_llm.tools.motor_skills")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_tools_perception():
    try:
        importlib.import_module("habitat_llm.tools.perception")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_tools_tool():
    try:
        importlib.import_module("habitat_llm.tools.tool")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
#   habitat_llm.utils
# ----------------------


def test_import_utils_core():
    try:
        importlib.import_module("habitat_llm.utils.core")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_utils_sim():
    try:
        importlib.import_module("habitat_llm.utils.sim")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


# ----------------------
# habitat_llm.world_model
# ----------------------


def test_import_world_model_entity():
    try:
        importlib.import_module("habitat_llm.world_model.entity")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_world_model_graph():
    try:
        importlib.import_module("habitat_llm.world_model.graph")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")


def test_import_world_model_world_graph():
    try:
        importlib.import_module("habitat_llm.world_model.world_graph")
    except Exception as e:
        raise AssertionError(f"Failed to import module: {e}")
