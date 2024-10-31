#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from habitat_llm.llm import instantiate_llm


def test_llm_instantiation() -> None:
    llm = instantiate_llm("mock")
    llm.generate("The meaning of life is")
