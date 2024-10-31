#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree


import os

DEFAULT_PORT = 8738
DEFAULT_PORT_RANGE = 127
DEFAULT_MAIN_ADDR = "127.0.0.1"

IGNORE_INDEX = -100


def init_distrib():
    os.environ["LOCAL_RANK"] = str(os.environ.get("SLURM_LOCALID", 0))
    os.environ["RANK"] = os.environ.get("SLURM_PROCID")
    os.environ["WORLD_SIZE"] = str(os.environ.get("SLURM_NTASKS", 1))
    SLURM_JOBID = os.environ.get("SLURM_JOBID")
    main_port = int(os.environ.get("MAIN_PORT", DEFAULT_PORT))
    if SLURM_JOBID is not None:
        main_port += int(SLURM_JOBID) % int(
            os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["MASTER_ADDR"] = os.environ.get("MAIN_ADDR", DEFAULT_MAIN_ADDR)
