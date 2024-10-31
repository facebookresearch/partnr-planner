#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from pathlib import Path
from typing import Dict, List

import filelock
from rlm.llm import RemoteLanguageModel


class RemotePoolLanguageModel(RemoteLanguageModel):
    """
    Allows multiple processes to access a pool of llms. Works by keeping a json of
    LLM addresses and the number of agents that are on the queue for each address
    when called, assigns an agent with the llm with the shortest queue.
    In particular let's say that the serve_model function initialized
    4 nodes, with the files inside out_dir/{hostname_i}:{port_i}.
    This LanguageModel will create a JSON of the form
    {
      "{hostname_i}:{port_i}": ct_i
    }
    for i 1..4. Where ct_i will measure how many processes are calling
    an llm in that host. This call keeps track of this JSON and whenever a new
    process wants to call an LLM, it assigns the port with lower ct_i
    """

    def __init__(self, serverdir: str) -> None:
        self.addresses = [
            f"http://{p.name}" for p in Path(f"{serverdir}/server_list/").glob("*")
        ]
        self.lockfile = f"{serverdir}/lock.json"
        self.lockfilelock = f"{serverdir}/lock.json.lock"

    def get_address_with_shortest_queue(self):
        """
        Get the address of the LLM that has less processes waiting,
        and increase the number of processes waiting for that LLM
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        t_address = None
        with filelock.FileLock(lockfilelock):
            with open(lockfile, "r") as f:
                content = json.load(f)
            # Sort addresses by number of calls, and return the first one
            # this is to get the address with lowest number of calls.
            t_address = sorted(content.items(), key=lambda kv: kv[1])[0][0]
            content[t_address] += 1
            with open(lockfile, "w+") as f:
                f.write(json.dumps(content))
        return t_address

    def free_address(self, address):
        """
        Whenever the LLM completes the call reduce the number of processes
        pointing to that LLM
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        with filelock.FileLock(lockfilelock):
            with open(lockfile, "r") as f:
                content = json.load(f)
            content[address] -= 1
            with open(lockfile, "w+") as f:
                f.write(json.dumps(content))

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling: bool = False,
        generation_args: Dict = None,
    ) -> List[Dict]:
        """
        Generate an LLM output. Find the most free LLM, assign the process to that LLM
        when done, free that LLM.
        """
        lockfile = self.lockfile
        lockfilelock = self.lockfilelock
        # If the json file does not exist, start it here
        with filelock.FileLock(lockfilelock):
            if not os.path.isfile(lockfile):
                with open(lockfile, "w+") as f:
                    lock_contents = {addr: 0 for addr in self.addresses}
                    f.write(json.dumps(lock_contents))
        # Get the llm with fewer calls
        self.address = self.get_address_with_shortest_queue()
        # Run the llm with te given prompt
        result = super().batch_generate(
            prompts,
            max_new_tokens,
            temperature,
            sampling,
            generation_args=generation_args,
        )
        # Free that llm in the json file.
        self.free_address(self.address)
        return result
