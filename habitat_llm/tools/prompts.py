#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import abc


class Prompt(abc.ABC):
    def __init__(self, name, llm_conf) -> None:
        self._name = name
        self._llm_conf = llm_conf
        super().__init__()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

    def is_gt(self):
        return "GT" in self._name


class FindRoomPrompt(Prompt):
    def __init__(self, name, llm_conf) -> None:
        super().__init__(name, llm_conf)
        self._prompt = f"""{self._llm_conf.system_tag}
You are an expert at finding the most suitable room or rooms that satisfy the given query. Remember that you HAVE to guess one or more rooms from the given list. Do not give empty answer!!! Here are a few examples:

START OF EXAMPLES
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 1:
Following rooms are present in the house:
- living_room_1
- living_room_2
- bedroom_1
- bedroom_0
- game_room_0
- unknown_room

Query: the room that can have blankets
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the list, I think that the rooms matching the query are:
- bedroom_0
- bedroom_1
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 2:
Following rooms are present in the house:
- living_room_1
- living_room_2
- bedroom_1
- bedroom_0
- game_room_0
- kitchen_9
- unknown_room

Query: the room that may have cups, mugs, lemon, and snacks
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the list, I think that the rooms matching the query are:
- kitchen_9
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 3:
Following rooms are present in the house:
- living_room_1
- living_room_2
- bedroom_1
- bedroom_0
- game_room_0
- kitchen_0
- storage_0
- unknown_room

Query: rooms which may have soap, shampoo, tooth paste.
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the list, I think that the rooms matching the query are:
- storage_0
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 4:
Following rooms are present in the house:
- living_room_1
- living_room_2
- office_3
- bedroom_1
- bedroom_0
- game_room_0
- kitchen_0
- unknown_room

Query: rooms that may have laptop and spoon
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the list, I think that the rooms matching the query are:
- office_3
- kitchen_0
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 5:
Following rooms are present in the house:
<room_list>

Query: <query>
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the list, I think that the rooms matching the query are:
"""

    def __call__(self, room_list, query):
        if not self._prompt:
            raise ValueError("Prompt not set, use set_env to set the environment")

        if not query:
            raise ValueError("query not set")

        # Create prompt
        prompt = self._prompt.replace("<room_list>", room_list)
        prompt = prompt.replace("<query>", query)
        return prompt


class FRT_CG_Prompt(Prompt):
    def __init__(self, name, llm_conf) -> None:
        super().__init__(name, llm_conf)
        self._prompt = """[INST]<<SYS>>You are an intelligent agent who supports fulfilling
the human's provided instructions in the house. This includes finding receptacles of
interest in the house. You will be provided with a list of receptacles in the house as a
list of <receptacle_id> : <receptacle_name> and a query to find the receptacle of
interest. The names of receptacles may not completely match the queried receptacle, so
you need to find the most semantically similar and relevant receptacle from the list.
Follow the instructions below:
- If there's an unique receptacle that matches the query type then please return it's name.
- If several receptacles match the query type then return all their ids separated by a comma.
- If no receptacle with query type is present in the house, please answer with a message
explaining so.

Be careful in your responses, think step-by-step and make sure to respond correctly!<</SYS>>

Receptacles in the house:
<receptacles>

Query: <target_receptacle>
Answer: After looking at the scene I can tell that the receptacles that match the query
type are:[/INST]"""

    def __call__(self, input_query, receptacles):
        if not self._prompt:
            raise ValueError("Prompt not set, use set_env to set the environment")

        # Create prompt
        prompt = self._prompt.replace("<receptacles>", receptacles)
        prompt = prompt.replace("<target_receptacle>", input_query)
        return prompt


class FRT_FEW_SHOT_Prompt(Prompt):
    def __init__(self, name, llm_conf) -> None:
        super().__init__(name, llm_conf)
        self._prompt = f"""{self._llm_conf.system_tag}
You are an expert at summarizing information about furniture present in a house. Use the following examples to format your answers:

START OF EXAMPLES
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 1:
Following furniture is present:
- couch: sofa_0 in living room, couch_1 in bedroom
- bed: bed_0 in bedroom, bed_1 in bedroom
- chest of drawers: chest_of_drawer_0 in bedroom, chest_of_drawers_2 in bedroom

Query: Couch in living room
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the scene I can tell that the furniture that match the query are:
- sofa_0 in living room
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 2:
Following furniture is present:
- couch: sofa_0 in living room, couch_1 in bedroom
- bed: bed_0 in bedroom, bed_1 in bedroom
- chest of drawers: chest_of_drawer_0 in bedroom, chest_of_drawers_2 in bedroom
- chair: chair_0 in dining room, chair_1 in living room, chair_12 in dining room

Query: Chair in bedroom
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the scene I can tell that the furniture that match the query are:
No chair was found in the bedroom. I was able to find following chairs though:
- chair_0 in dining room, chair_1 in living room, chair_12 in dining room
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 3:
Following furniture is present:
- couch: sofa_0 in living room, couch_1 in bedroom
- bed: bed_0 in bedroom, bed_1 in bedroom
- chest of drawers: chest_of_drawer_0 in bedroom, chest_of_drawers_2 in bedroom
- chair: chair_0 in dining room, chair_1 in living room, chair_12 in dining room

Query: Bed with cushion on it
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the scene I can tell that the furniture that match the query are:
I do not have any information about objects, please use FindObjectTool to query such information.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.system_tag}
END OF EXAMPLES

Use the above examples to generalize as well as you can to the new query below:
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Following furniture is present:
<receptacles>

Query: <query>
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Answer: After looking at the scene I can tell that the furniture that match the query are:
"""

    def __call__(self, query_fur, receptacles, verbose=False):
        prompt = self._prompt.replace("<receptacles>", receptacles)
        prompt = prompt.replace("<query>", query_fur)
        if verbose:
            print(f"[FRT_FEW_SHOT] {prompt}=")
        return prompt


class FOT_FEW_SHOT_Prompt(Prompt):
    def __init__(self, name, llm_conf) -> None:
        super().__init__(name, llm_conf)
        self._prompt = f"""{self._llm_conf.system_tag}
You are an expert at summarizing information about objects present in a house. Use the following examples to format your answers:

START OF EXAMPLES
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 1:
Following objects are present:
- cherry_0 on couch_0 in living_room_0 1.57 meters away
- banana_0 on sofa_0 in bedroom_0 1.88 meters away
- pear_0 on table_9 in living_room_0 1.75 meters away

Query: Objects on the sofa
Answer: After looking at the scene I can tell that the objects that match the query are:
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
- cherry_0 on couch_0 in living_room_0 1.57 meters away
- banana_0 on sofa_0 in bedroom_0 1.88 meters away
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 2:
Following objects are present:
- toy_construction_set_0 on table_15 in living_room_0 2.5 meters away
- toy_bee_1 on bed_0 in bedroom 1/2 meters away
- apple_0 on bathtub_0 in bathroom_0 3.67 meters away

Query: toy vehicle
Answer: After looking at the scene I can tell that the objects that match the query are:
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
No objects with name toy vehicle were found.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 3:
Following objects are present:
- No objects found yet
Query: dumb-bell
Answer: After looking at the scene I can tell that the objects that match the query are:
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
No objects are found yet, please explore the house by navigating to different rooms.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.system_tag}
END OF EXAMPLES

Use the above examples to generalize as well as you can to the new query below:
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
The following objects are present:
<objects>

Query: <query>
Answer: After looking at the scene I can tell that the objects that match the query are:
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
"""

    def __call__(self, query_object, objects, verbose=False):
        prompt = self._prompt.replace("<objects>", objects)
        prompt = prompt.replace("<query>", query_object)
        if verbose:
            print(f"[FOT_FEW_SHOT] {prompt}=")
        return prompt


class FAAT_FEW_SHOT_Prompt(Prompt):
    def __init__(self, name, llm_conf) -> None:
        super().__init__(name, llm_conf)
        self._prompt = f"""{self._llm_conf.system_tag}
You are an expert in summarizing agent activities based on provided tags. Given a sequence of tags representing the states/actions of a agent, summarize the overall activity performed by the agent in a coherent and concise manner. The tags may include actions such as "standing in <room name>", "walking in <room_name>", "picking up <object>", "placing on <location>", "opening <object>", "closing <object>", "waiting" etc.

Your task is to generate a summary sentence that captures the essence of the agent's activity.

START OF EXAMPLES
{self._llm_conf.eot_tag}


{self._llm_conf.user_tag}
Example 1:
Activity Tags: Standing in living_room_1, Walking in living_room_1, Walking in bedroom_1
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Summary: The agent has walked from living room into the bedroom and is still walking.
<Done>
{self._llm_conf.eot_tag}


{self._llm_conf.user_tag}
Example 2:
Activity Tags: Standing in bedroom_1, Walking in bedroom_1, Picking up banana_0, Walking in bedroom_1, Walking in living_room_1, Placing banana_0 on shelf_1
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Summary: The agent moved banana_0 from bedroom to shelf in the living room.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 3:
Activity Tags: Standing in office_0, Walking in office_0, Picking up banana_0, Walking in office_0, Placing banana_0 on shelf_1, Walking in office_0, Picking up bowl_2, Walking in office_0
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Summary: The agent picked up banana_0 from office and moved it to the shelf in office. The agent is currently holding bowl_2 and walking somewhere to place it.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Example 4:
Activity tags: Standing in living_room_1, Walking in living_room_1, Standing in living_room_1, Walking in living_room_1
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Summary: The agent seems to be exploring living room.
<Done>
{self._llm_conf.eot_tag}

{self._llm_conf.system_tag}
END OF EXAMPLES

Ensure that the summary is grammatically correct and logically coherent based on the sequence of actions described by the tags. Use the above examples to generalize as well as you can to the new activity tags below:
{self._llm_conf.eot_tag}

{self._llm_conf.user_tag}
Activity Tags: <activity>
{self._llm_conf.eot_tag}
{self._llm_conf.assistant_tag}
Summary:
"""

    def __call__(self, activity_tags, verbose=False):
        prompt = self._prompt.replace("<activity>", activity_tags)
        if verbose:
            print(prompt)
        return prompt


def get_prompt(prompt_type, llm_conf):
    if prompt_type == "FRT_CG":
        return FRT_CG_Prompt(prompt_type, llm_conf)
    elif prompt_type == "FindRoomPrompt":
        return FindRoomPrompt(prompt_type, llm_conf)
    elif prompt_type == "FOT_FEW_SHOT":
        return FOT_FEW_SHOT_Prompt(prompt_type, llm_conf)
    elif prompt_type == "FAAT_FEW_SHOT":
        return FAAT_FEW_SHOT_Prompt(prompt_type, llm_conf)
    elif prompt_type == "FRT_FEW_SHOT":
        return FRT_FEW_SHOT_Prompt(prompt_type, llm_conf)
    else:
        raise ValueError(f"Prompt type {prompt_type} not recognized")
