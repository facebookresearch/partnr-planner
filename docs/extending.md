# Extending the PARTNR Environment
## How to add new tools and agents?

- To add a new tool, you need to first create two files:
    - ``habitat_llm/tools/<category>/<my_tool>.py``
    - ``habitat_llm/conf/tools/<category>/<my_tool>.yaml``
- The file ``<my_tool>.py`` contains the class defination derived from the ABC called ``"Tool"``
- The file ``<my_tool>.yaml`` contains the configurable parameters.
- Update the ``habitat_llm/tools/<category>/__init__.py`` file to successfully import your file.
- Make an new agent config (i.e. ``habitat_llm/conf/agent/<my_agent>.yaml`` ) or modify an existing agent config.
- Add your tool name to the config file of your chosen agent. Populate the agent config with whichever tools the agent should have access to.
- To use this agent an experiment, override the agent property with the new agent name. I.e. to use the new agent in a single agent experiment, set the agent with `override /agent@evaluation.agents.agent_0.config: <my_agent>` in ``habitat_llm/conf/baselines/single_agent_zero_shot_react_summary.yaml``

## How to add a new prompt to the tools?
If you are adding a new LLM based tool (or changing the prompt of and existing LLM based tool), you can use the built-in prompt manager.

- To add a new prompt, user needs to create a new class in
  `habitat_llm/tools/prompts.py`
- `Prompts` class defines an interface for your prompt template by exposing
  `__call__` API which creates the prompt based on your template and input variables
- Steps for new prompt:
  - Create a new class with your preferred name for that prompt category, store your template in `self._prompt` variable
  - Implement `__call__` which will populate your template based on provided inputs and returns the filled out prompt string
- Example for using a prompt can be seen in
  [find_object_tool.py](../habitat_llm/tools/perception/find_object_tool.py) and see implementation for the prompt `FOT_GT_Prompt` in [`prompts.py`](../habitat_llm/tools/prompts.py)

## How to add a custom action to the agent

Actions in habitat are low level constructs which alter the state of the environment. Every action consists of an action space which parameterizes the action. For example, the action space of BaseVelAction in habitat-lab consists of linear and angular velocities. Every action has a corresponding skill defined in habitat-llm which takes in the observations from the agent and populate the corresponding action space. The actions defined in habitat-lab may not be sufficient for all purposes and one may want to add custom actions for their own task. This section describes how to add custom actions inside habitat-llm (without modifying habitat-lab).

Adding new action (Update ``habitat_llm/agent/env/actions.py``):
- Step 1: Add your custom action to ``HabitatSimActions``.
- Step 2: Define your action. Each action class should contain a constructor, a reset method, an action_space property and a step method. The step method should change the state of the environment.
- Step 3: Declare data class for your action.
- Step 4: Register your action

Notes:
- The elements of the action_space from all of the actions are eventually alphabetically sorted into a list, so make sure that no two element of the action_space across all actions have same name. (Otherwise they will get overwritten)
- Make sure that the skill corresponding to this action populates the action space correctly.
- Make sure that the clipping values for your action space is correct.
