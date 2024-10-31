## Information about Instruct LLM

### Core Concepts

The implemented system revolves around the idea of putting the Language Model (LLM) in a tight loop of action-observation with the environment. This methodology allows the LLM to interact more efficiently with the environment, ensuring seamless processing of tasks and smooth communication with users.

The crucial elements in this setup are:

1. **Stopword**: This is a token that signifies when the LLM should stop generating text and wait for a response from the environment. The stopword is the last token that the LLM predicts when it calls an action in the environment.
2. **PostObservation**: This token is used to separate the observation output (coming from the environment) from the next generation phase. The PostObservation is the first token that the LLM predicts after receiving an observation from the environment.
3. **EndExpression**: This token indicates the end of the interaction.

### Implementations

A new way to add an llm agent in a loop with the environment can be done by adding it to `conf/agent/instruct`. The basic structure is as follows:

```
prompt: |-
        <Here we explain the task and how to solve it>


# This is crucial, it's when the LLM stops generating text and waits for the
# environment to respond
stopword       : "Observation:"

# Added manually, separates the observation output from the next generation phase
post_observation : "\nThought:"

# End of the interaction.
end_expression : "Final Answer:"

# The parser is a function that maps LLM output to a tuple of (action, action_input)
action_parser:
  _target_     : habitat_llm.llm.instruct.utils.zero_shot_prompt_action_parser

  # It's a function so we do partial initialization
  _partial_    : true
```
