# LLMs
This folder contains the classes to run different language models in the PARTNR benchmark. Note that these language models should not be run in isolation, instead these classes will be initialized when you run different agent baselines, as specified in the [README.md](../../README.md). Our current code supports the following language models.

- Llama 2.X and 3.X, with both the base and instruction tuned model.
- OpenAI Chat models, via the [AzureOpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) interface.

## Configuring the LLMs

#### Llama
To run the Llama based model, you will need to set the model path. You can do this by manually editing the
files `llama.yaml`, `llama_non_instruct.yaml`, changing the field:

```
generation_params:
    engine: "{PATH to Llama}/Meta-Llama-3.1-70B
```

to point to your model path, or model name if you use the Hugging Face Hub interface.

You can also set it via a command line argument, using `plan_config.llm.generation_params.engine`, as specified in [README.md](../../README.md).

You can then test the model using:

```python
from habitat_llm.llm.llama import Llama
from omegaconf import OmegaConf
model_config_path = "habitat_llm/conf/llm/llama_non_instruct.yaml"
config = OmegaConf.load(model_config_path)
config.generation_params.engine = "Path to your model here"
model = Llama(config)
model.generate("The answer is 42 to", max_length=10)
```

#### OpenAI
To run the OpenAI models, you will need to set up your API keys and endpoints:

```python
import os
os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_ENDPOINT"]

from habitat_llm.llm.OpenAIChat import OpenAIChat
from omegaconf import OmegaConf
model_config_path = "habitat_llm/conf/llm/openai_chat.yaml"
config = OmegaConf.load(model_config_path)
model = OpenAIChat(config)
model.generate("The answer is 42 to", max_length=10)

```
