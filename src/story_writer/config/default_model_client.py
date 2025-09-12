import os

from autogen_ext.models.openai import OpenAIChatCompletionClient

DEFAULT_MODEL_CLIENT = OpenAIChatCompletionClient(
    model="qwen3-235b-a22b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": False,
    },
)
