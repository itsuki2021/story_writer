import asyncio
import os

from autogen_ext.models.openai import OpenAIChatCompletionClient

from story_writer import StoryWriter


async def main(premise: str, output_dir: str) -> None:
    model_client = OpenAIChatCompletionClient(
        model='qwen3-next-80b-a3b-instruct',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ['DASHSCOPE_API_KEY'],
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
            'family': 'unknown',
            'structured_output': False,
        },
        temperature=0.7,
    )

    story_writer = StoryWriter(model_client=model_client)
    await story_writer.write(premise=premise, output_dir=output_dir)


if __name__ == '__main__':
    premise = 'A battle-hardened veteran and his loyal comrade travel through a mountain range after a failed campaign. Strange signs suggest someone betrayed them.'
    output_dir = './output'

    asyncio.run(main(premise, output_dir))
