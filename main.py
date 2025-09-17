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
    premise = """西比拉系统（Sibyl System）被宣传为一项划时代的伟大发明，它带来了一个近乎零犯罪的“理想社会”。系统通过遍布城市的传感器（穹顶扫描仪）持续地、无差别地扫描所有市民的心理状态。
然而西比拉系统并非纯粹的AI，其核心是由数百名免罪体质者（Latent Criminal） 的大脑组成的脑集合体。这些大脑被取出后，通过先进的神经接口连接在一起，共同进行演算和判断。
一位坚信西比拉系统绝对公正的新人执行官，在奉命处理一起看似普通的“潜在犯”清除任务时，却发现目标的“犯罪系数”清澈如水——而系统依然下达了灭绝指令——这唯一的、致命的“错误”迫使她必须在服从毕生信仰的法则与听从自己良知的呐喊之间做出选择，并由此揭开一桩足以动摇系统根基的可怕真相。
调查将她引向一个惊人的发现：这位老人并非普通人，他曾是西比拉系统早期开发团队的一名低级程序员，掌握着一段关于系统“原始代码”中某个被刻意隐藏的“后门”或“缺陷”的记忆。系统并非“出错”，而是在进行预防性的自我净化——清除任何可能威胁其绝对完美表象的知情人，哪怕他的心理状态毫无威胁。
她站在系统的十字路口：是销毁证据，维持系统的“神圣”，回到自己“螺丝钉”的位置，继续过着虚假但安全的生活；还是冒着被即刻处决的风险，将证据交给一个她唯一可能信任的人（比如一个像常守朱那样有怀疑精神的监视官），哪怕这会引发一场她无法预料的风暴？

**请使用中文编写这个故事**
"""
    output_dir = './output'

    asyncio.run(main(premise, output_dir))
