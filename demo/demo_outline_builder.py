import asyncio
import json

from loguru import logger

from story_writer.workflow import OutlineBuilder


async def main(max_events: int, premise: str):
    outline_builder = OutlineBuilder(max_events=max_events)
    outline = await outline_builder.build_outline(premise)
    return outline


if __name__ == "__main__":
    max_events = 10
    premise = "A battle-hardened veteran and his loyal comrade travel through a mountain range after a failed campaign. Strange signs suggest someone betrayed them."
    save_path = "outline.json"

    logger.add(sink="logs/{time}.log")

    outline = asyncio.run(main(max_events, premise))
    outline = [event.model_dump() for event in outline.values()]
    with open(save_path, "w") as f:
        json.dump(outline, f, indent=2, ensure_ascii=False)
    print(f"Outline saved to: {save_path}")
