import asyncio
import json

from loguru import logger

from story_writer.workflow import OutlineBuilder


async def main(k_candidates: int, max_revise: int, max_events: int, premise: str):
    outline_builder = OutlineBuilder(
        k_candidates=k_candidates, max_revise=max_revise, max_events=max_events
    )
    outline = await outline_builder.build_outline(premise)
    return outline


if __name__ == "__main__":
    k_candidates = 5
    max_revise = 5
    max_events = 10
    premise = "A battle-hardened veteran and his loyal comrade travel through a mountain range after a failed campaign. Strange signs suggest someone betrayed them."
    save_path = "outline.json"

    logger.add(sink="logs/{time}.log")

    outline = asyncio.run(
        main(
            k_candidates=k_candidates,
            max_revise=max_revise,
            max_events=max_events,
            premise=premise,
        )
    )
    outline = [event.model_dump() for event in outline.values()]
    with open(save_path, "w") as f:
        json.dump(outline, f, indent=2, ensure_ascii=False)
    logger.info(f"Outline saved to: {save_path}")
