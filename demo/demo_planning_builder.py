import asyncio
import json

from loguru import logger

from story_writer.schemas import Event
from story_writer.workflow import PlanningBuilder
from story_writer.workflow.outline_builder import EventGraph


async def main(premise: str, event_graph: EventGraph):
    planning_builder = PlanningBuilder()
    story_plan = await planning_builder.build_plan(premise, event_graph)
    return story_plan


if __name__ == '__main__':
    premise = """A battle-hardened veteran and his loyal comrade travel
 through a mountain range after a failed campaign. Strange signs suggest someone betrayed them."""
    event_graph_file = 'data/outline.json'
    save_path = 'data/story_plan.json'

    logger.add(sink='logs/{time}.log')

    with open(event_graph_file, 'r') as f:
        event_graph_json = json.load(f)

    event_graph: EventGraph = {}
    for event_dict in event_graph_json:
        event = Event.model_validate(event_dict)
        event_graph[event.event_id] = event

    story_plan = asyncio.run(main(premise, event_graph))
    story_dict = story_plan.model_dump()
    with open(save_path, 'w') as f:
        json.dump(story_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Story plan saved to {save_path}")
