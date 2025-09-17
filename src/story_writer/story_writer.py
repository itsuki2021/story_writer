import json
import os
import os.path as osp
import uuid

from autogen_core.models import ChatCompletionClient
from loguru import logger

from story_writer.workflow import OutlineBuilder, PlanningBuilder, WritingBuilder


class StoryWriter:

    def __init__(
        self,
        model_client: ChatCompletionClient,
        k_candidates: int = 3,
        max_val: int = 2,
        max_events: int = 30,
    ) -> None:
        """Instantiate StoryWriter

        Args:
            model_client (ChatCompletionClient): Model client.
            k_candidates (int, optional): Number of candidate events to generate in one iteration. defaults to 3
            max_val (int, optional): Maximum number of times to validate an event in one iteration. defaults to 2
            max_events (int, optional): Maximum number of events to generate. defaults to 30
        """
        self.outline_builder = OutlineBuilder(
            model_client=model_client,
            k_candidates=k_candidates,
            max_val=max_val,
            max_events=max_events,
        )
        self.planing_builder = PlanningBuilder(model_client=model_client)
        self.writing_builder = WritingBuilder(model_client=model_client)
        logger.add(sink='logs/{time}.log')

    async def write(self, premise: str, output_dir) -> None:
        """Write a story

        Args:
            premise (str): The premise of the story
            output_dir (str): The directory to write the story to
        """
        story_root = osp.join(output_dir, str(uuid.uuid4()))
        os.makedirs(story_root)

        # 1. Generate event graph
        event_graph = await self.outline_builder.build_outline(premise)
        outline_file = osp.join(story_root, 'outline.json')
        with open(outline_file, 'w') as f:
            json.dump(event_graph.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Outline saved to: {outline_file}")

        # 2. Generate sub-events and weaving it into chapters
        story_plan = await self.planing_builder.build_plan(premise=premise, event_graph=event_graph)
        story_plan_file = osp.join(story_root, 'story_plan.json')
        with open(story_plan_file, 'w') as f:
            json.dump(story_plan.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Story plan saved to: {story_plan_file}")

        # 3. Writing chapters
        chapter_text_list = await self.writing_builder.build_story(story_plan=story_plan)
        chapter_file = osp.join(story_root, 'chapters.json')
        with open(chapter_file, 'w') as f:
            json.dump(
                [ct.model_dump() for ct in chapter_text_list],
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Chapters saved to: {chapter_file}")
