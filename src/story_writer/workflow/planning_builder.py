from typing import Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from json_repair import repair_json
from loguru import logger

from story_writer.config import DEFAULT_MODEL_CLIENT
from story_writer.prompts import (SUBTASKER_SYSTEM_PROMPT, SUBTASKER_USER_PROMPT, WEAVER_SYSTEM_PROMPT,
                                  WEAVER_USER_PROMPT)
from story_writer.schemas import Event, StoryPlan, SubEvent
from story_writer.schemas.planning_schemas import Chapter
from story_writer.workflow.outline_builder import EventGraph


class PlanningBuilder:

    def __init__(
        self,
        model_client: ChatCompletionClient = DEFAULT_MODEL_CLIENT,
    ):
        """
        Initializes the PlanningBuilder.

        Args:
            model_client (ChatCompletionClient, optional): The client for interacting with the language model.
        """
        self.subtasker_agent = AssistantAgent(
            name='subtasker_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=SUBTASKER_SYSTEM_PROMPT,
        )
        self.weaver_agent = AssistantAgent(
            name='weaver_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=WEAVER_SYSTEM_PROMPT,
        )

    def _parse_sub_event(self, subevent_text: str) -> List[SubEvent]:
        """Parses sub-event text into SubEvent objects.

        Args:
            subevent_text (str): The text output from the SubTasker agent.

        Returns:
            List[SubEvent]: A list of SubEvent objects.
        """
        subevent_obj_list = repair_json(subevent_text, ensure_ascii=False, return_objects=True)
        if not isinstance(subevent_obj_list, list):
            subevent_obj_list = [subevent_obj_list]
        sub_events = []
        for subevent_obj in subevent_obj_list:
            try:
                subevent = SubEvent.model_validate(subevent_obj)
                sub_events.append(subevent)
            except Exception as e:
                logger.warning(f"Failed to parse subevent: {e}")
        return sub_events

    def _parse_chapter(self, chapter_text: str) -> List[Chapter]:
        """Parses chapter text into Chapter objects.

        Args:
            chapter_text (str): The text output from the Weaver agent.

        Returns:
            List[Chapter]: A list of Chapter objects.
        """
        chapter_obj_list = repair_json(chapter_text, ensure_ascii=False, return_objects=True)
        if not isinstance(chapter_obj_list, list):
            chapter_obj_list = [chapter_obj_list]
        chapters = []
        for chapter_obj in chapter_obj_list:
            try:
                chapter = Chapter.model_validate(chapter_obj)
                chapters.append(chapter)
            except Exception as e:
                logger.warning(f"Failed to parse chapter: {e}")
        return chapters

    async def _run_subtasker(self, premise: str, event: Event) -> List[SubEvent]:
        """Invokes the SubTasker agent to decompose an event into sub-events.

        Args:
            premise (str): The story premise.
            event (Event): The event to decompose.

        Returns:
            List[SubEvent]: A list of SubEvent objects.
        """
        subtasker_task = SUBTASKER_USER_PROMPT.format(premise=premise, parent_event=event.model_dump_json())
        await self.subtasker_agent.on_reset(CancellationToken())  # clear model context
        subtasker_task_result = await Console(self.subtasker_agent.run_stream(task=subtasker_task))
        assert isinstance(subtasker_task_result.messages[-1], TextMessage)
        sub_events = self._parse_sub_event(subtasker_task_result.messages[-1].content)
        return sub_events

    async def _run_weaver(
        self,
        premise: str,
        event_graph: EventGraph,
        all_sub_events: Dict[str, SubEvent],
    ) -> List[Chapter]:
        """Invokes the Weaver agent to weave sub-events into a story plan.

        Args:
            premise (str): The story premise.
            event_graph (Dict[str, Event]): The event graph from the OutlineBuilder.
            all_sub_events (Dict[str, SubEvent]): All sub-events from the SubTasker.

        Returns:
            List[Chapter]: A list of Chapter objects.
        """
        weaver_task = WEAVER_USER_PROMPT.format(
            premise=premise,
            event_graph=[e.model_dump() for e in event_graph.values()],
            sub_events=[se.model_dump() for se in all_sub_events.values()],
        )
        await self.weaver_agent.on_reset(CancellationToken())  # clear model context
        weaver_task_result = await Console(self.weaver_agent.run_stream(task=weaver_task))
        assert isinstance(weaver_task_result.messages[-1], TextMessage)
        chapters = self._parse_chapter(weaver_task_result.messages[-1].content)
        return chapters

    async def build_plan(self, premise: str, event_graph: EventGraph) -> StoryPlan:
        """Orchestrates the full planning process.

        Args:
            premise (str): The story premise.
            event_graph (Dict[str, Event]): The event graph from the OutlineBuilder.

        Returns:
            StoryPlan: The complete story plan for the Writing Agents.
        """
        # Step 1: Decompose all events into sub-events using SubTasker
        logger.info('Starting Step 1: Decomposing events into sub-events...')
        all_sub_events: Dict[str, SubEvent] = {}
        for event_id, event in event_graph.items():
            logger.info(f"Processing event {event_id}...")
            sub_events_for_event = await self._run_subtasker(premise, event)
            for sub_event in sub_events_for_event:
                all_sub_events[sub_event.sub_event_id] = sub_event

        logger.info(f"Finished Step 1: Generated {len(all_sub_events)} total sub-events.")

        # Step 2: Generate only the chapter structure using the Weaver
        logger.info('Starting Step 2: Weaving sub-events into chapters...')
        chapters = await self._run_weaver(premise, event_graph, all_sub_events)
        logger.info(f"Finished Step 2: Generated {len(chapters)} chapters.")

        # Step 3: Assemble the final StoryPlan object
        logger.info('Starting Step 3: Assembling the final StoryPlan object...')
        final_story_plan = StoryPlan(event_graph=event_graph, sub_events=all_sub_events, chapters=chapters)
        logger.info('Finished Step 3: Final story plan assembled.')

        return final_story_plan
