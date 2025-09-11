import os
from copy import deepcopy
from typing import Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from json_repair import repair_json
from loguru import logger

from story_writer.prompts import (
    EVENT_REVISE_SYSTEM_PROMPT,
    EVENT_REVISE_USER_PROMPT,
    EVENT_SEED_SYSTEM_PROMPT,
    EVENT_SEED_USER_PROMPT,
    EVENT_VALID_SYSTEM_PROMPT,
    EVENT_VALID_USER_PROMPT,
)
from story_writer.schemas import Event, EventValidate

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
EventGraph = Dict[str, Event]


class OutlineBuilder:
    def __init__(
        self,
        model_client: ChatCompletionClient = DEFAULT_MODEL_CLIENT,
        k_candidates: int = 3,
        max_revise: int = 3,
        max_events: int = 30,
    ) -> None:
        """Outline builder

        Args:
            model_client (ChatCompletionClient, optional): Model client. defaults to DEFAULT_MODEL_CLIENT
            k_candidates (int, optional): Number of candidate events to generate in one iteration. defaults to 3
            max_revise (int, optional): Maximum number of times to revise an event in one iteration. defaults to 3
            max_events (int, optional): Maximum number of events to generate. defaults to 30
        """
        self.event_seed_agent = AssistantAgent(
            name="event_seed_agent",
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_SEED_SYSTEM_PROMPT,
        )
        self.event_validator_agent = AssistantAgent(
            name="event_validator_agent",
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_VALID_SYSTEM_PROMPT,
        )
        self.event_revise_agent = AssistantAgent(
            name="event_revise_agent",
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_REVISE_SYSTEM_PROMPT,
        )
        self.k_candidates = k_candidates
        self.max_revise = max_revise
        self.max_events = max_events

    def _parse_event(self, event_text: str) -> List[Event]:
        """Parse event text into Event objects

        Args:
            event_text (str): Event text

        Returns:
            List[Event]: List of Event objects
        """
        event_obj_list = repair_json(
            event_text, ensure_ascii=False, return_objects=True
        )
        if not isinstance(event_obj_list, list):
            event_obj_list = [event_obj_list]
        event_list = []
        for event_obj in event_obj_list:
            try:
                event = Event.model_validate(event_obj)
                event_list.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
        return event_list

    def _parse_event_valid(self, event_valid_text: str) -> List[EventValidate]:
        """Parse event valid text into EventValidate objects

        Args:
            event_valid_text (str): Event valid text

        Returns:
            List[EventValidate]: List of EventValidate objects
        """
        event_val_obj_list = repair_json(
            event_valid_text, ensure_ascii=False, return_objects=True
        )
        if not isinstance(event_val_obj_list, list):
            event_val_obj_list = [event_val_obj_list]
        event_val_list = []
        for event_obj in event_val_obj_list:
            try:
                event_val = EventValidate.model_validate(event_obj)
                event_val_list.append(event_val)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
        return event_val_list

    async def _eventseed_generate(
        self, premise: str, partial_graph: EventGraph
    ) -> List[Event]:
        """Generate event candidates

        Args:
            premise (str): Story premise
            partial_graph (EventGraph): Partial event graph

        Returns:
            List[Event]: List of event candidates
        """
        event_seed_task = EVENT_SEED_USER_PROMPT.format(
            premise=premise,
            partial_graph=[e.model_dump() for e in partial_graph.values()],
            k_candidates=self.k_candidates,
        )
        await self.event_seed_agent.on_reset(CancellationToken())  # clear model context
        event_seed_task_result = await Console(
            self.event_seed_agent.run_stream(task=event_seed_task)
        )
        assert isinstance(event_seed_task_result.messages[-1], TextMessage)
        event_candidates = self._parse_event(
            event_seed_task_result.messages[-1].content
        )
        return event_candidates

    async def _eventvalidator_validate(
        self, premise: str, partial_graph: EventGraph, event_candidates: List[Event]
    ) -> List[EventValidate]:
        """Validate event candidates

        Args:
            premise (str): Story premise
            partial_graph (EventGraph): Partial event graph
            event_candidates (List[Event]): List of event candidates

        Returns:
            List[EventValidate]: List of event validations
        """
        event_valid_task = EVENT_VALID_USER_PROMPT.format(
            premise=premise,
            partial_graph=[e.model_dump() for e in partial_graph.values()],
            candidates=[e.model_dump() for e in event_candidates],
        )
        await self.event_validator_agent.on_reset(
            CancellationToken()
        )  # clear model context
        event_valid_task_result = await Console(
            self.event_validator_agent.run_stream(task=event_valid_task)
        )
        assert isinstance(event_valid_task_result.messages[-1], TextMessage)
        event_val_list = self._parse_event_valid(
            event_valid_task_result.messages[-1].content
        )
        return event_val_list

    async def _eventrevise_revise(
        self,
        premise: str,
        partial_graph: EventGraph,
        original_candidate: List[Event],
        validator_feedback: List[EventValidate],
    ) -> List[Event]:
        """Revise event candidates

        Args:
            premise (str): Story premise
            partial_graph (EventGraph): Partial event graph
            original_candidate (List[Event]): List of original event candidates (to be revised)
            validator_feedback (List[EventValidate]): List of event validations (issues to fix)

        Returns:
            List[Event]: List of revised event
        """
        event_revise_task = EVENT_REVISE_USER_PROMPT.format(
            premise=premise,
            partial_graph=[e.model_dump() for e in partial_graph.values()],
            original_candidate=[e.model_dump() for e in original_candidate],
            validator_feedback=[e.model_dump() for e in validator_feedback],
        )
        await self.event_revise_agent.on_reset(
            CancellationToken()
        )  # clear model context
        event_revise_task_result = await Console(
            self.event_revise_agent.run_stream(task=event_revise_task)
        )
        assert isinstance(event_revise_task_result.messages[-1], TextMessage)
        revised_event_list = self._parse_event(
            event_revise_task_result.messages[-1].content
        )
        return revised_event_list

    async def build_outline(
        self, premise: str, partial_graph: EventGraph = {}
    ) -> EventGraph:
        partial_graph = deepcopy(partial_graph)
        event_counter = 0
        while event_counter < self.max_events:
            # 1. generate K candidate events
            event_candidates = await self._eventseed_generate(premise, partial_graph)
            logger.info(f"Generated {len(event_candidates)} event candidates")
            logger.debug(f"Event candidates: {event_candidates}")
            if len(event_candidates) == 0:
                logger.warning("No event candidates generated, stopping")
                break
            # 2. validate each candidate and revise if needed
            accepted_any = False
            for _ in range(self.max_revise):
                event_validates = await self._eventvalidator_validate(
                    premise, partial_graph, event_candidates
                )
                logger.debug(f"Event validations: {event_validates}")
                event_candidate_dict = {e.event_id: e for e in event_candidates}
                event_validates_dict = {e.event_id: e for e in event_validates}
                event_ids = [
                    event_id
                    for event_id in event_candidate_dict.keys()
                    if event_id in event_validates_dict
                ]
                # update partial graph with valid events
                event_passed_dict = {
                    event_id: event_candidate_dict[event_id]
                    for event_id in event_ids
                    if event_validates_dict[event_id].valid
                }
                partial_graph.update(event_passed_dict)
                event_counter += len(event_passed_dict)
                accepted_any |= len(event_passed_dict) > 0
                logger.info(
                    f"Accepted {len(event_passed_dict)} events, {event_counter} total"
                )
                # revise events
                event_need_revised = [
                    event_candidate_dict[event_id]
                    for event_id in event_ids
                    if not event_validates_dict[event_id].valid
                ]
                if len(event_need_revised) == 0:
                    break
                event_need_revised_feedback = [
                    event_validates_dict[event_id]
                    for event_id in event_ids
                    if not event_validates_dict[event_id].valid
                ]
                event_candidates = await self._eventrevise_revise(
                    premise,
                    partial_graph,
                    event_need_revised,
                    event_need_revised_feedback,
                )
                logger.debug(f"Revised event candidates: {event_candidates}")
            if not accepted_any:
                logger.warning("No event accepted, stopping")
                break
        logger.info(f"Accepted {len(partial_graph)} events")
        return partial_graph
