from copy import deepcopy
from typing import Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from json_repair import repair_json
from loguru import logger

from story_writer.config import DEFAULT_MODEL_CLIENT
from story_writer.prompts import (EVENT_COMPLETE_SYSTEM_PROMPT, EVENT_COMPLETE_USER_PROMPT, EVENT_REVISE_SYSTEM_PROMPT,
                                  EVENT_REVISE_USER_PROMPT, EVENT_SEED_SYSTEM_PROMPT, EVENT_SEED_USER_PROMPT,
                                  EVENT_VALID_SYSTEM_PROMPT, EVENT_VALID_USER_PROMPT)
from story_writer.schemas import Event, EventCompleteness, EventValidate

EventGraph = Dict[str, Event]


class OutlineBuilder:

    def __init__(
        self,
        model_client: ChatCompletionClient = DEFAULT_MODEL_CLIENT,
        k_candidates: int = 5,
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
            name='event_seed_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_SEED_SYSTEM_PROMPT,
        )
        self.event_validator_agent = AssistantAgent(
            name='event_validator_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_VALID_SYSTEM_PROMPT,
        )
        self.event_revise_agent = AssistantAgent(
            name='event_revise_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_REVISE_SYSTEM_PROMPT,
        )
        self.event_completeness_agent = AssistantAgent(
            name='event_completeness_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_COMPLETE_SYSTEM_PROMPT,
        )
        self.k_candidates = k_candidates
        self.max_revise = max_revise
        self.max_events = max_events

    def _resolve_event_id_conflict(self, graph: EventGraph, event: Event) -> Event:
        """Resolve event id conflict

        Args:
            graph (EventGraph): Event graph
            event (Event): Event

        Returns:
            Event: Resolved event
        """
        eid = event.event_id
        if eid in graph:
            suffix = 1
            while f"{eid}_{suffix}" in graph:
                suffix += 1
            event.event_id = f"{eid}_{suffix}"
        return event

    def _parse_event(self, event_text: str) -> List[Event]:
        """Parse event text into Event objects

        Args:
            event_text (str): Event text

        Returns:
            List[Event]: List of Event objects
        """
        event_obj_list = repair_json(event_text, ensure_ascii=False, return_objects=True)
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
        event_val_obj_list = repair_json(event_valid_text, ensure_ascii=False, return_objects=True)
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

    def _parse_event_completeness(self, completeness_text: str) -> EventCompleteness:
        """Parse event completeness text into EventCompleteness objects

        Args:
            completeness_text (str): Event completeness text

        Returns:
            EventCompleteness: EventCompleteness object
        """
        completeness_obj = repair_json(completeness_text, ensure_ascii=False, return_objects=True)
        try:
            return EventCompleteness.model_validate(completeness_obj)
        except Exception as e:
            logger.warning(f"Failed to parse completeness check: {e}")
            return EventCompleteness(complete=False, reason='Parse error', missing_elements=['unknown'])

    async def _check_completeness(self, premise: str, partial_graph: EventGraph) -> EventCompleteness:
        """Check completeness of the outline

        Args:
            premise (str): Story premise
            partial_graph (EventGraph): Partial event graph

        Returns:
            EventCompleteness: EventCompleteness object
        """
        completeness_task = EVENT_COMPLETE_USER_PROMPT.format(
            premise=premise,
            partial_graph=[e.model_dump() for e in partial_graph.values()],
        )
        await self.event_completeness_agent.on_reset(CancellationToken())
        completeness_result = await Console(self.event_completeness_agent.run_stream(task=completeness_task))
        assert isinstance(completeness_result.messages[-1], TextMessage)
        return self._parse_event_completeness(completeness_result.messages[-1].content)

    async def _eventseed_generate(
        self,
        premise: str,
        partial_graph: EventGraph,
        completeness_reason: str,
        missing_elements: List[str],
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
            completeness_reason=completeness_reason,
            missing_elements=missing_elements,
        )
        await self.event_seed_agent.on_reset(CancellationToken())  # clear model context
        event_seed_task_result = await Console(self.event_seed_agent.run_stream(task=event_seed_task))
        assert isinstance(event_seed_task_result.messages[-1], TextMessage)
        event_candidates = self._parse_event(event_seed_task_result.messages[-1].content)
        return event_candidates

    async def _eventvalidator_validate(self, premise: str, partial_graph: EventGraph,
                                       event_candidates: List[Event]) -> List[EventValidate]:
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
        await self.event_validator_agent.on_reset(CancellationToken())  # clear model context
        event_valid_task_result = await Console(self.event_validator_agent.run_stream(task=event_valid_task))
        assert isinstance(event_valid_task_result.messages[-1], TextMessage)
        event_val_list = self._parse_event_valid(event_valid_task_result.messages[-1].content)
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
        await self.event_revise_agent.on_reset(CancellationToken())  # clear model context
        event_revise_task_result = await Console(self.event_revise_agent.run_stream(task=event_revise_task))
        assert isinstance(event_revise_task_result.messages[-1], TextMessage)
        revised_event_list = self._parse_event(event_revise_task_result.messages[-1].content)
        return revised_event_list

    async def build_outline(self, premise: str, partial_graph: EventGraph = {}) -> EventGraph:
        """Build story outline

        Args:
            premise (str): Story premise
            partial_graph (EventGraph, optional): Partial event graph. Defaults to {}.

        Returns:
            EventGraph: Event graph (story outline)
        """
        logger.info('Building story outline')
        partial_graph = deepcopy(partial_graph)
        max_total_iters = self.max_events * self.max_revise * 2
        for gen_iter in range(max_total_iters):
            if len(partial_graph) >= self.max_events:  # max events reached
                logger.info('Max events reached, stopping')
                break

            # 1. check completeness
            logger.info(f"Checking completeness, iteration {gen_iter + 1}/{max_total_iters}")
            completeness = await self._check_completeness(premise, partial_graph)
            if completeness.complete:
                logger.info(f"Outline complete: {completeness.reason}")
                break
            logger.info(f"Outline incomplete: {completeness.reason}")

            # 2. generate K candidate events
            logger.info(f"Generating event candidates, iteration {gen_iter + 1}/{max_total_iters}")
            event_candidates = await self._eventseed_generate(
                premise,
                partial_graph,
                completeness.reason,
                completeness.missing_elements,
            )
            logger.info(f"Generated {len(event_candidates)} event candidates")
            logger.info(f"Event candidates: {event_candidates}")
            if len(event_candidates) == 0:
                logger.warning('No event candidates generated, stopping')
                break

            # 3. validate each candidate and revise if needed
            for revise_iter in range(self.max_revise):
                # generate K validations
                logger.info(f"Validating event candidates, iteration {revise_iter + 1}/{self.max_revise}")
                event_validates = await self._eventvalidator_validate(premise, partial_graph, event_candidates)
                logger.info(f"Generated {len(event_validates)} event validations")
                logger.info(f"Event validations: {event_validates}")
                # filter out invalid events
                event_candidate_dict = {e.event_id: e for e in event_candidates}
                event_validates_dict = {e.event_id: e for e in event_validates}
                event_ids = [event_id for event_id in event_candidate_dict.keys() if event_id in event_validates_dict]
                if len(event_ids) < len(event_candidates):
                    logger.warning(
                        f"Some event candidates were not validated, {len(event_ids)}/{len(event_candidates)}")
                # update partial graph with valid events
                event_passed_dict = {
                    event_id: event_candidate_dict[event_id]
                    for event_id in event_ids
                    if event_validates_dict[event_id].valid
                }
                for event_id, event in event_passed_dict.items():
                    event = self._resolve_event_id_conflict(partial_graph, event)
                    partial_graph[event.event_id] = event
                logger.info(f"Accepted {len(event_passed_dict)} events, {len(partial_graph)} total")
                # revise events
                event_need_revised = [
                    event_candidate_dict[event_id] for event_id in event_ids if not event_validates_dict[event_id].valid
                ]
                event_need_revised_feedback = [
                    event_validates_dict[event_id] for event_id in event_ids if not event_validates_dict[event_id].valid
                ]
                if len(event_need_revised) == 0:
                    break  # no need to revise
                logger.info(f"Need to revise {len(event_need_revised)} events")
                event_candidates = await self._eventrevise_revise(
                    premise,
                    partial_graph,
                    event_need_revised,
                    event_need_revised_feedback,
                )
                logger.info(f"Revised {len(event_candidates)} event candidates")
                logger.info(f"Revised event candidates: {event_candidates}")

        logger.info(f"Accepted {len(partial_graph)} events")
        return partial_graph
