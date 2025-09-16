from copy import deepcopy
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from json_repair import repair_json
from loguru import logger

from story_writer.config import DEFAULT_MODEL_CLIENT
from story_writer.prompts import (EVENT_COMPLETE_SYSTEM_PROMPT, EVENT_COMPLETE_USER_PROMPT,
                                  EVENT_RELATION_SYSTEM_PROMPT, EVENT_RELATION_USER_PROMPT, EVENT_REVISE_SYSTEM_PROMPT,
                                  EVENT_REVISE_USER_PROMPT, EVENT_SEED_SYSTEM_PROMPT, EVENT_SEED_USER_PROMPT,
                                  EVENT_VALID_SYSTEM_PROMPT, EVENT_VALID_USER_PROMPT)
from story_writer.schemas import Event, EventCompleteness, EventGraph, EventValidate, Relation


class EventCompletenessAgent(AssistantAgent):
    """Event completeness agent for checking whether a partial event list is complete enough to stop generation"""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            name='event_completeness_agent',
            model_client=model_client,
            system_message=EVENT_COMPLETE_SYSTEM_PROMPT,
            *args,
            **kwargs,
        )

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

    async def check_completeness(self, premise: str, partial_event_list: List[Event]) -> EventCompleteness:
        """Check the completeness of the event list against the premise

        Args:
            premise (str): The premise of the story
            partial_event_list (List[Event]): The list of events to check

        Returns:
            EventCompleteness: The completeness of the event list
        """
        await self.on_reset(CancellationToken())
        task = EVENT_COMPLETE_USER_PROMPT.format(
            premise=premise,
            partial_event_list=[e.model_dump() for e in partial_event_list],
        )
        task_result = await Console(self.run_stream(task=task))
        assert isinstance(task_result.messages[-1], TextMessage)
        return self._parse_event_completeness(task_result.messages[-1].content)


class EventSeedAgents:
    """Event seed agents for generating and revising events"""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *args,
        **kwargs,
    ) -> None:
        self.event_seed_agent = AssistantAgent(
            name='event_seed_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_SEED_SYSTEM_PROMPT,
            *args,
            **kwargs,
        )
        self.event_revise_agent = AssistantAgent(
            name='event_revise_agent',
            model_client=model_client,
            model_client_stream=True,
            system_message=EVENT_REVISE_SYSTEM_PROMPT,
            *args,
            **kwargs,
        )

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

    async def generate_events(
        self,
        premise: str,
        partial_event_list: List[Event],
        completeness_reason: str,
        missing_elements: List[str],
        k_candidates: int,
    ) -> List[Event]:
        """Generate events candidates

        Args:
            premise (str): Story premise
            partial_event_list (List[Event]): Partial event list
            completeness_reason (str): Completeness reason
            missing_elements (List[str]): Missing elements
            k_candidates (int): Number of candidates to generate

        Returns:
            List[Event]: List of Event objects
        """
        await self.event_seed_agent.on_reset(CancellationToken())  # clear model context (history)
        task = EVENT_SEED_USER_PROMPT.format(
            premise=premise,
            partial_event_list=[e.model_dump() for e in partial_event_list],
            completeness_reason=completeness_reason,
            missing_elements=missing_elements,
            k_candidates=k_candidates,
        )
        task_result = await Console(self.event_seed_agent.run_stream(task=task))
        assert isinstance(task_result.messages[-1], TextMessage)  # assert model response is text
        event_list = self._parse_event(task_result.messages[-1].content)
        return event_list

    async def revise_events(
        self,
        premise: str,
        partial_event_list: List[Event],
        original_candidate: List[Event],
        validator_feedback: List[EventValidate],
    ) -> List[Event]:
        """Revise event candidates

        Args:
            premise (str): Story premise
            partial_event_list (List[Event]): Partial event list
            original_candidate (List[Event]): Original candidate events
            validator_feedback (List[EventValidate]): Validator feedback

        Returns:
            List[Event]: List of Event objects
        """
        await self.event_revise_agent.on_reset(CancellationToken())  # clear model context (history)
        task = EVENT_REVISE_USER_PROMPT.format(
            premise=premise,
            partial_event_list=[e.model_dump() for e in partial_event_list],
            original_candidate=[e.model_dump() for e in original_candidate],
            validator_feedback=[e.model_dump() for e in validator_feedback],
        )
        task_result = await Console(self.event_revise_agent.run_stream(task=task))
        assert isinstance(task_result.messages[-1], TextMessage)
        revised_event_list = self._parse_event(task_result.messages[-1].content)
        return revised_event_list


class EventValidatorAgent(AssistantAgent):
    """Event validator agent for validating event candidates"""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            name='event_validator_agent',
            model_client=model_client,
            system_message=EVENT_VALID_SYSTEM_PROMPT,
            *args,
            **kwargs,
        )

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

    async def validate_events(self, premise: str, partial_event_list: List[Event],
                              candidates: List[Event]) -> List[EventValidate]:
        """Validate event candidates

        Args:
            premise (str): Story premise
            partial_event_list (List[Event]): Partial event list
            candidates (List[Event]): Candidate events

        Returns:
            List[EventValidate]: List of EventValidate objects
        """
        await self.on_reset(CancellationToken())  # clear model context (history)
        task = EVENT_VALID_USER_PROMPT.format(
            premise=premise,
            partial_event_list=[e.model_dump() for e in partial_event_list],
            candidates=[e.model_dump() for e in candidates],
        )
        task_result = await Console(self.run_stream(task=task))
        assert isinstance(task_result.messages[-1], TextMessage)  # assert model response is text
        event_val_list = self._parse_event_valid(task_result.messages[-1].content)
        return event_val_list


class EventRelationAgent(AssistantAgent):
    """Event Relation Agent for generating event relations"""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            name='event_relation_agent',
            model_client=model_client,
            system_message=EVENT_RELATION_SYSTEM_PROMPT,
            *args,
            **kwargs,
        )

    def _parse_event_relations(self, event_relation_text: str) -> List[Relation]:
        """Parse event relations text into Relation objects

        Args:
            event_relations_text (str): Text containing event relations

        Returns:
            List[Relation]: List of Relation objects
        """
        relation_obj_list = repair_json(event_relation_text, ensure_ascii=False, return_objects=True)
        if not isinstance(relation_obj_list, list):
            relation_obj_list = [relation_obj_list]
        relation_list = []
        for relation_obj in relation_obj_list:
            try:
                relation = Relation.model_validate(relation_obj)
                relation_list.append(relation)
            except Exception as e:
                logger.warning(f"Failed to validate relation: {e}")
        return relation_list

    async def generate_ralations(self, premise: str, event_list: List[Event]) -> List[Relation]:
        """Generate relations between events

        Args:
            premise (str): The premise of the story
            event_list (List[Event]): List of events

        Returns:
            List[Relation]: List of relations
        """
        await self.on_reset(CancellationToken())
        task = EVENT_RELATION_USER_PROMPT.format(premise=premise, event_list=[e.model_dump() for e in event_list])
        task_result = await Console(self.run_stream(task=task))
        assert isinstance(task_result.messages[-1], TextMessage)
        relation_list = self._parse_event_relations(task_result.messages[-1].content)
        return relation_list


class OutlineBuilder:

    def __init__(
        self,
        model_client: ChatCompletionClient = DEFAULT_MODEL_CLIENT,
        k_candidates: int = 3,
        max_val: int = 2,
        max_events: int = 30,
    ) -> None:
        """Outline builder

        Args:
            model_client (ChatCompletionClient, optional): Model client. defaults to DEFAULT_MODEL_CLIENT
            k_candidates (int, optional): Number of candidate events to generate in one iteration. defaults to 3
            max_val (int, optional): Maximum number of times to validate an event in one iteration. defaults to 2
            max_events (int, optional): Maximum number of events to generate. defaults to 30
        """
        self.event_completeness_agent = EventCompletenessAgent(model_client=model_client)
        self.event_seed_agents = EventSeedAgents(model_client=model_client)
        self.event_validator_agent = EventValidatorAgent(model_client=model_client)
        self.event_relation_agent = EventRelationAgent(model_client=model_client)
        self.k_candidates = k_candidates
        self.max_val = max_val
        self.max_events = max_events

    async def build_outline(self, premise: str, partial_event_list: List[Event] = []) -> EventGraph:
        """Build story outline

        Args:
            premise (str): Story premise
            partial_event_list (List[Event], optional): Partial event list. Defaults to [].

        Returns:
            EventGraph: Event graph (story outline)
        """
        logger.info('Building story outline...')
        partial_event_list = deepcopy(partial_event_list)
        max_total_iters = max(30, self.max_events * self.max_val)  # set max iterations to avoid infinite loop
        for gen_iter in range(max_total_iters):
            if len(partial_event_list) >= self.max_events:
                logger.info('Max events reached, stopping generation')
                break

            # 1. check completeness
            logger.info(f"Checking completeness, current iteration {gen_iter + 1}, max iterations {max_total_iters}")
            completeness = await self.event_completeness_agent.check_completeness(premise, partial_event_list)
            if completeness.complete:
                logger.info(f"Outline complete: {completeness.reason}")
                break
            logger.info(f"Outline incomplete: {completeness.reason}")

            # 2. generate K candidate events
            logger.info(
                f"Generating candidate events, current iteration {gen_iter + 1}, max iterations {max_total_iters}")
            candidates = await self.event_seed_agents.generate_events(
                premise,
                partial_event_list,
                completeness.reason,
                completeness.missing_elements,
                self.k_candidates,
            )
            logger.info(f"Generated {len(candidates)} event candidates")
            logger.info(f"Event candidates: {candidates}")
            if len(candidates) == 0:
                logger.warning('No event candidates generated, stopping')
                break

            # 3. validate and revise each candidate
            for val_iter in range(self.max_val):
                logger.info(f"Validating candidate events, {val_iter + 1}/{self.max_val}")
                event_validates = await self.event_validator_agent.validate_events(
                    premise=premise,
                    partial_event_list=partial_event_list,
                    candidates=candidates,
                )
                logger.info(f"Event validations: {event_validates}")
                # filter out invalid events
                event_candidate_dict = {e.event_id: e for e in candidates}
                event_validate_dict = {e.event_id: e for e in event_validates}
                event_ids = [event_id for event_id in event_candidate_dict.keys() if event_id in event_validate_dict]
                if len(event_ids) < len(candidates):
                    logger.warning(f"Some event candidates were not validated, {len(event_ids)}/{len(candidates)}")
                # update partial event list with valid events
                valid_event_list = [
                    event_candidate_dict[event_id] for event_id in event_ids if event_validate_dict[event_id].valid
                ]
                partial_event_list.extend(valid_event_list)
                logger.info(f"Accepted {len(valid_event_list)} events, {len(partial_event_list)} total")
                # revise invalid events
                invalid_event_list = [
                    event_candidate_dict[event_id] for event_id in event_ids if not event_validate_dict[event_id].valid
                ]
                invalid_event_feedback = [
                    event_validate_dict[event_id] for event_id in event_ids if not event_validate_dict[event_id].valid
                ]
                if len(invalid_event_list) == 0:
                    break  # All events are valid
                logger.info(f"Need to revise {len(invalid_event_list)} events")
                candidates = await self.event_seed_agents.revise_events(
                    premise=premise,
                    partial_event_list=partial_event_list,
                    original_candidate=invalid_event_list,
                    validator_feedback=invalid_event_feedback,
                )
                logger.info(f"Revised {len(candidates)} event candidates")
                logger.info(f"Revised event candidates: {candidates}")

        logger.info(f"Accepted {len(partial_event_list)} events, generating relations")
        relations = await self.event_relation_agent.generate_ralations(
            premise=premise,
            event_list=partial_event_list,
        )
        logger.info(f"Generated {len(relations)} relations")
        event_graph = EventGraph(
            nodes=partial_event_list,
            edges=relations,
        )
        return event_graph
