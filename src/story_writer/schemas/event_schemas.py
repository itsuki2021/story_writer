from typing import List

from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str = Field(..., description="Name of the character, e.g. Winston Smith")
    role: str = Field(..., description="Role of the character, e.g. protagonist")
    state: str = Field(..., description="State of the character, e.g. wounded")


class Relation(BaseModel):
    type: str = Field(..., description="Type of relation, e.g. causal")
    target_event_id: str = Field(..., description="ID of the target event")
    rationale: str = Field(..., description="Rationale for the relation")


class Event(BaseModel):
    event_id: str = Field(..., description="ID of the event, e.g. E1")
    title: str = Field(..., description="Title of the event")
    summary: str = Field(..., description="Summary of the event")
    time: str = Field(..., description="Time of the event")
    location: str = Field(..., description="Location of the event")
    characters: List[Character] = Field(
        ..., description="Characters involved in the event"
    )
    goal: str = Field(..., description="Goal of the event")
    conflict: str = Field(..., description="Conflict involved in the event")
    relations: List[Relation] = Field(..., description="Relations between each events")


class EventValidate(BaseModel):
    event_id: str = Field(..., description="ID of the event, e.g. E1")
    suggestion: str = Field(..., description="Suggestion for the event")
    novelty_score: float = Field(
        ..., description="Novelty score of the event, 0-1, 1 being most novel"
    )
    coherence_score: float = Field(
        ..., description="Coherence score of the event, 0-1, 1 being most coherent"
    )
    valid: bool = Field(..., description="Whether the event is valid")
