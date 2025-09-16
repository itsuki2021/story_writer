import json

from story_writer.schemas import Event, EventCompleteness, EventValidate, Relation

# EventCompleteness prompt
EVENT_COMPLETE_SYSTEM_PROMPT = f"""You are EventCompletenessAgent.
Your job is to analyze the current Partial Event List and decide whether the story outline is complete.

Completeness definition:
1. Narrative arc: Beginning, Conflict, Climax, and Resolution should all be present.
2. Character coverage: All major characters from the premise should appear at least once.
3. Causal chain: At least one coherent chain of cause-effect relations should exist.
4. No major gaps: Events should connect logically without large unexplained jumps.

Output MUST be valid JSON following the EventCompleteness schema.

EventCompleteness schema:
{json.dumps(EventCompleteness.model_json_schema(), indent=2, ensure_ascii=False)}

Requirements:
- Always output a JSON object, never plain text
- `complete=true` if the outline is narratively sufficient
- If incomplete, set `complete=false` and list what is missing in `missing_elements`
"""

EVENT_COMPLETE_USER_PROMPT = """Premise:
{premise}

PartialEventList:
{partial_event_list}

Task:
- Analyze whether the outline is complete.
- Return a JSON object following the EventCompleteness schema.
"""

# EventSeed prompt
EVENT_SEED_SYSTEM_PROMPT = f"""You are EventSeedAgent.
Your task: given a story premise and an existing Partial Event List, generate one or multiple candidate events that extend the list.
Output MUST be valid JSON following the Event schema.

{json.dumps(Event.model_json_schema(), indent=2, ensure_ascii=False)}

Requirements:
1. You need to output JSON array.
2. Each event must include time, location, characters, goal, and conflict.
3. Keep outputs concise. Do NOT include any commentary outside JSON.
"""

EVENT_SEED_USER_PROMPT = """Premise:
{premise}

PartialEventList:
{partial_event_list}

CompletenessStatus:
Reason: {completeness_reason}
MissingElements: {missing_elements}

Requirements:
- Produce up to {k_candidates} candidate events
- Focus especially on covering the missing elements listed in CompletenessStatus
- Ensure consistency with the existing PartialEventList
"""

# EventValidator prompt
EVENT_VALID_SYSTEM_PROMPT = f"""You are EventValidatorAgent.
Your job: examine each candidate event (JSON) and validate it against the provided Premise and PartialEventList.
Validation rules (apply in order):
1. Temporal Consistency: a character dead in prior events cannot be alive here without explanation.
2. Causal Plausibility: claimed causal relations must reference events that plausibly cause the effect.
3. Character State Consistency: character states (injured, healthy, mental-state) must be consistent.
4. World/Setting Consistency: events must not violate premise world rules.
5. Redundancy: flag near-duplicate events (novelty_score < 0.2 vs existing events).

For each candidate event, output MUST be valid JSON following the Validate schema.

{json.dumps(EventValidate.model_json_schema(), indent=2, ensure_ascii=False)}

Requirements:
- You MUST validate EACH candidate event individually
- For EACH candidate event, output a separate valid JSON following the Validate schema
- Output MUST be a JSON array containing one validation result per candidate event
- Do NOT provide a single aggregated validation result
- Do NOT include any commentary or explanation outside the JSON array
"""

EVENT_VALID_USER_PROMPT = """Premise:
{premise}

PartialEventList:
{partial_event_list}

Candidates:
{candidates}
"""

# EventSeed revise prompt
EVENT_REVISE_SYSTEM_PROMPT = f"""You are EventSeedAgent. Your role is to generate or revise story events in structured JSON format.
This time, you are asked to REVISE an existing candidate event according to the validator's feedback.
Strictly follow these rules:
1. Always return a JSON array of one or more candidate events, following the Event schema.
2. You MUST correct issues raised in ValidatorFeedback (e.g., temporal inconsistency, character state conflicts).
3. Ensure coherence with the Premise and consistency with the PartialEventList.
4. Do not output any explanation or commentary outside JSON.
5. If the feedback cannot be fixed, discard the event directly.

Event schema:
{json.dumps(Event.model_json_schema(), indent=2, ensure_ascii=False)}
"""

EVENT_REVISE_USER_PROMPT = """Premise:
{premise}

PartialEventList:
{partial_event_list}

OriginalCandidate (to be revised):
{original_candidate}

ValidatorFeedback (issues to fix):
{validator_feedback}
"""

# EventRelation prompt
EVENT_RELATION_SYSTEM_PROMPT = f"""You are EventRelationAgent.
Your task: given a story premise, an existing list of events, generate all appropriate relations (edges) between the events to form a coherent graph.

Output MUST be valid JSON array following the Relation schema.

{json.dumps(Relation.model_json_schema(), indent=2, ensure_ascii=False)}

Requirements:
1. Generate relations with types like 'causal' (one event causes another), 'temporal' (one follows another in time), 'thematic' (shared theme or motif), or other relevant types.
2. For each relation, specify source_event_id (the originating event), target_event_id (the connected event), type, and a brief rationale explaining why the relation exists.
3. Ensure relations maintain temporal, causal, and character consistency based on the premise and event details (e.g., time, characters, goals, conflicts).
4. Avoid duplicates: do not create redundant relations (e.g., multiple identical causal links).
5. Keep relations concise and relevant to story progression. Do NOT include any commentary or explanation outside the JSON array.
"""

EVENT_RELATION_USER_PROMPT = """Premise:
{premise}

Event List:
{event_list}

Requirements:
- Generate relations for the entire set of events.
- Prioritize creating causal chains to cover narrative arcs (e.g., build-up to climax).
- Ensure relations enhance coherence and novelty based on event scores.
"""
