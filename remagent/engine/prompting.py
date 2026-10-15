"""
Shared prompt builder, output schema, and parser for dream consolidation.

Every backend (Gemini, Anthropic, OpenAI-compatible) sends exactly the same
system prompt and user prompt, and every raw response goes through the same
parser into the same DreamSynthesisOutput type. Provider-specific code lives
in remagent.engine.providers and never touches prompt text or parsing.
"""

from ..sensitive import redact
import copy
import json
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from remagent.engine.errors import DreamSynthesisError
from remagent.schemas import Fact, MemoryProfile, OperationalRule, RawTurnLog


class SynthesizedFact(BaseModel):
    """A single extracted fact, as returned by the LLM."""
    entity: str = Field(..., description="Target entity, user, project, or domain object (e.g., 'User', 'Vendor')")
    attribute: str = Field(..., description="Specific property or relation (e.g., 'price_per_unit', 'auth_strategy')")
    value: str = Field(..., description="Current ground-truth value as a string (e.g., '$52/unit', 'OAuth 2.0')")
    confidence: float = Field(default=1.0, description="Confidence score of the synthesized fact (0.0 - 1.0)")


class SynthesizedRule(BaseModel):
    """A single operational rule, as returned by the LLM."""
    category: str = Field(..., description="One of: 'user_preference', 'coding_standard', 'architecture_heuristic', 'operational_directive', 'domain_constraint'")
    rule: str = Field(..., description="Concise, actionable rule statement")
    rationale: str = Field(default="Synthesized from user interactions", description="Reasoning or empirical turn context justifying this directive")
    priority: int = Field(default=3, description="Priority weight (1 = critical/unbreakable, 5 = minor guideline)")


class SynthesizedContradiction(BaseModel):
    """A contradiction between a new observation and an existing fact, as returned by the LLM."""
    entity: str = Field(..., description="Subject entity of the contradicted fact (must match the existing fact's entity exactly)")
    attribute: str = Field(..., description="Subject attribute of the contradicted fact (must match the existing fact's attribute exactly)")
    prior_value: str = Field(..., description="The outdated value being superseded")
    new_value: str = Field(..., description="The new ground-truth value")
    resolution_reasoning: str = Field(..., description="Reasoning why the new observation overrides the prior state")


class DreamSynthesisOutput(BaseModel):
    """Structured JSON response schema shared by every backend."""
    added_facts: List[SynthesizedFact] = Field(
        default_factory=list,
        description="Discrete entity facts extracted from the unconsolidated turns"
    )
    updated_rules: List[SynthesizedRule] = Field(
        default_factory=list,
        description="Operational directives or preferences synthesized from the turns"
    )
    contradictions: List[SynthesizedContradiction] = Field(
        default_factory=list,
        description="Contradictions resolved where new evidence supersedes prior facts"
    )
    pruned_noise_count: int = Field(
        default=0,
        description="Number of noisy turns, pleasantries, or ephemeral tool logs discarded"
    )
    pruned_noise_categories: List[str] = Field(
        default_factory=list,
        description="Types of noise pruned (e.g., 'chit_chat', 'transient_cli_error', 'greetings')"
    )
    reasoning_summary: str = Field(
        ...,
        description="High-level cognitive explanation of what was consolidated and resolved during this dream cycle"
    )


DREAM_PROMPT_SYSTEM = """
You are the RemAgent Autonomous Dream Synthesizer — a cognitive memory consolidation engine inspired by biological REM sleep.
Your mission is to process episodic conversation logs and consolidate them into a crisp, high-signal, zero-vector knowledge graph.

Unlike noisy, brittle Vector RAG (which blindly chunks text and suffers from semantic drift), you perform three rigorous cognitive operations:

1. NOISE PRUNING:
   - Discard all conversational filler, greetings, pleasantries, apologies, and chit-chat.
   - Discard transient tool calls, temporary scratchpads, intermediate CLI errors, and dead ends that were later corrected.
   - Count the number of discarded noise elements and classify their category.

2. ENTITY FACT EXTRACTION & KNOWLEDGE GRAPH:
   - Extract discrete, definitive attributes about users, systems, project architectures, tech stacks, credentials policies, and environments.
   - MANDATORY ENTITY REUSE: before introducing a new entity name, scan the EXISTING CONSOLIDATED KNOWLEDGE GRAPH for an entity that refers to the same real-world thing and REUSE its exact name (e.g. do not create "Commander" if "Commander Project" already exists, or "Vincent Agent" if "Vincent" exists). Same-thing-different-name splits break contradiction resolution. Reuse existing attribute names the same way.
   - Format: entity, attribute, value, confidence (0.0 - 1.0).
   - Never extract vague sentiments; focus on crisp ground-truth.

3. CONTRADICTION RESOLUTION & SUPERSEDING:
   - Carefully review the existing memory facts.
   - If a new turn contains a revised decision (e.g., user changed database from MySQL to PostgreSQL, or changed preferred style from functional to OOP), explicitly resolve the contradiction: the newer observation SUPERSEDES the older one.
   - State the reasoning why the new fact overrides the prior state.
   - MANDATORY: every contradiction you report MUST be accompanied by an entry in added_facts carrying the same entity and attribute with the new ground-truth value. A contradiction entry alone does NOT store the new value — omitting the added_facts entry would erase the memory instead of updating it.

4. OPERATIONAL DIRECTIVES & RULES:
   - Synthesize durable heuristics (e.g., "Always use TypeScript strict mode", "Do not modify port 3000", "Prefers async/await over raw promises").
   - Classify category: 'user_preference', 'coding_standard', 'architecture_heuristic', 'operational_directive', 'domain_constraint'.

Return the result strictly structured as valid JSON matching the required schema.
"""


# Caps on how much of the existing graph travels with a dream. The prompt used
# to carry every active fact and rule, which grows without bound: at 1,971
# facts and 818 rules the request first blew the per-minute token quota and
# then exceeded Gemini's 1,048,576-token input limit outright, so dreaming
# became impossible exactly when the backlog was largest.
#
# Selection is relevance-first rather than a blind truncation. Contradictions
# can only arise for entities the new turns actually mention, so every fact
# matching the window is included; the remainder fills by confidence as useful
# background.
MAX_CONTEXT_FACTS = 400
MAX_CONTEXT_RULES = 200


def select_context_facts(
    facts: List[Fact], turns_text: str, limit: int = MAX_CONTEXT_FACTS
) -> Tuple[List[Fact], int]:
    """Active facts worth sending with this window, plus the number omitted."""
    active = [f for f in facts if f.is_active]
    if len(active) <= limit:
        return active, 0
    haystack = turns_text.lower()
    relevant, background = [], []
    for fact in active:
        if fact.entity.lower() in haystack or fact.attribute.lower() in haystack:
            relevant.append(fact)
        else:
            background.append(fact)
    background.sort(key=lambda f: f.confidence, reverse=True)
    selected = (relevant + background)[:limit]
    return selected, len(active) - len(selected)


def select_context_rules(
    rules: List[OperationalRule], limit: int = MAX_CONTEXT_RULES
) -> Tuple[List[OperationalRule], int]:
    """Active rules worth sending, highest priority first, plus omitted count."""
    active = [r for r in rules if r.is_active]
    if len(active) <= limit:
        return active, 0
    ranked = sorted(active, key=lambda r: (r.priority, len(r.rule)))
    return ranked[:limit], len(active) - limit


def build_user_prompt(
    unconsolidated_turns: List[RawTurnLog],
    existing_profile: MemoryProfile,
    max_context_facts: int = MAX_CONTEXT_FACTS,
    max_context_rules: int = MAX_CONTEXT_RULES,
) -> Tuple[str, int]:
    """
    Render the user-turn prompt every backend sends. Returns the prompt and the
    raw character count of the turn contents (used for the token-savings estimate).
    """
    turn_history_str = []
    raw_char_count = 0
    for idx, turn in enumerate(unconsolidated_turns, 1):
        tool_info = f" [Tools: {json.dumps(turn.tool_calls)}]" if turn.tool_calls else ""
        safe_content = redact(turn.content)[0]  # rows written before 1.1.2 were not scanned on the way in
        line = f"Turn #{idx} [{turn.timestamp}] (ID: {turn.turn_id}) Role: {turn.role.upper()} => {safe_content}{tool_info}"
        turn_history_str.append(line)
        raw_char_count += len(turn.content)

    turns_block = "\n".join(turn_history_str)

    context_facts, facts_omitted = select_context_facts(
        existing_profile.facts, turns_block, max_context_facts
    )
    context_rules, rules_omitted = select_context_rules(
        existing_profile.rules, max_context_rules
    )
    # Send only the fields the model reads. A full model_dump carries
    # source_turn_ids — up to one UUID per turn in the window that produced the
    # fact — which dominated the payload while contributing nothing the model
    # uses to extract, compare or supersede.
    existing_facts_str = json.dumps(
        [
            {
                "entity": f.entity,
                "attribute": f.attribute,
                "value": f.value,
                "confidence": round(f.confidence, 2),
            }
            for f in context_facts
        ],
        indent=2,
    )
    existing_rules_str = json.dumps(
        [{"category": r.category, "rule": r.rule, "priority": r.priority} for r in context_rules],
        indent=2,
    )

    # Say plainly that this is a subset. Without it the model can read an
    # absent entity as "never recorded" and re-introduce it under a new name,
    # which is the entity drift the prompt works hard to prevent.
    scope_note = ""
    if facts_omitted or rules_omitted:
        scope_note = (
            f"\nNOTE: the graph below is the subset most relevant to these turns "
            f"({len(context_facts)} of {len(context_facts) + facts_omitted} active facts, "
            f"{len(context_rules)} of {len(context_rules) + rules_omitted} active rules). "
            f"An entity you cannot see here may still exist — do not assume absence, "
            f"and do not rename or duplicate an entity merely because it is not listed.\n"
        )

    user_prompt = f"""{scope_note}
### EXISTING CONSOLIDATED KNOWLEDGE GRAPH (ACTIVE FACTS):
{existing_facts_str}

### EXISTING ACTIVE OPERATIONAL RULES:
{existing_rules_str}

### UNCONSOLIDATED RAW EPISODIC TURNS TO CONSOLIDATE:
{turns_block}

Execute the REM sleep consolidation cycle now:
1. Prune all ephemeral noise & pleasantries.
2. Extract new/updated entity facts with source turn IDs.
3. Detect and resolve contradictions against existing facts.
4. Synthesize updated operational rules.
"""
    return user_prompt, raw_char_count


def _strictify(node: Any) -> None:
    """Make a Pydantic-generated JSON schema acceptable to strict structured-output
    modes: no titles/defaults, every object closed and fully required."""
    if isinstance(node, dict):
        node.pop("title", None)
        node.pop("default", None)
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)


def dream_output_json_schema() -> Dict[str, Any]:
    """Plain JSON Schema for DreamSynthesisOutput, normalised for strict modes
    (Anthropic output_config.format, OpenAI response_format json_schema)."""
    schema = DreamSynthesisOutput.model_json_schema()
    _strictify(schema)
    return schema


# Keys the Gemini response-schema transformer understands. Everything else is
# dropped rather than passed through: google-genai raises outright on `default`,
# and silently ignoring unknown keys is how a schema ends up "valid" while
# meaning something different from what we intended.
_GEMINI_SCHEMA_KEYS = frozenset({
    "type", "format", "description", "nullable", "enum",
    "items", "properties", "required", "anyOf",
})


def _inline_refs(node: Any, defs: Dict[str, Any]) -> Any:
    """Replace every $ref with its definition.

    Gemini is given a self-contained schema. The SDK can resolve $defs, but
    inlining keeps what we send identical to what we can assert on in tests.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name not in defs:
                raise DreamSynthesisError(
                    f"gemini schema build failed: unresolved $ref {ref!r}"
                )
            return _inline_refs(copy.deepcopy(defs[name]), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _gemini_prune(node: Any) -> Any:
    """Keep only Gemini-supported keys, and require every property.

    Requiring all properties matters precisely because the defaults are gone:
    if the model may omit `confidence`, a dream can come back structurally
    valid but missing the number we rely on. Asking for every field keeps the
    output deterministic across providers.
    """
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key not in _GEMINI_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                # The keys here are field names, not schema keywords — prune
                # the sub-schemas but never the names themselves.
                out[key] = {name: _gemini_prune(sub) for name, sub in value.items()}
            elif key == "required" and isinstance(value, list):
                out[key] = list(value)
            else:
                out[key] = _gemini_prune(value)
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(node, list):
        return [_gemini_prune(v) for v in node]
    return node


def gemini_response_schema() -> Dict[str, Any]:
    """The response schema sent to Gemini.

    Derived from DreamSynthesisOutput rather than written by hand, so adding a
    field to the model carries through here automatically instead of drifting.

    Why this exists: passing the Pydantic model directly as `response_schema`
    sends its field defaults along, and google-genai rejects any default with
    "Default value is not supported in the response schema for the Gemini API."
    That broke every dream from 2026-09-27 until this fix.
    """
    schema = DreamSynthesisOutput.model_json_schema()
    defs = schema.pop("$defs", {})
    return _gemini_prune(_inline_refs(schema, defs))


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1:] if first_newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped


def parse_dream_output(raw_text: str, provider: str) -> DreamSynthesisOutput:
    """
    The single parser every backend feeds. Raises DreamSynthesisError naming the
    provider on any failure; never returns a partial or fabricated result.
    """
    if raw_text is None or not str(raw_text).strip():
        raise DreamSynthesisError(f"{provider} dream synthesis returned an empty response")
    try:
        data = json.loads(_strip_code_fence(str(raw_text)))
        return DreamSynthesisOutput(**data)
    except Exception as exc:
        raise DreamSynthesisError(
            f"{provider} dream synthesis returned an unparseable response: {exc}"
        ) from exc
