"""
Pydantic schemas for the RemAgent autonomous zero-vector memory framework.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid


def generate_uuid() -> str:
    return str(uuid.uuid4())


def current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Fact(BaseModel):
    """
    Represents a discrete, extracted entity fact stored in the structured memory graph.
    Replaces amorphous vector chunks with high-signal, attributed knowledge.
    """
    id: str = Field(default_factory=generate_uuid, description="Unique fact identifier")
    entity: str = Field(..., description="Target entity, user, project, or domain object (e.g., 'User', 'ProjectAlpha')")
    attribute: str = Field(..., description="Specific property or relation (e.g., 'primary_language', 'auth_strategy')")
    value: Any = Field(..., description="Current ground-truth value (e.g., 'TypeScript', 'OAuth 2.0')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of the synthesized fact (0.0 - 1.0)")
    timestamp: Union[str, float] = Field(default_factory=current_utc_iso, description="ISO timestamp or unix epoch when fact was synthesized")
    source_turn_ids: List[str] = Field(default_factory=list, description="IDs of raw turns providing evidence for this fact")
    superseded_by: Optional[str] = Field(default=None, description="Fact ID that invalidated/superseded this record")
    is_active: bool = Field(default=True, description="Whether this fact is currently active or historically superseded")


class OperationalRule(BaseModel):
    """
    Synthesized user preferences, operational directives, and task heuristics
    discovered and consolidated during REM sleep cycles.
    """
    id: str = Field(default_factory=generate_uuid, description="Unique rule identifier")
    category: Literal[
        "user_preference",
        "coding_standard",
        "architecture_heuristic",
        "operational_directive",
        "domain_constraint"
    ] = Field(..., description="Categorical classification of the operational rule")
    rule: str = Field(..., description="Concise, actionable rule statement")
    rationale: str = Field(..., description="Reasoning or empirical turn context justifying this directive")
    priority: int = Field(default=3, ge=1, le=5, description="Priority weight (1 = critical/unbreakable, 5 = minor guideline)")
    is_active: bool = Field(default=True, description="Whether the rule is currently active")
    updated_at: str = Field(default_factory=current_utc_iso, description="Last update timestamp")


class ContradictionResolution(BaseModel):
    """
    Captures an explicit contradiction resolution during memory synthesis.
    Demonstrates zero-vector deterministic supersession over vector drift.
    """
    prior_fact_id: Optional[str] = Field(default=None, description="Identifier of obsolete fact")
    entity: str = Field(..., description="Subject entity")
    attribute: str = Field(..., description="Subject attribute")
    prior_value: Any = Field(..., description="Outdated value")
    new_value: Any = Field(..., description="New ground-truth value")
    resolution_reasoning: str = Field(..., description="Reasoning why the new observation overrides the prior state")


class RawTurnLog(BaseModel):
    """
    Captures raw, unconsolidated user/agent interaction turns in episodic memory buffer.
    """
    turn_id: str = Field(default_factory=generate_uuid, description="Unique turn identifier")
    session_id: str = Field(default="default_session", description="Conversation session or agent task ID")
    role: Literal["user", "assistant", "system", "tool"] = Field(..., description="Actor role for the turn")
    content: str = Field(..., description="Raw text content or conversational message")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool execution requests or outputs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary context metadata (tokens, latency, etc.)")
    timestamp: Union[str, float] = Field(default_factory=current_utc_iso, description="Turn recording timestamp")
    is_consolidated: bool = Field(default=False, description="Whether this turn has been processed by a dream cycle")

    @field_validator("content", mode="before")
    @classmethod
    def _hide_secrets(cls, value: Any) -> Any:
        """Secret hiding (remagent/sensitive.py): keys, tokens, passwords and card numbers never enter the buffer."""
        if isinstance(value, str) and value:
            from .sensitive import redact
            clean, found = redact(value)
            if found:
                return clean
        return value


class DreamConsolidationResult(BaseModel):
    """
    Output of an autonomous REM sleep consolidation run.
    Contains clean structured updates, pruned noise metrics, and contradiction resolutions.
    """
    run_id: str = Field(default_factory=generate_uuid, description="Consolidation run ID")
    added_facts: List[Fact] = Field(default_factory=list, description="New or updated discrete facts extracted")
    updated_rules: List[OperationalRule] = Field(default_factory=list, description="New or updated operational directives")
    contradiction_resolutions: List[ContradictionResolution] = Field(default_factory=list, description="Contradictions resolved")
    pruned_noise_count: int = Field(default=0, ge=0, description="Count of noisy, transient, or filler items discarded")
    pruned_noise_reasons: List[str] = Field(default_factory=list, description="Summaries of pruned noise classes")
    reasoning_summary: str = Field(..., description="Cognitive summary of what was learned and pruned during REM sleep")
    consolidated_turn_ids: List[str] = Field(default_factory=list, description="IDs of raw turns processed and archived")
    timestamp: Union[str, float] = Field(default_factory=current_utc_iso, description="Consolidation execution timestamp")
    estimated_token_savings: int = Field(default=0, description="Estimated prompt token overhead eliminated vs raw history")
    is_fallback: bool = Field(default=False, description="True if this result was NOT produced by a real, successful synthesis call; such results must never be persisted")
    error: Optional[str] = Field(default=None, description="Error message when the consolidation run failed; None on success")


class MemoryProfile(BaseModel):
    """
    Complete consolidated zero-vector memory state for an agent or user.
    """
    agent_id: str = Field(default="default_agent", description="Identifier of target agent or user persona")
    facts: List[Fact] = Field(default_factory=list, description="Active entity facts")
    rules: List[OperationalRule] = Field(default_factory=list, description="Active operational rules")
    audit_history: List[DreamConsolidationResult] = Field(default_factory=list, description="Consolidation run logs")
    total_pruned_turns: int = Field(default=0, description="Lifetime count of pruned turns")
    last_dream_at: Optional[str] = Field(default=None, description="Timestamp of the most recent consolidation")
