"""
Abstract base storage adapter interface for RemAgent.
Defines contracts for saving/loading raw turn logs, active facts, rules, and memory profiles.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from remagent.schemas import (
    Fact,
    OperationalRule,
    RawTurnLog,
    MemoryProfile,
    DreamConsolidationResult,
)


class StorageAdapter(ABC):
    """
    Abstract interface for episodic turn logs and synthesized memory profile storage.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend connection, tables, indices, or collections."""
        pass

    @abstractmethod
    async def save_turn(self, turn: RawTurnLog) -> bool:
        """Append a raw interaction turn to the unconsolidated buffer.
        Returns True when the turn was stored now, False when a turn with this turn_id already existed (the stored
        row is left untouched: a retried log never overwrites or resets consolidation)."""
        return True

    @abstractmethod
    async def get_unconsolidated_turns(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> List[RawTurnLog]:
        """Fetch pending turns that have not yet undergone REM consolidation."""
        pass

    @abstractmethod
    async def mark_turns_consolidated(self, turn_ids: List[str]) -> None:
        """Mark a batch of turn IDs as successfully consolidated."""
        pass

    @abstractmethod
    async def load_memory_profile(self, agent_id: str = "default_agent") -> MemoryProfile:
        """Retrieve the synthesized zero-vector memory profile for an agent."""
        pass

    @abstractmethod
    async def save_memory_profile(self, profile: MemoryProfile) -> None:
        """Persist or overwrite the updated structured memory profile."""
        pass

    @abstractmethod
    async def record_consolidation_audit(
        self, result: DreamConsolidationResult, agent_id: str = "default_agent"
    ) -> None:
        """Append an audit log record for a completed dream consolidation cycle."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Gracefully close connections or resource pools."""
        pass
