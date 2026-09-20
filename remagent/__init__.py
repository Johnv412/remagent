"""
RemAgent: Autonomous Zero-Vector Memory Framework for AI Agents.
Built on biological sleep/REM consolidation principles; consolidation runs on
Gemini, Anthropic, OpenAI-compatible, or xAI (see remagent.engine.providers).
"""

from remagent.schemas import (
    Fact,
    OperationalRule,
    DreamConsolidationResult,
    RawTurnLog,
    MemoryProfile,
    ContradictionResolution,
)
from remagent.storage.base import StorageAdapter
from remagent.storage.sqlite import SQLiteStorageAdapter
from remagent.storage.firestore import FirestoreStorageAdapter
from remagent.engine.synthesizer import DreamSynthesizer
from remagent.daemon import DreamDaemon
from remagent.governor import TokenBudgetGovernor
from remagent.decay import MemoryDecayEngine
from remagent.integrations.hermes import HermesMemoryConnector, RemAgentTool

__version__ = "1.1.1"
__all__ = [
    "Fact",
    "OperationalRule",
    "DreamConsolidationResult",
    "RawTurnLog",
    "MemoryProfile",
    "ContradictionResolution",
    "StorageAdapter",
    "SQLiteStorageAdapter",
    "FirestoreStorageAdapter",
    "DreamSynthesizer",
    "DreamDaemon",
    "TokenBudgetGovernor",
    "MemoryDecayEngine",
    "HermesMemoryConnector",
    "RemAgentTool",
]
