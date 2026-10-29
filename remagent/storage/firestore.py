"""
Google Cloud Firestore storage adapter implementation for RemAgent.
Designed for cloud-native and enterprise multi-agent deployments.
"""

import json
from typing import Any, Dict, List, Optional
from remagent.schemas import (
    Fact,
    OperationalRule,
    RawTurnLog,
    MemoryProfile,
    DreamConsolidationResult,
)
from remagent.storage.base import StorageAdapter


class FirestoreStorageAdapter(StorageAdapter):
    """
    Asynchronous Google Cloud Firestore adapter for RemAgent.
    Stores raw episodic turn logs and synthesized zero-vector memory graphs.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        database_id: str = "(default)",
        turns_collection: str = "remagent_raw_turns",
        profiles_collection: str = "remagent_memory_profiles",
        audit_collection: str = "remagent_dream_audits",
    ):
        self.project_id = project_id
        self.database_id = database_id
        self.turns_collection = turns_collection
        self.profiles_collection = profiles_collection
        self.audit_collection = audit_collection
        self._db = None

    async def _get_client(self):
        if self._db is None:
            try:
                from google.cloud import firestore  # type: ignore
                self._db = firestore.AsyncClient(
                    project=self.project_id, database=self.database_id
                )
            except ImportError:
                raise ImportError(
                    "google-cloud-firestore package is required to use FirestoreStorageAdapter. "
                    "Install it via `pip install remagent[firestore]` or `pip install google-cloud-firestore`."
                )
        return self._db

    async def initialize(self) -> None:
        """Verifies Firestore client initialization."""
        await self._get_client()

    async def save_turn(self, turn: RawTurnLog) -> bool:
        db = await self._get_client()
        doc_ref = db.collection(self.turns_collection).document(turn.turn_id)
        data = turn.model_dump()
        try:
            await doc_ref.create(data)  # fails if the document exists: a retried log never overwrites
        except Exception as exc:  # google.api_core.exceptions.AlreadyExists, without importing the SDK here
            if type(exc).__name__ == "AlreadyExists" or "already exists" in str(exc).lower():
                return False
            raise
        return True

    async def get_unconsolidated_turns(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> List[RawTurnLog]:
        db = await self._get_client()
        query = (
            db.collection(self.turns_collection)
            .where("is_consolidated", "==", False)
        )
        if session_id:
            query = query.where("session_id", "==", session_id)

        query = query.order_by("timestamp").limit(limit)
        docs = query.stream()
        turns: List[RawTurnLog] = []
        async for doc in docs:
            turns.append(RawTurnLog(**doc.to_dict()))
        return turns

    async def mark_turns_consolidated(self, turn_ids: List[str]) -> None:
        if not turn_ids:
            return
        db = await self._get_client()
        batch = db.batch()
        for turn_id in turn_ids:
            doc_ref = db.collection(self.turns_collection).document(turn_id)
            batch.update(doc_ref, {"is_consolidated": True})
        await batch.commit()

    async def load_memory_profile(self, agent_id: str = "default_agent") -> MemoryProfile:
        db = await self._get_client()
        doc_ref = db.collection(self.profiles_collection).document(agent_id)
        snapshot = await doc_ref.get()

        if not snapshot.exists:
            return MemoryProfile(agent_id=agent_id, facts=[], rules=[], audit_history=[])

        data = snapshot.to_dict() or {}
        facts = [Fact(**f) for f in data.get("facts", [])]
        rules = [OperationalRule(**r) for r in data.get("rules", [])]

        # Retrieve recent audits
        audit_query = (
            db.collection(self.audit_collection)
            .where("agent_id", "==", agent_id)
            .order_by("timestamp", direction="DESCENDING")
            .limit(20)
        )
        audit_docs = audit_query.stream()
        audit_history: List[DreamConsolidationResult] = []
        async for a_doc in audit_docs:
            a_data = a_doc.to_dict()
            audit_history.append(DreamConsolidationResult(**a_data))

        return MemoryProfile(
            agent_id=agent_id,
            facts=facts,
            rules=rules,
            audit_history=audit_history,
            total_pruned_turns=data.get("total_pruned_turns", 0),
            last_dream_at=data.get("last_dream_at"),
        )

    async def save_memory_profile(self, profile: MemoryProfile) -> None:
        db = await self._get_client()
        doc_ref = db.collection(self.profiles_collection).document(profile.agent_id)
        data = {
            "agent_id": profile.agent_id,
            "facts": [f.model_dump() for f in profile.facts],
            "rules": [r.model_dump() for r in profile.rules],
            "total_pruned_turns": profile.total_pruned_turns,
            "last_dream_at": profile.last_dream_at,
        }
        await doc_ref.set(data, merge=True)

    async def record_consolidation_audit(
        self, result: DreamConsolidationResult, agent_id: str = "default_agent"
    ) -> None:
        db = await self._get_client()
        doc_ref = db.collection(self.audit_collection).document(result.run_id)
        data = result.model_dump()
        data["agent_id"] = agent_id
        await doc_ref.set(data)

    async def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
