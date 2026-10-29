"""
SQLite storage adapter implementation for RemAgent.
Designed for zero-dependency standalone and local agent deployments.
"""

import asyncio
import json
import os
import sqlite3
import sys
from typing import List, Optional
from remagent.schemas import (
    Fact,
    OperationalRule,
    RawTurnLog,
    MemoryProfile,
    DreamConsolidationResult,
)
from remagent.storage.base import StorageAdapter


class SQLiteStorageAdapter(StorageAdapter):
    """
    Asynchronous SQLite adapter for RemAgent local storage.
    Uses standard library sqlite3 wrapped in asynchronous worker threads.
    """

    def __init__(self, db_path: str = "remagent_memory.db", database_path: Optional[str] = None):
        self.db_path = database_path if database_path is not None else db_path
        self._lock = asyncio.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _sync_initialize(self) -> None:
        with self._get_connection() as conn:
            # Raw turns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    metadata_json TEXT,
                    timestamp TEXT NOT NULL,
                    is_consolidated INTEGER NOT NULL DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_unconsolidated ON raw_turns(session_id, is_consolidated);")

            # Structured Memory Profile table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_profiles (
                    agent_id TEXT PRIMARY KEY,
                    facts_json TEXT NOT NULL DEFAULT '[]',
                    rules_json TEXT NOT NULL DEFAULT '[]',
                    total_pruned_turns INTEGER NOT NULL DEFAULT 0,
                    last_dream_at TEXT
                );
            """)

            # Dream Audit History table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dream_audit_history (
                    run_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    added_facts_json TEXT NOT NULL,
                    updated_rules_json TEXT NOT NULL,
                    contradiction_resolutions_json TEXT NOT NULL,
                    pruned_noise_count INTEGER NOT NULL,
                    pruned_noise_reasons_json TEXT NOT NULL,
                    reasoning_summary TEXT NOT NULL,
                    consolidated_turn_ids_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    estimated_token_savings INTEGER NOT NULL DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent ON dream_audit_history(agent_id, timestamp);")
            conn.commit()

    async def initialize(self) -> None:
        # Creating a brand-new empty database must never be silent: a user
        # pointed at the wrong path would otherwise see "no memories" and
        # have no idea why. Silent empty is a fake-empty.
        is_new = self.db_path != ":memory:" and not os.path.exists(self.db_path)
        async with self._lock:
            await asyncio.to_thread(self._sync_initialize)
        if is_new:
            print(
                f"🆕 [RemAgent] Created a NEW EMPTY database at "
                f"{os.path.abspath(self.db_path)} — if you expected existing "
                f"memories, check your --db flag or the REMAGENT_DB env var.",
                file=sys.stderr,
            )

    def _sync_save_turn(self, turn: RawTurnLog) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO raw_turns (
                    turn_id, session_id, role, content, tool_calls_json, metadata_json, timestamp, is_consolidated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                turn.turn_id,
                turn.session_id,
                turn.role,
                turn.content,
                json.dumps(turn.tool_calls) if turn.tool_calls is not None else None,
                json.dumps(turn.metadata),
                turn.timestamp,
                1 if turn.is_consolidated else 0,
            ))
            conn.commit()
            return cur.rowcount == 1  # 0 = a row with this turn_id already existed and was left alone

    async def save_turn(self, turn: RawTurnLog) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._sync_save_turn, turn)

    def _sync_get_unconsolidated_turns(self, session_id: Optional[str], limit: int) -> List[RawTurnLog]:
        with self._get_connection() as conn:
            if session_id:
                cursor = conn.execute("""
                    SELECT * FROM raw_turns
                    WHERE session_id = ? AND is_consolidated = 0
                    ORDER BY timestamp ASC
                    LIMIT ?;
                """, (session_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM raw_turns
                    WHERE is_consolidated = 0
                    ORDER BY timestamp ASC
                    LIMIT ?;
                """, (limit,))
            
            rows = cursor.fetchall()
            turns: List[RawTurnLog] = []
            for row in rows:
                tool_calls = json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else None
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                turns.append(
                    RawTurnLog(
                        turn_id=row["turn_id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        tool_calls=tool_calls,
                        metadata=metadata,
                        timestamp=row["timestamp"],
                        is_consolidated=bool(row["is_consolidated"]),
                    )
                )
            return turns

    async def get_unconsolidated_turns(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> List[RawTurnLog]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_unconsolidated_turns, session_id, limit)

    def _sync_mark_turns_consolidated(self, turn_ids: List[str]) -> None:
        if not turn_ids:
            return
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in turn_ids)
            conn.execute(f"""
                UPDATE raw_turns
                SET is_consolidated = 1
                WHERE turn_id IN ({placeholders});
            """, turn_ids)
            conn.commit()

    async def mark_turns_consolidated(self, turn_ids: List[str]) -> None:
        if not turn_ids:
            return
        async with self._lock:
            await asyncio.to_thread(self._sync_mark_turns_consolidated, turn_ids)

    def _sync_load_memory_profile(self, agent_id: str) -> MemoryProfile:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_profiles WHERE agent_id = ?;
            """, (agent_id,))
            row = cursor.fetchone()
            
            if not row:
                return MemoryProfile(agent_id=agent_id, facts=[], rules=[], audit_history=[])

            facts_data = json.loads(row["facts_json"])
            rules_data = json.loads(row["rules_json"])
            facts = [Fact(**f) for f in facts_data]
            rules = [OperationalRule(**r) for r in rules_data]

            # Also load recent audit history
            audit_cursor = conn.execute("""
                SELECT * FROM dream_audit_history
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 20;
            """, (agent_id,))
            audit_rows = audit_cursor.fetchall()
            audit_history = []
            for a_row in audit_rows:
                audit_history.append(
                    DreamConsolidationResult(
                        run_id=a_row["run_id"],
                        added_facts=[Fact(**f) for f in json.loads(a_row["added_facts_json"])],
                        updated_rules=[OperationalRule(**r) for r in json.loads(a_row["updated_rules_json"])],
                        contradiction_resolutions=json.loads(a_row["contradiction_resolutions_json"]),
                        pruned_noise_count=a_row["pruned_noise_count"],
                        pruned_noise_reasons=json.loads(a_row["pruned_noise_reasons_json"]),
                        reasoning_summary=a_row["reasoning_summary"],
                        consolidated_turn_ids=json.loads(a_row["consolidated_turn_ids_json"]),
                        timestamp=a_row["timestamp"],
                        estimated_token_savings=a_row["estimated_token_savings"],
                    )
                )

            return MemoryProfile(
                agent_id=agent_id,
                facts=facts,
                rules=rules,
                audit_history=audit_history,
                total_pruned_turns=row["total_pruned_turns"],
                last_dream_at=row["last_dream_at"],
            )

    async def load_memory_profile(self, agent_id: str = "default_agent") -> MemoryProfile:
        async with self._lock:
            return await asyncio.to_thread(self._sync_load_memory_profile, agent_id)

    def _sync_save_memory_profile(self, profile: MemoryProfile) -> None:
        with self._get_connection() as conn:
            facts_json = json.dumps([f.model_dump() for f in profile.facts])
            rules_json = json.dumps([r.model_dump() for r in profile.rules])
            conn.execute("""
                INSERT OR REPLACE INTO memory_profiles (
                    agent_id, facts_json, rules_json, total_pruned_turns, last_dream_at
                ) VALUES (?, ?, ?, ?, ?);
            """, (
                profile.agent_id,
                facts_json,
                rules_json,
                profile.total_pruned_turns,
                profile.last_dream_at,
            ))
            conn.commit()

    async def save_memory_profile(self, profile: MemoryProfile) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_save_memory_profile, profile)

    def _sync_record_consolidation_audit(self, result: DreamConsolidationResult, agent_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO dream_audit_history (
                    run_id, agent_id, added_facts_json, updated_rules_json,
                    contradiction_resolutions_json, pruned_noise_count,
                    pruned_noise_reasons_json, reasoning_summary,
                    consolidated_turn_ids_json, timestamp, estimated_token_savings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                result.run_id,
                agent_id,
                json.dumps([f.model_dump() for f in result.added_facts]),
                json.dumps([r.model_dump() for r in result.updated_rules]),
                json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in result.contradiction_resolutions]),
                result.pruned_noise_count,
                json.dumps(result.pruned_noise_reasons),
                result.reasoning_summary,
                json.dumps(result.consolidated_turn_ids),
                result.timestamp,
                result.estimated_token_savings,
            ))
            conn.commit()

    async def record_consolidation_audit(
        self, result: DreamConsolidationResult, agent_id: str = "default_agent"
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_record_consolidation_audit, result, agent_id)

    async def close(self) -> None:
        """No persistent pool needed for worker threads."""
        pass
