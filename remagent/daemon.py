"""
DreamDaemon: Autonomous background worker for RemAgent.
Monitors agent activity, triggers biological sleep / REM consolidation cycles during idle windows,
and commits consolidated structured memory to storage.
"""

import asyncio
import logging
import time
from typing import Callable, Coroutine, List, Optional
from datetime import datetime, timezone

from remagent.schemas import (
    Fact,
    MemoryProfile,
    DreamConsolidationResult,
    current_utc_iso,
)
from remagent.storage.base import StorageAdapter
from remagent.engine.synthesizer import DreamSynthesizer

logger = logging.getLogger("remagent.daemon")


class ConsolidationBusyError(RuntimeError):
    """
    Raised when a consolidation cycle is already in progress. This is NOT
    "memory is up to date": unconsolidated turns remain queued and untouched.
    Callers must report busy as busy and retry after the running cycle ends.
    """


class DreamDaemon:
    """
    Background daemon that runs autonomous REM consolidation passes
    when the agent is idle or when explicitly triggered.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        synthesizer: Optional[DreamSynthesizer] = None,
        agent_id: str = "default_agent",
        idle_threshold_seconds: float = 30.0,
        check_interval_seconds: float = 5.0,
        min_turns_to_dream: int = 1,
        on_dream_completed: Optional[Callable[[DreamConsolidationResult], Coroutine]] = None,
    ):
        self.storage = storage
        self.synthesizer = synthesizer or DreamSynthesizer()
        self.agent_id = agent_id
        self.idle_threshold_seconds = idle_threshold_seconds
        self.check_interval_seconds = check_interval_seconds
        self.min_turns_to_dream = min_turns_to_dream
        self.on_dream_completed = on_dream_completed

        self._last_activity_time: float = time.time()
        self._is_running: bool = False
        self._is_dreaming: bool = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def record_activity(self) -> None:
        """Call whenever user or agent interacts to reset idle countdown."""
        self._last_activity_time = time.time()

    @property
    def is_idle(self) -> bool:
        return (time.time() - self._last_activity_time) >= self.idle_threshold_seconds

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_activity_time

    @property
    def is_dreaming(self) -> bool:
        return self._is_dreaming

    async def start(self) -> None:
        """Start the background daemon loop."""
        if self._is_running:
            return
        self._is_running = True
        await self.storage.initialize()
        self._task = asyncio.create_task(self._daemon_loop())
        logger.info(f"RemAgent DreamDaemon started for agent '{self.agent_id}' (idle threshold: {self.idle_threshold_seconds}s)")

    async def stop(self) -> None:
        """Stop background daemon gracefully."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"RemAgent DreamDaemon stopped for agent '{self.agent_id}'")

    async def _daemon_loop(self) -> None:
        while self._is_running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                if not self._is_running:
                    break

                if self.is_idle and not self._is_dreaming:
                    # Check if there are unconsolidated turns
                    turns = await self.storage.get_unconsolidated_turns(limit=50)
                    if len(turns) >= self.min_turns_to_dream:
                        logger.info(f"Idle detected ({self.idle_seconds:.1f}s). Triggering REM consolidation for {len(turns)} turns.")
                        await self.consolidate_now()

            except asyncio.CancelledError:
                break
            except ConsolidationBusyError:
                logger.debug("Skipped scheduled consolidation: a cycle is already in progress.")
            except Exception as e:
                logger.error(f"Error in DreamDaemon loop: {e}", exc_info=True)

    async def consolidate_now(self) -> Optional[DreamConsolidationResult]:
        """
        Forces an immediate REM consolidation pass regardless of idle timer.

        Returns None ONLY when there are no unconsolidated turns (memory is
        genuinely up to date). Raises ConsolidationBusyError when another
        cycle is already running — turns remain queued in that case.
        """
        if self._is_dreaming or self._lock.locked():
            raise ConsolidationBusyError(
                "A REM consolidation cycle is already in progress; "
                "unconsolidated turns remain queued. Retry shortly."
            )

        async with self._lock:
            self._is_dreaming = True
            try:
                # 1. Fetch pending turns
                unconsolidated = await self.storage.get_unconsolidated_turns(limit=100)
                if not unconsolidated:
                    logger.debug("No turns to consolidate.")
                    return None

                # 2. Load existing profile
                profile = await self.storage.load_memory_profile(agent_id=self.agent_id)

                # 3. Run Gemini Dream Synthesizer
                result = await self.synthesizer.consolidate_window(
                    unconsolidated_turns=unconsolidated,
                    existing_profile=profile,
                )

                # A failed or fabricated synthesis must never be persisted:
                # no facts, no profile update, and turns stay unconsolidated
                # so they are reprocessed on the next dream cycle.
                if result.is_fallback or result.error:
                    raise RuntimeError(
                        f"Dream synthesis returned a failure result "
                        f"(is_fallback={result.is_fallback}, error={result.error}); "
                        f"refusing to persist memory or mark turns consolidated."
                    )

                # 4. Resolve contradictions. The replacing fact is determined
                # (or materialized) BEFORE the existing graph is touched, so
                # every deactivated fact records the ID of the fact that
                # superseded it. The daemon must stay correct even when the
                # model returns an unexpected shape: an update must never
                # leave the entity/attribute without an active fact.
                def _find_replacement(entity: str, attribute: str) -> Optional[Fact]:
                    for f in result.added_facts:
                        if (
                            f.is_active
                            and f.entity.lower() == entity.lower()
                            and f.attribute.lower() == attribute.lower()
                        ):
                            return f
                    return None

                for resolution in result.contradiction_resolutions:
                    replacement = _find_replacement(resolution.entity, resolution.attribute)
                    if replacement is None:
                        logger.warning(
                            f"Synthesizer resolved contradiction on "
                            f"{resolution.entity}.{resolution.attribute} without emitting a "
                            f"replacement in added_facts; materializing active fact from "
                            f"new_value={resolution.new_value!r}."
                        )
                        replacement = Fact(
                            entity=resolution.entity,
                            attribute=resolution.attribute,
                            value=resolution.new_value,
                            confidence=1.0,
                            source_turn_ids=list(result.consolidated_turn_ids),
                            is_active=True,
                        )
                        result.added_facts.append(replacement)

                    # Deactivate the pre-existing facts, pointing each at the
                    # fact that replaced it (added_facts are not yet appended,
                    # so the replacement itself cannot be deactivated here).
                    for fact in profile.facts:
                        if (
                            fact.entity.lower() == resolution.entity.lower()
                            and fact.attribute.lower() == resolution.attribute.lower()
                            and fact.is_active
                        ):
                            fact.is_active = False
                            fact.superseded_by = replacement.id

                # 5. Append new active facts
                for new_fact in result.added_facts:
                    profile.facts.append(new_fact)

                def _has_active_fact(entity: str, attribute: str) -> bool:
                    return any(
                        f.is_active
                        and f.entity.lower() == entity.lower()
                        and f.attribute.lower() == attribute.lower()
                        for f in profile.facts
                    )

                # 5c. Invariant: after applying contradictions, every resolved
                # entity/attribute must have an active fact. A deactivated fact
                # with no active replacement means memory was erased, not
                # updated — fail the run instead of committing that state.
                for resolution in result.contradiction_resolutions:
                    if not _has_active_fact(resolution.entity, resolution.attribute):
                        raise RuntimeError(
                            f"Memory-erasure invariant violated: "
                            f"{resolution.entity}.{resolution.attribute} was superseded but has "
                            f"no active replacement fact. Refusing to persist; turns remain "
                            f"unconsolidated for retry."
                        )

                # 6. Update or append operational rules
                # The map is updated as we go, not just seeded once. A single
                # dream can return the same rule many times — one on
                # 2026-09-26 emitted 430 rules containing roughly 32 copies
                # each of a dozen texts — and seeding the map only from the
                # existing profile let every repeat inside that batch through.
                # Matching is on normalised text so trivial whitespace or case
                # differences do not read as a new rule.
                def _rule_key(text: str) -> str:
                    return " ".join(str(text).lower().split())

                existing_rule_map = {_rule_key(r.rule): r for r in profile.rules}
                for new_rule in result.updated_rules:
                    key = _rule_key(new_rule.rule)
                    existing = existing_rule_map.get(key)
                    if existing is not None:
                        # Re-asserting a rule refreshes it rather than cloning
                        # it; the record keeps its original id so anything
                        # referring to it still resolves.
                        existing.priority = new_rule.priority
                        existing.rationale = new_rule.rationale
                        existing.updated_at = current_utc_iso()
                        if not existing.is_active:
                            existing.is_active = True
                    else:
                        profile.rules.append(new_rule)
                        existing_rule_map[key] = new_rule

                # 7. Update profile metadata
                profile.total_pruned_turns += result.pruned_noise_count
                profile.last_dream_at = result.timestamp

                # 8. Save updated profile & mark turns consolidated
                await self.storage.save_memory_profile(profile)
                await self.storage.mark_turns_consolidated(result.consolidated_turn_ids)
                await self.storage.record_consolidation_audit(result, agent_id=self.agent_id)

                logger.info(
                    f"REM consolidation finished. Added {len(result.added_facts)} facts, "
                    f"updated {len(result.updated_rules)} rules, pruned {result.pruned_noise_count} noise items. "
                    f"Estimated savings: {result.estimated_token_savings} tokens."
                )

                # Reset activity clock so it doesn't dream in a tight loop
                self.record_activity()

                if self.on_dream_completed:
                    try:
                        await self.on_dream_completed(result)
                    except Exception as cb_err:
                        logger.warning(f"Error in on_dream_completed callback: {cb_err}")

                return result

            except Exception as e:
                logger.error(f"Failed to execute REM consolidation: {e}", exc_info=True)
                raise
            finally:
                self._is_dreaming = False
