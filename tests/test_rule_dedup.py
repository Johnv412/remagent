"""
Rule-growth regression tests.

A single dream on 2026-09-26 emitted 430 rules, roughly 32 identical copies
each of a dozen texts, and the profile accepted all of them. That one batch is
most of how the brain reached 744 active rules — which in turn is what made
recall refuse to build at all.

The cause was narrow: the daemon seeded its dedup map from the existing profile
and never updated it while appending, so duplicates *within* one batch were
invisible to the check. These tests pin that closed from both directions —
repeats inside a batch, and repeats across successive dreams.
"""

import asyncio
import os
import tempfile
import unittest

from remagent.daemon import DreamDaemon
from remagent.schemas import (
    DreamConsolidationResult,
    OperationalRule,
    RawTurnLog,
    current_utc_iso,
    generate_uuid,
)
from remagent.storage.sqlite import SQLiteStorageAdapter

RULE = "Always run the test suite before any deploy, no exceptions."


def rule(text=RULE, priority=1, category="operational_directive"):
    return OperationalRule(category=category, rule=text, rationale="synthesized", priority=priority)


class _StubSynthesizer:
    """Returns a fixed result, so the test measures merge behaviour only."""

    def __init__(self, rules):
        self._rules = rules

    async def consolidate_window(self, unconsolidated_turns, existing_profile):
        return DreamConsolidationResult(
            run_id=generate_uuid(),
            added_facts=[],
            updated_rules=list(self._rules),
            contradiction_resolutions=[],
            pruned_noise_count=0,
            pruned_noise_reasons=[],
            reasoning_summary="stub consolidation",
            consolidated_turn_ids=[t.turn_id for t in unconsolidated_turns],
            timestamp=current_utc_iso(),
            estimated_token_savings=0,
        )


class TestRuleDedup(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".dedup.db")
        os.close(fd)
        self.storage = SQLiteStorageAdapter(db_path=self.db)
        await self.storage.initialize()

    async def asyncTearDown(self):
        await self.storage.close()
        if os.path.exists(self.db):
            os.remove(self.db)

    async def _dream(self, rules):
        await self.storage.save_turn(RawTurnLog(role="user", content="a substantive turn"))
        daemon = DreamDaemon(
            storage=self.storage, synthesizer=_StubSynthesizer(rules), agent_id="a"
        )
        return await daemon.consolidate_now()

    async def _active_rules(self):
        profile = await self.storage.load_memory_profile("a")
        return [r for r in profile.rules if r.is_active]

    async def test_duplicates_within_one_batch_collapse(self):
        """The exact 2026-09-26 shape: one dream returning the same rule many
        times must yield one rule, not 32."""
        await self._dream([rule() for _ in range(32)])
        active = await self._active_rules()
        self.assertEqual(len(active), 1, f"expected 1 rule, got {len(active)}")

    async def test_repeated_dreams_do_not_accumulate_copies(self):
        """Re-asserting the same doctrine across sessions is normal and must
        not grow the rule set each time."""
        for _ in range(5):
            await self._dream([rule()])
        active = await self._active_rules()
        self.assertEqual(len(active), 1)

    async def test_whitespace_and_case_variants_are_the_same_rule(self):
        await self._dream([
            rule(RULE),
            rule(RULE.upper()),
            rule("  Always run the test   suite before any deploy, no exceptions.  "),
        ])
        active = await self._active_rules()
        self.assertEqual(len(active), 1)

    async def test_distinct_rules_are_all_kept(self):
        """Dedup must not be so eager that it loses real doctrine."""
        await self._dream([
            rule("Never commit secrets to git."),
            rule("Prefer configuration changes over code edits."),
            rule("Archive stale code instead of deleting it."),
        ])
        active = await self._active_rules()
        self.assertEqual(len(active), 3)

    async def test_reassertion_refreshes_in_place_and_keeps_the_id(self):
        """A re-asserted rule updates rather than clones, so anything holding
        its id still resolves."""
        await self._dream([rule(priority=3)])
        first = (await self._active_rules())[0]
        original_id = first.id

        await self._dream([rule(priority=1)])
        active = await self._active_rules()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, original_id, "re-assertion must not mint a new id")
        self.assertEqual(active[0].priority, 1, "priority should reflect the latest assertion")

    async def test_a_large_batch_stays_proportional_to_its_distinct_rules(self):
        """430 rules in, 10 distinct texts, 10 rules out."""
        batch = []
        for i in range(10):
            batch += [rule(f"Distinct directive number {i}.")] * 43
        self.assertEqual(len(batch), 430)
        await self._dream(batch)
        active = await self._active_rules()
        self.assertEqual(len(active), 10)


if __name__ == "__main__":
    unittest.main()
