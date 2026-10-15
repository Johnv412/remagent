"""
Unit and Integration Test Suite for RemAgent.
Covers Schemas, SQLite Storage, Governor, Decay, and Daemon Logic.
"""

import asyncio
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from remagent.schemas import Fact, OperationalRule, RawTurnLog, MemoryProfile, DreamConsolidationResult
from remagent.storage.sqlite import SQLiteStorageAdapter
from remagent.governor import GovernorBudgetError, TokenBudgetGovernor
from remagent.decay import MemoryDecayEngine


class TestRemAgentSuite(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.storage = SQLiteStorageAdapter(database_path=self.db_path)
        await self.storage.initialize()

    async def asyncTearDown(self):
        await self.storage.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def test_turn_lifecycle_and_consolidation(self):
        turn1 = RawTurnLog(
            turn_id="turn_001",
            session_id="test_session",
            role="user",
            content="Please use PostgreSQL 16 on Cloud SQL.",
            timestamp=time.time(),
        )
        turn2 = RawTurnLog(
            turn_id="turn_002",
            session_id="test_session",
            role="assistant",
            content="Understood, configured PostgreSQL 16.",
            timestamp=time.time() + 1,
        )

        await self.storage.save_turn(turn1)
        await self.storage.save_turn(turn2)

        unconsolidated = await self.storage.get_unconsolidated_turns(session_id="test_session")
        self.assertEqual(len(unconsolidated), 2)

        # Mark consolidated
        await self.storage.mark_turns_consolidated(["turn_001", "turn_002"])
        remaining = await self.storage.get_unconsolidated_turns(session_id="test_session")
        self.assertEqual(len(remaining), 0)

    async def test_memory_profile_persistence(self):
        fact = Fact(
            id="f_1",
            entity="Architecture",
            attribute="database",
            value="PostgreSQL 16",
            confidence=0.95,
            source_turn_ids=["turn_001"],
        )
        rule = OperationalRule(
            id="r_1",
            category="coding_standard",
            rule="Always use strict TypeScript types",
            rationale="Enforces code safety",
            priority=1,
        )

        profile = MemoryProfile(
            agent_id="test_agent",
            facts=[fact],
            rules=[rule],
        )

        await self.storage.save_memory_profile(profile)
        loaded = await self.storage.load_memory_profile("test_agent")

        self.assertEqual(len(loaded.facts), 1)
        self.assertEqual(loaded.facts[0].attribute, "database")
        self.assertEqual(loaded.facts[0].value, "PostgreSQL 16")
        self.assertEqual(len(loaded.rules), 1)
        self.assertEqual(loaded.rules[0].priority, 1)

    def test_token_budget_governor(self):
        governor = TokenBudgetGovernor(default_max_tokens=60)
        profile = MemoryProfile(
            agent_id="test_agent",
            facts=[
                Fact(id="f1", entity="User", attribute="name", value="Alex", confidence=0.9),
                Fact(id="f2", entity="User", attribute="role", value="Admin", confidence=0.8),
                Fact(id="f3", entity="Project", attribute="framework", value="React Vite Tailwind", confidence=0.7),
            ],
            rules=[
                OperationalRule(id="r1", category="user_preference", rule="Prefer dark mode", priority=1, rationale=""),
            ],
        )

        injection = governor.build_budgeted_prompt_injection(profile, query_context="framework", max_tokens=150)
        self.assertIn("REMAGENT DETERMINISTIC MEMORY CONTEXT", injection)
        self.assertIn("Prefer dark mode", injection)
        self.assertNotIn("TOKEN BUDGET OVERFLOW", injection)

    def test_large_rule_set_is_bounded_not_refused(self):
        # A big rule set used to make recall refuse outright, so every session
        # lost its memory the moment the brain grew. Recall now selects within
        # the budget and states what it left out.
        # 20 rules at ~42 tokens each is ~840 tokens of rules against a
        # 300-token budget, so most must be left out and said so.
        governor = TokenBudgetGovernor(default_max_tokens=300)
        rules = [
            OperationalRule(
                id=f"r{i}",
                category="operational_directive",
                rule=f"Critical unbreakable directive number {i}: " + ("x" * 120),
                rationale="",
                priority=1,
            )
            for i in range(20)
        ]
        profile = MemoryProfile(agent_id="overflow_test", facts=[], rules=rules)

        injection = governor.build_budgeted_prompt_injection(profile, max_tokens=300)
        self.assertTrue(injection.strip(), "recall must never come back silently empty")
        self.assertLessEqual(governor.estimate_tokens(injection), 300)
        self.assertIn("rule(s) omitted by token budget", injection)
        self.assertIn("of 20 shown", injection)
        self.assertIn("OPERATIONAL RULES", injection, "at least one rule must survive")

    def test_budget_too_small_for_anything_still_raises(self):
        # The loud failure is reserved for a misconfigured budget, not a big brain.
        governor = TokenBudgetGovernor()
        rules = [OperationalRule(id="r1", category="operational_directive",
                                 rule="x" * 400, rationale="", priority=1)]
        profile = MemoryProfile(agent_id="tiny_budget", facts=[], rules=rules)
        with self.assertRaises(GovernorBudgetError):
            governor.build_budgeted_prompt_injection(profile, max_tokens=5)

    def test_token_budget_governor_all_rules_present_facts_truncated(self):
        # All rules (every priority) must appear; facts fill what's left and
        # the cut is explicitly noted, with the output still under budget.
        governor = TokenBudgetGovernor()
        rules = [
            OperationalRule(id="r1", category="operational_directive",
                            rule="P1 unbreakable directive", rationale="", priority=1),
            OperationalRule(id="r2", category="coding_standard",
                            rule="P2 secondary standard", rationale="", priority=2),
            OperationalRule(id="r3", category="architecture_heuristic",
                            rule="P3 minor guideline", rationale="", priority=3),
        ]
        facts = [
            Fact(id=f"f{i}", entity="Thing", attribute=f"attr_{i}",
                 value="v" * 80, confidence=0.9)
            for i in range(40)
        ]
        profile = MemoryProfile(agent_id="trunc_test", facts=facts, rules=rules)

        budget = 300
        injection = governor.build_budgeted_prompt_injection(profile, max_tokens=budget)
        self.assertLessEqual(governor.estimate_tokens(injection), budget)
        for r in rules:
            self.assertIn(r.rule, injection, f"rule {r.id} must never be dropped")
        self.assertIn("omitted by token budget", injection)
        self.assertIn("Thing.attr_0", injection, "highest-value facts should fill remaining budget")

    async def test_env_var_defaults_point_bare_commands_at_real_memory(self):
        """REMAGENT_DB / REMAGENT_AGENT make bare `remagent status` (no flags,
        any cwd) hit the real brain instead of creating a stray empty DB."""
        await self.storage.save_memory_profile(MemoryProfile(
            agent_id="envagent",
            facts=[Fact(entity="Env", attribute="works", value="yes")],
        ))
        env = {**os.environ, "REMAGENT_DB": self.db_path, "REMAGENT_AGENT": "envagent"}
        with tempfile.TemporaryDirectory() as elsewhere:
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys; sys.argv = ['remagent', 'status']; "
                 "from remagent.cli import main; main()"],
                capture_output=True, text=True, timeout=60, env=env, cwd=elsewhere,
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("envagent", proc.stdout)
        self.assertIn("Env.works = yes", proc.stdout)
        self.assertNotIn("Created a NEW EMPTY database", proc.stderr)

    def test_new_empty_db_creation_is_loud_once(self):
        """Creating a brand-new DB must print a loud notice; opening an
        existing one must not. Silent empty is a fake-empty."""
        ghost = os.path.join(tempfile.mkdtemp(), "fresh.db")

        def run_status():
            return subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.argv = ['remagent', 'status', '--db', {ghost!r}]; "
                 "from remagent.cli import main; main()"],
                capture_output=True, text=True, timeout=60,
            )

        first = run_status()
        self.assertEqual(first.returncode, 0, msg=first.stderr)
        self.assertIn("Created a NEW EMPTY database", first.stderr)
        self.assertIn(ghost, first.stderr)

        second = run_status()
        self.assertNotIn("Created a NEW EMPTY database", second.stderr,
                         "notice must only fire when the file is actually new")

    async def test_recall_cli_succeeds_with_a_large_rule_set(self):
        """Recall must keep working as the brain grows. It previously exited 1
        once the rules alone exceeded the budget, which is how a 744-rule brain
        silently stopped reaching any session."""
        rules = [
            OperationalRule(id=f"r{i}", category="operational_directive",
                            rule=f"Directive {i}: " + ("y" * 300), rationale="", priority=1)
            for i in range(40)
        ]
        await self.storage.save_memory_profile(
            MemoryProfile(agent_id="overflow_agent", rules=rules)
        )
        await self.storage.close()

        args = ["remagent", "recall", "--format", "injection", "--agent", "overflow_agent",
                "--db", self.db_path, "--max-tokens", "2000"]
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.argv = {args!r}; from remagent.cli import main; main()"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("REMAGENT DETERMINISTIC MEMORY CONTEXT", proc.stdout)
        self.assertIn("rule(s) omitted by token budget", proc.stdout)

    def test_memory_decay_engine(self):
        decay_engine = MemoryDecayEngine(half_life_days=1.0, min_confidence_floor=0.3)
        old_fact = Fact(
            id="f_decay",
            entity="Temp",
            attribute="scratchpad",
            value="123",
            confidence=0.5,
            timestamp=time.time() - (86400 * 2),  # 2 days old
        )
        profile = MemoryProfile(agent_id="decay_test", facts=[old_fact], rules=[])

        updated_profile, pruned = decay_engine.apply_decay(profile, current_timestamp=time.time())
        # After 2 half-lives, 0.5 * 0.5 * 0.5 = 0.125 < 0.3 floor -> pruned
        self.assertEqual(len(pruned), 1)
        self.assertFalse(updated_profile.facts[0].is_active)

    def _run_decay_cli(self, agent: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.argv = ['remagent', 'decay', '--db', {self.db_path!r}, "
                f"'--agent', {agent!r}, '--half-life-days', '1.0', '--floor', '0.3']; "
                "from remagent.cli import main; main()",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    async def test_decay_cli_happy_path_persists(self):
        stale = Fact(
            id="f_stale", entity="Temp", attribute="scratch", value="old",
            confidence=0.5, timestamp=time.time() - (86400 * 2),
        )
        durable = Fact(
            id="f_durable", entity="Arch", attribute="db", value="PostgreSQL",
            confidence=1.0, timestamp=time.time() - (86400 * 2),
        )
        await self.storage.save_memory_profile(
            MemoryProfile(agent_id="decay_agent", facts=[stale, durable], rules=[])
        )

        proc = self._run_decay_cli("decay_agent")
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")
        self.assertIn("Decay pass complete", proc.stdout)

        loaded = await self.storage.load_memory_profile("decay_agent")
        by_id = {f.id: f for f in loaded.facts}
        self.assertFalse(by_id["f_stale"].is_active, "decayed fact must be persisted as inactive")
        self.assertTrue(by_id["f_durable"].is_active, "confidence-1.0 facts never decay")

    async def test_decay_cli_missing_profile_fails_nonzero(self):
        proc = self._run_decay_cli("ghost_agent")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAILED", proc.stderr)


class TestLogContentFile(unittest.IsolatedAsyncioTestCase):
    """`remagent log --content-file` — the path Hermes/agent hooks use to send a
    transcript that would blow past the OS argument-length limit."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "log.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _log(self, *args, stdin=None):
        argv = ["remagent", "log", "--role", "user", "--db", self.db, *args]
        return subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.argv = {argv!r}; from remagent.cli import main; main()"],
            capture_output=True, text=True, timeout=60, input=stdin,
        )

    def _contents(self):
        con = sqlite3.connect(self.db)
        try:
            return [r[0] for r in con.execute("SELECT content FROM raw_turns ORDER BY rowid")]
        finally:
            con.close()

    def test_content_file_from_disk(self):
        path = os.path.join(self.dir, "turn.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("vendor price is $40/unit\nsecond line")
        proc = self._log("--content-file", path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(self._contents(), ["vendor price is $40/unit\nsecond line"])

    def test_content_file_stdin_accepts_payload_too_long_for_argv(self):
        # 400 KB: comfortably past what a single argv entry can carry on macOS.
        big = "x" * 400_000
        proc = self._log("--content-file", "-", stdin=big)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        stored = self._contents()
        self.assertEqual(len(stored), 1)
        self.assertEqual(len(stored[0]), 400_000)

    def test_content_flag_still_works(self):
        proc = self._log("--content", "plain path unchanged")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(self._contents(), ["plain path unchanged"])

    def test_content_and_content_file_are_mutually_exclusive(self):
        proc = self._log("--content", "a", "--content-file", "-", stdin="b")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not allowed with", proc.stderr)
        self.assertFalse(os.path.exists(self.db), "rejected args must not create the DB")

    def test_one_of_the_two_is_required(self):
        proc = self._log()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("required", proc.stderr)

    def test_empty_stdin_is_a_noop_not_a_failure(self):
        # A hook firing on a session with nothing to say must not report failure.
        proc = self._log("--content-file", "-", stdin="   \n  ")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(self._contents(), [])

    def test_missing_content_file_fails_loudly_and_logs_nothing(self):
        proc = self._log("--content-file", os.path.join(self.dir, "absent.txt"))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAILED", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(self._contents(), [])


if __name__ == "__main__":
    unittest.main()
