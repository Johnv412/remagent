"""
Tests for `remagent soak` — the plain-English soak verdict.

The verdict must be as honest as everything else: PASS only when the
evidence supports it, FAIL with the reason named, IN PROGRESS otherwise.
"""

import json
import os
import shutil
import tempfile
import unittest

from remagent.schemas import Fact, MemoryProfile, RawTurnLog, current_utc_iso, generate_uuid
from remagent.soak import run_soak_report
from remagent.storage.sqlite import SQLiteStorageAdapter

import sqlite3

START, END = "2026-08-25", "2026-08-31"


def snapshot_line(day: str, ok: bool = True, failed_check: str = "no_fallback") -> str:
    checks = [{"name": "no_fallback", "passed": True, "detail": "clean"},
              {"name": "no_erasure", "passed": True, "detail": "clean"},
              {"name": "queue", "passed": True, "detail": "queue empty"}]
    if not ok:
        for c in checks:
            if c["name"] == failed_check:
                c["passed"] = False
                c["detail"] = "dirty"
    return json.dumps({"ok": ok, "timestamp": f"{day}T21:00:00+00:00", "checks": checks})


class TestSoakVerdict(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="soak_test_")
        self.db_path = os.path.join(self.dir, "memory.db")
        self.log_path = os.path.join(self.dir, "soak.jsonl")
        self.cfg_path = os.path.join(self.dir, "soak_config.json")
        self.storage = SQLiteStorageAdapter(db_path=self.db_path)
        await self.storage.initialize()

    async def asyncTearDown(self):
        await self.storage.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_config(self, **overrides):
        cfg = {
            "start_date": START, "end_date": END, "days": 7,
            "db": self.db_path, "agent": "john", "soak_log": self.log_path,
            "baseline_db": self.db_path, "baseline_taken_at": f"{START}T00:00:00+00:00",
            "baseline_fact_count": 0, "baseline_audit_count": 0, "baseline_turn_count": 0,
        }
        cfg.update(overrides)
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f)

    def write_log(self, lines):
        with open(self.log_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    async def seed_passing_memory(self):
        """Facts with a proper supersession + an audit row recording it."""
        old = Fact(entity="Vendor", attribute="price", value="$40", is_active=False)
        new = Fact(entity="Vendor", attribute="price", value="$52")
        old.superseded_by = new.id
        await self.storage.save_memory_profile(MemoryProfile(agent_id="john", facts=[old, new]))
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO dream_audit_history (run_id, agent_id, added_facts_json, updated_rules_json, "
            "contradiction_resolutions_json, pruned_noise_count, pruned_noise_reasons_json, "
            "reasoning_summary, consolidated_turn_ids_json, timestamp, estimated_token_savings) "
            "VALUES (?, 'john', '[]', '[]', ?, 0, '[]', 'real dream', '[]', ?, 0)",
            (generate_uuid(), json.dumps([{"entity": "Vendor"}]), f"{START}T12:00:00+00:00"),
        )
        con.commit()
        con.close()

    def test_no_config_fails(self):
        code, report = run_soak_report(config_path=os.path.join(self.dir, "missing.json"))
        self.assertEqual(code, 1)
        self.assertIn("No soak is configured", report)

    async def test_in_progress_healthy(self):
        self.write_config()
        self.write_log([snapshot_line("2026-08-25"), snapshot_line("2026-08-26")])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-08-27")
        self.assertEqual(code, 0)
        self.assertIn("IN PROGRESS", report)
        self.assertIn("day 3", report)

    async def test_in_progress_with_failed_snapshot_fails_early(self):
        self.write_config()
        self.write_log([snapshot_line("2026-08-25", ok=False)])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-08-26")
        self.assertEqual(code, 1)
        self.assertIn("FAILING", report)

    async def test_complete_all_criteria_pass(self):
        await self.seed_passing_memory()
        self.write_config()
        self.write_log([snapshot_line(f"2026-08-{d}") for d in range(25, 32)])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-09-01")
        self.assertEqual(code, 0, msg=report)
        self.assertIn("SOAK PASSED", report)
        self.assertIn("Vendor.price", report)  # injection is printed for the human check

    async def test_complete_insufficient_snapshots_fails(self):
        await self.seed_passing_memory()
        self.write_config()
        self.write_log([snapshot_line("2026-08-25"), snapshot_line("2026-08-26")])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-09-01")
        self.assertEqual(code, 1)
        self.assertIn("insufficient evidence", report)

    async def test_complete_no_contradiction_fails(self):
        await self.storage.save_memory_profile(
            MemoryProfile(agent_id="john", facts=[Fact(entity="X", attribute="y", value="z")])
        )
        self.write_config()
        self.write_log([snapshot_line(f"2026-08-{d}") for d in range(25, 32)])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-09-01")
        self.assertEqual(code, 1)
        self.assertIn("no organic contradiction", report)

    async def test_complete_large_rule_set_passes_with_trim_noted(self):
        """A brain big enough to outgrow the budget is a healthy brain, not a
        failed soak. Recall is bounded now, so the soak must pass and say what
        was trimmed instead of refusing. It should still fail on an empty
        injection, which is a different thing entirely."""
        from remagent.schemas import OperationalRule
        rules = [
            OperationalRule(id=f"r{i}", category="operational_directive",
                            rule=f"Massive directive {i}: " + ("z" * 900),
                            rationale="", priority=1)
            for i in range(30)  # ~30 * 900 chars >> 6000-token default budget
        ]
        old = Fact(entity="Vendor", attribute="price", value="$40", is_active=False)
        new = Fact(entity="Vendor", attribute="price", value="$52")
        old.superseded_by = new.id
        await self.storage.save_memory_profile(
            MemoryProfile(agent_id="john", facts=[old, new], rules=rules)
        )
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO dream_audit_history (run_id, agent_id, added_facts_json, updated_rules_json, "
            "contradiction_resolutions_json, pruned_noise_count, pruned_noise_reasons_json, "
            "reasoning_summary, consolidated_turn_ids_json, timestamp, estimated_token_savings) "
            "VALUES (?, 'john', '[]', '[]', ?, 0, '[]', 'real dream', '[]', ?, 0)",
            (generate_uuid(), json.dumps([{"entity": "Vendor"}]), f"{START}T12:00:00+00:00"),
        )
        con.commit()
        con.close()
        self.write_config()
        self.write_log([snapshot_line(f"2026-08-{d}") for d in range(25, 32)])
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-09-01")
        self.assertEqual(code, 0, msg=report)
        self.assertIn("within budget", report)
        self.assertIn("trimmed to fit", report)
        self.assertNotIn("cannot be built", report)

    async def test_complete_bad_snapshot_fails_with_check_named(self):
        await self.seed_passing_memory()
        self.write_config()
        lines = [snapshot_line(f"2026-08-{d}") for d in range(25, 31)]
        lines.append(snapshot_line("2026-08-31", ok=False, failed_check="no_erasure"))
        self.write_log(lines)
        code, report = run_soak_report(config_path=self.cfg_path, today="2026-09-01")
        self.assertEqual(code, 1)
        self.assertIn("no_erasure", report)


if __name__ == "__main__":
    unittest.main()
