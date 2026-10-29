"""remagent log --turn-id: the same id twice is stored once, never overwritten, and the command says which."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from remagent.schemas import RawTurnLog
from remagent.storage.sqlite import SQLiteStorageAdapter


def run_log(db, *extra, content="hello"):
    cmd = [sys.executable, "-m", "remagent.cli", "log", "--role", "user", "--db", db, "--content", content, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


class TestSaveTurnReturnsWhetherItWasNew(unittest.IsolatedAsyncioTestCase):
    async def test_insert_or_ignore(self):
        d = tempfile.mkdtemp()
        s = SQLiteStorageAdapter(db_path=os.path.join(d, "m.db"))
        await s.initialize()
        try:
            first = RawTurnLog(turn_id="t-1", role="user", content="first words")
            self.assertTrue(await s.save_turn(first))
            later = RawTurnLog(turn_id="t-1", role="user", content="DIFFERENT words", is_consolidated=True)
            self.assertFalse(await s.save_turn(later))
            rows = await s.get_unconsolidated_turns()
            self.assertEqual([(t.turn_id, t.content, t.is_consolidated) for t in rows], [("t-1", "first words", False)])
        finally:
            await s.close()


class TestLogCommand(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "m.db")

    def count(self):
        return sqlite3.connect(self.db).execute("SELECT count(*) FROM raw_turns").fetchone()[0]

    def test_same_id_twice_is_one_row_and_the_second_says_duplicate(self):
        a = run_log(self.db, "--turn-id", "abc-1")
        b = run_log(self.db, "--turn-id", "abc-1", content="a retry with other text")
        self.assertEqual((a.returncode, b.returncode), (0, 0), (a.stderr, b.stderr))
        self.assertIn("✅ saved: turn abc-1", a.stdout)
        self.assertIn("↩️  duplicate: turn abc-1", b.stdout)
        self.assertEqual(self.count(), 1)
        self.assertEqual(sqlite3.connect(self.db).execute("SELECT content FROM raw_turns").fetchone()[0], "hello")

    def test_json_output_and_no_id_means_a_fresh_turn_each_time(self):
        a = json.loads(run_log(self.db, "--json").stdout.strip())
        b = json.loads(run_log(self.db, "--json").stdout.strip())
        self.assertEqual((a["status"], b["status"]), ("saved", "saved"))
        self.assertNotEqual(a["turn_id"], b["turn_id"])
        self.assertEqual(self.count(), 2)
        c = json.loads(run_log(self.db, "--json", "--turn-id", a["turn_id"]).stdout.strip())
        self.assertEqual(c, {"status": "duplicate", "turn_id": a["turn_id"], "session_id": "default_session",
                             "secrets_redacted": False})

    def test_failed_is_loud_and_exits_1(self):
        bad = os.path.join(tempfile.mkdtemp(), "nope", "deeper", "m.db")  # parent missing: sqlite cannot open
        r = run_log(bad, "--json", "--turn-id", "x")
        self.assertNotEqual(r.returncode, 0)
        out = (r.stdout + r.stderr)
        self.assertTrue('"status": "failed"' in out or "FAILED" in out, out)
        self.assertEqual(self.count() if os.path.exists(self.db) else 0, 0)

    def test_secrets_are_redacted_and_reported(self):
        r = json.loads(run_log(self.db, "--json", content="key AIza" + "Z" * 35).stdout.strip())
        self.assertEqual((r["status"], r["secrets_redacted"]), ("saved", True))
        self.assertNotIn("Z" * 35, sqlite3.connect(self.db).execute("SELECT content FROM raw_turns").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
