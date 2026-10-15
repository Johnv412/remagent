"""Secret hiding (remagent/sensitive.py): keys, tokens, passwords and card numbers never enter the buffer or a dream."""

import asyncio
import os
import sqlite3
import tempfile
import unittest

from remagent.engine.prompting import build_user_prompt
from remagent.schemas import MemoryProfile, RawTurnLog
from remagent.sensitive import detect_sensitive_categories, redact
from remagent.storage.sqlite import SQLiteStorageAdapter

FAKE_GOOGLE = "AIza" + "Q" * 35          # shaped like a Google key, not a real one
FAKE_OPENAI = "sk-" + "a1B2" * 12
FAKE_JWT = "eyJ" + "x" * 20 + "." + "y" * 20 + "." + "z" * 20
FAKE_CARD = "4111 1111 1111 1111"        # passes Luhn


class TestRedact(unittest.TestCase):
    def test_each_kind_is_replaced_with_a_tag_and_the_value_is_gone(self):
        cases = {
            f"use key {FAKE_GOOGLE} for gemini": ("api_key", FAKE_GOOGLE),
            f"OPENAI_API_KEY={FAKE_OPENAI}": ("api_key", FAKE_OPENAI),
            f"Authorization: Bearer {FAKE_JWT}": ("token", FAKE_JWT),
            "password: hunter2!": ("password", "hunter2!"),
            "the db password is swordfish": ("password", "swordfish"),
            f"card {FAKE_CARD} exp 12/29": ("payment_card", FAKE_CARD),
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----": ("private_key", "MIIE"),
            "ghp_" + "A" * 36: ("api_key", "ghp_" + "A" * 36),
        }
        for text, (category, value) in cases.items():
            clean, found = redact(text)
            self.assertIn(category, found, text)
            self.assertNotIn(value, clean, text)
            self.assertIn(f"[REDACTED:{category}]", clean, text)

    def test_labels_survive_values_do_not(self):
        clean, _ = redact("set api_key = abcdefgh12345678 then restart")
        self.assertEqual(clean, "set api_key = [REDACTED:api_key] then restart")

    def test_plain_text_and_ordinary_numbers_are_untouched(self):
        for text in ("Reply with OK", "John's average job is $450 and 60 of 200 calls go unanswered",
                     "order 2026-10-15 total 12,064 dollars", "call 215-555-0100 tomorrow",
                     "the score was 9/10 APPROVE", "1234567890123"):  # 13 digits that fail Luhn
            clean, found = redact(text)
            self.assertEqual((clean, found), (text, []), text)

    def test_detect_returns_categories_only(self):
        self.assertEqual(detect_sensitive_categories(f"k={FAKE_OPENAI} password: x1"), ["api_key", "password"])


class TestNothingSecretIsStored(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "m.db")
        self.storage = SQLiteStorageAdapter(db_path=self.db)
        await self.storage.initialize()

    async def asyncTearDown(self):
        await self.storage.close()

    async def test_turn_is_clean_in_the_model_and_on_disk(self):
        turn = RawTurnLog(role="user", content=f"my gemini key is {FAKE_GOOGLE}, password: tiger99")
        self.assertNotIn(FAKE_GOOGLE, turn.content)
        self.assertNotIn("tiger99", turn.content)
        self.assertIn("[REDACTED:api_key]", turn.content)
        await self.storage.save_turn(turn)
        raw = sqlite3.connect(self.db).execute("SELECT content FROM raw_turns").fetchone()[0]
        self.assertNotIn(FAKE_GOOGLE, raw)
        self.assertNotIn("tiger99", raw)

    async def test_old_rows_are_scrubbed_on_the_way_into_a_dream(self):
        # a row written by an older version, straight into SQLite with the secret intact
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO raw_turns (turn_id, session_id, role, content, timestamp, is_consolidated) "
                    "VALUES ('old-1', 's', 'user', ?, '2026-01-01T00:00:00Z', 0)", (f"token {FAKE_JWT} ok",))
        con.commit(); con.close()
        turns = await self.storage.get_unconsolidated_turns()
        prompt, _ = build_user_prompt(turns, MemoryProfile(agent_id="t"))
        self.assertNotIn(FAKE_JWT, prompt)
        self.assertIn("[REDACTED:token]", prompt)


if __name__ == "__main__":
    unittest.main()
