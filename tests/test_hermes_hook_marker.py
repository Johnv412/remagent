"""hermes_session_log_hook: done means every prompt landed; reruns skip prompts already saved."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HOOK = Path(__file__).resolve().parents[1] / "remagent" / "integrations" / "hermes_session_log_hook.py"


def load_hook():
    spec = importlib.util.spec_from_file_location("hermes_hook", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRuns:
    """Stands in for subprocess.run: one hermes export, then one remagent log per prompt."""

    def __init__(self, prompts, fail_at=None, already=()):
        self.prompts, self.fail_at, self.already = prompts, fail_at, set(already)
        self.logged = []

    def __call__(self, cmd, **kw):
        if "sessions" in cmd:
            out = "\n".join(json.dumps({"text": p}) for p in self.prompts)
            return subprocess.CompletedProcess(cmd, 0, out, "")
        text = kw.get("input")
        turn_id = cmd[cmd.index("--turn-id") + 1]
        n = len(self.logged)
        if self.fail_at is not None and n == self.fail_at:
            return subprocess.CompletedProcess(cmd, 1, "", "disk full")
        self.logged.append((turn_id, text))
        status = "duplicate" if turn_id in self.already else "saved"
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": status, "turn_id": turn_id}), "")


class TestMarker(unittest.TestCase):
    def setUp(self):
        self.hook = load_hook()
        self.state = Path(tempfile.mkdtemp())
        self.hook.STATE_DIR = self.state
        self.hook.resolve = lambda name: f"/fake/{name}"
        self.hook.note = lambda *a, **k: None

    def run_hook(self, runs, session="sess/1"):
        import io
        payload = json.dumps({"hook_event_name": self.hook.EVENT, "session_id": session})
        with mock.patch.object(self.hook.subprocess, "run", runs), \
                mock.patch.object(self.hook.sys, "stdin", io.StringIO(payload)), \
                mock.patch.dict(os.environ, {"REMAGENT_DB": str(self.state / "m.db")}):
            return self.hook.main()

    def marker(self, session="sess/1"):
        return self.state / f"{session.replace('/', '_')}.done"

    def test_all_prompts_land_marks_done_with_stable_ids(self):
        runs = FakeRuns(["one", "two", "three"])
        self.run_hook(runs)
        self.assertTrue(self.marker().exists())
        self.assertEqual(self.marker().read_text().strip(), "3")
        ids = [t for t, _ in runs.logged]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(i.startswith("hermes-") and len(i) == 39 for i in ids))
        # the same session again produces the same ids (so a retry is a duplicate, not a double)
        again = FakeRuns(["one", "two", "three"])
        self.marker().unlink()
        self.run_hook(again)
        self.assertEqual([t for t, _ in again.logged], ids)

    def test_a_failure_mid_way_leaves_no_marker_and_the_rerun_finishes_the_job(self):
        first = FakeRuns(["one", "two", "three"], fail_at=1)
        self.run_hook(first)
        self.assertFalse(self.marker().exists())          # NOT done: prompt two never landed
        self.assertEqual([x for _, x in first.logged], ["one"])
        saved_ids = {t for t, _ in first.logged}
        rerun = FakeRuns(["one", "two", "three"], already=saved_ids)
        self.run_hook(rerun)
        self.assertTrue(self.marker().exists())           # now every prompt landed
        self.assertEqual([x for _, x in rerun.logged], ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()
