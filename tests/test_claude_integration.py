"""
Unit & Integration Tests for Claude Code MCP Server and Configuration Hooks.
Verifies tool registration, execution, hook generation, and CLI subcommands.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from unittest import mock

from remagent.daemon import DreamDaemon
from remagent.engine.synthesizer import DreamSynthesisError
from remagent.schemas import Fact, OperationalRule, RawTurnLog, MemoryProfile
from remagent.storage.sqlite import SQLiteStorageAdapter
from remagent.integrations.claude_code import create_claude_mcp_server
from remagent.integrations.claude_hooks import (
    detect_native_auto_memory,
    generate_claude_configuration,
)


class TestClaudeIntegrationSuite(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="claude_test_")
        self.db_path = os.path.join(self.temp_dir, "test_claude_mem.db")
        self.storage = SQLiteStorageAdapter(db_path=self.db_path)
        await self.storage.initialize()

    async def asyncTearDown(self):
        await self.storage.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_mcp_server_tools_registered(self):
        server = create_claude_mcp_server(
            name="test_remagent",
            default_db_path=self.db_path,
            default_agent_id="test_claude_agent",
        )
        self.assertIsNotNone(server)

        # Verify tool registration via tool manager
        tool_names = list(server._tool_manager._tools.keys())
        self.assertIn("remagent_recall", tool_names)
        self.assertIn("remagent_log", tool_names)
        self.assertIn("remagent_dream", tool_names)

    async def test_mcp_log_and_recall_tools(self):
        server = create_claude_mcp_server(
            name="test_remagent",
            default_db_path=self.db_path,
            default_agent_id="test_claude_agent",
        )

        # 1. Log a turn via MCP tool call
        log_result = await server.call_tool(
            "remagent_log",
            {
                "role": "user",
                "content": "Configure backend to use FastAPI with Python 3.11",
                "session_id": "claude_session_1",
                "agent_id": "test_claude_agent",
                "db_path": self.db_path,
            },
        )
        self.assertFalse(log_result.is_error)
        
        # Verify turn in SQLite
        turns = await self.storage.get_unconsolidated_turns(session_id="claude_session_1")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].content, "Configure backend to use FastAPI with Python 3.11")

        # 2. Seed memory profile with fact and rule
        fact = Fact(
            entity="Backend",
            attribute="framework",
            value="FastAPI",
            confidence=0.98,
        )
        rule = OperationalRule(
            category="coding_standard",
            rule="Always write async route handlers in FastAPI",
            rationale="Maintains non-blocking I/O performance",
            priority=1,
        )
        profile = MemoryProfile(
            agent_id="test_claude_agent",
            facts=[fact],
            rules=[rule],
        )
        await self.storage.save_memory_profile(profile)

        # 3. Recall memory via MCP tool call
        recall_result = await server.call_tool(
            "remagent_recall",
            {
                "query_context": "FastAPI async",
                "agent_id": "test_claude_agent",
                "db_path": self.db_path,
                "max_tokens": 300,
            },
        )
        self.assertFalse(recall_result.is_error)
        
        # Parse output
        output_text = recall_result.content[0].text
        data = json.loads(output_text)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_active_facts"], 1)
        self.assertEqual(data["total_active_rules"], 1)
        self.assertIn("REMAGENT DETERMINISTIC MEMORY CONTEXT", data["prompt_injection"])
        self.assertIn("Always write async route handlers in FastAPI", data["prompt_injection"])

    async def test_mcp_dream_tool(self):
        server = create_claude_mcp_server(
            name="test_remagent",
            default_db_path=self.db_path,
            default_agent_id="test_claude_agent",
        )

        # Trigger dream when no unconsolidated turns exist -> skipped status
        dream_result = await server.call_tool(
            "remagent_dream",
            {
                "agent_id": "test_claude_agent",
                "db_path": self.db_path,
            },
        )
        self.assertFalse(dream_result.is_error)
        data = json.loads(dream_result.content[0].text)
        self.assertEqual(data["status"], "skipped")

    async def test_mcp_dream_synthesis_failure_surfaces_as_error(self):
        """A failed dream must surface as an MCP tool error (ToolError at the
        framework boundary), never as a success payload."""
        from mcp.server.mcpserver.exceptions import ToolError

        server = create_claude_mcp_server(
            name="test_remagent", default_db_path=self.db_path, default_agent_id="a"
        )
        with mock.patch.object(
            DreamDaemon, "consolidate_now",
            side_effect=DreamSynthesisError("Gemini dream synthesis failed: boom"),
        ):
            with self.assertRaises(ToolError) as ctx:
                await server.call_tool("remagent_dream", {"db_path": self.db_path})
        # Whether ToolError carries the underlying message is mcp-version-
        # dependent (newer versions mask it); the load-bearing invariant is
        # that the failure surfaces as an error and never as a success/
        # up-to-date payload.
        self.assertNotIn("up to date", str(ctx.exception))
        self.assertNotIn("consolidated", str(ctx.exception))

    async def test_mcp_log_rejects_invalid_role(self):
        """Invalid roles must error, not be silently coerced into a logged turn."""
        from mcp.server.mcpserver.exceptions import ToolError

        server = create_claude_mcp_server(
            name="test_remagent", default_db_path=self.db_path, default_agent_id="a"
        )
        with self.assertRaises(ToolError):
            await server.call_tool(
                "remagent_log",
                {"role": "wizard", "content": "should not be stored", "db_path": self.db_path},
            )
        turns = await self.storage.get_unconsolidated_turns()
        self.assertEqual(turns, [], "rejected turn must not be persisted")

    def test_claude_hooks_and_settings_generation(self):
        target_dir = os.path.join(self.temp_dir, "workspace")
        os.makedirs(target_dir, exist_ok=True)

        results = generate_claude_configuration(
            target_dir=target_dir,
            db_path="workspace_memory.db",
            agent_id="dev_claude",
            force=False,
        )

        settings_file = Path(target_dir) / ".claude" / "settings.json"
        session_start_script = Path(target_dir) / ".claude" / "hooks" / "session_start.py"
        prompt_hook = Path(target_dir) / ".claude" / "hooks" / "user_prompt_submit.py"

        self.assertTrue(settings_file.exists())
        self.assertTrue(session_start_script.exists())
        self.assertTrue(prompt_hook.exists())

        # Orphan shell scripts are no longer generated: nothing ever
        # referenced them, and stop.sh implied a paid dream per session.
        self.assertFalse((Path(target_dir) / ".claude" / "hooks" / "session_start.sh").exists())
        self.assertFalse((Path(target_dir) / ".claude" / "hooks" / "stop.sh").exists())

        # Every generated hook script must be valid Python, or it fails
        # silently at runtime the way the old flat-schema config did.
        import ast
        ast.parse(session_start_script.read_text())
        ast.parse(prompt_hook.read_text())

        # Verify executable permissions
        self.assertTrue(os.access(session_start_script, os.X_OK))
        self.assertTrue(os.access(prompt_hook, os.X_OK))

        # Check settings.json content
        with open(settings_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.assertIn("mcpServers", config)
        self.assertIn("remagent", config["mcpServers"])
        self.assertEqual(config["mcpServers"]["remagent"]["command"], "remagent-mcp")
        self.assertIn("hooks", config)
        self.assertIn("SessionStart", config["hooks"])
        self.assertIn("UserPromptSubmit", config["hooks"])

        # No Stop -> dream hook: consolidation is a paid LLM call and must not
        # fire at the end of every session.
        self.assertNotIn("Stop", config["hooks"])

        # Claude Code requires matcher-group objects. A flat
        # [{"type": "command", ...}] list parses but never fires — that bug
        # shipped once and produced hooks that silently did nothing.
        for event in ("SessionStart", "UserPromptSubmit"):
            group = config["hooks"][event][0]
            self.assertIn("hooks", group, f"{event} must use the matcher-group shape")
            self.assertNotIn("type", group, f"{event} must not use the flat shape")
            self.assertEqual(group["hooks"][0]["type"], "command")

        # Hook commands must be absolute: hooks do not reliably run from the
        # repository root.
        prompt_cmd = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        start_cmd = config["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        self.assertIn("user_prompt_submit.py", prompt_cmd)
        self.assertIn("session_start.py", start_cmd)
        for cmd in (prompt_cmd, start_cmd):
            self.assertNotIn(" .claude/", cmd, "hook path must not be relative")
            self.assertIn(str(Path(target_dir).resolve()), cmd)

        # .gitignore must cover the memory db and markdown mirror: committing
        # agent memory is opt-in (may contain sensitive session content).
        gitignore = (Path(target_dir) / ".gitignore").read_text()
        self.assertIn("workspace_memory.db*", gitignore)
        self.assertIn("workspace_memory_md/", gitignore)
        self.assertIn("workspace_memory_context.md", gitignore)
        self.assertIn("opt-in", gitignore)

        # Check idempotency (including that the .gitignore block is appended once)
        second_run = generate_claude_configuration(target_dir=target_dir, force=False)
        for status in second_run.values():
            self.assertIn("skipped", status)
        gitignore_again = (Path(target_dir) / ".gitignore").read_text()
        self.assertEqual(gitignore, gitignore_again, "gitignore block must not duplicate")

    # ------------------------------------------------------------------
    # Native Auto Memory / Auto Dream detection (best-effort, honest tiers)
    # ------------------------------------------------------------------

    def test_native_detection_env_disable_wins(self):
        target = os.path.join(self.temp_dir, "ws1")
        os.makedirs(target, exist_ok=True)
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}):
            state, detail = detect_native_auto_memory(target, home=Path(self.temp_dir) / "home")
        self.assertEqual(state, "disabled")
        self.assertIn("CLAUDE_CODE_DISABLE_AUTO_MEMORY", detail)

    def test_native_detection_settings_disable(self):
        target = os.path.join(self.temp_dir, "ws2")
        home = Path(self.temp_dir) / "home2"
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / "settings.json").write_text('{"autoMemoryEnabled": false}')
        os.makedirs(target, exist_ok=True)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)
            state, detail = detect_native_auto_memory(target, home=home)
        self.assertEqual(state, "disabled")
        self.assertIn("autoMemoryEnabled", detail)

    def test_native_detection_memory_files_present(self):
        target = os.path.join(self.temp_dir, "ws3")
        os.makedirs(target, exist_ok=True)
        home = Path(self.temp_dir) / "home3"
        import re as _re
        slug = _re.sub(r"[^A-Za-z0-9]", "-", str(Path(target).resolve()))
        mem_dir = home / ".claude" / "projects" / slug / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").write_text("# index\n")
        os.environ.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)
        state, detail = detect_native_auto_memory(target, home=home)
        self.assertEqual(state, "active")
        self.assertIn("memory", detail)

    def test_native_detection_unknown_when_no_signal(self):
        target = os.path.join(self.temp_dir, "ws4")
        os.makedirs(target, exist_ok=True)
        os.environ.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)
        state, detail = detect_native_auto_memory(target, home=Path(self.temp_dir) / "home4")
        self.assertEqual(state, "unknown")
        self.assertIn("has not written memory files", detail)

    # ------------------------------------------------------------------
    # UserPromptSubmit turn-capture hook (functional, via subprocess)
    # ------------------------------------------------------------------

    def _generate_and_get_prompt_hook(self) -> str:
        target_dir = os.path.join(self.temp_dir, "hookws")
        os.makedirs(target_dir, exist_ok=True)
        generate_claude_configuration(target_dir=target_dir, db_path=self.db_path)
        return os.path.join(target_dir, ".claude", "hooks", "user_prompt_submit.py")

    def _run_prompt_hook(self, hook: str, stdin_text: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, hook, "--db", self.db_path],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=60,
        )

    async def test_prompt_hook_logs_prompt_verbatim(self):
        hook = self._generate_and_get_prompt_hook()
        payload = {"prompt": "We use PostgreSQL 16 on Cloud SQL for prod.", "session_id": "sess_42"}
        proc = self._run_prompt_hook(hook, json.dumps(payload))
        self.assertEqual(proc.returncode, 0, msg=f"stderr: {proc.stderr}")

        turns = await self.storage.get_unconsolidated_turns()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].content, "We use PostgreSQL 16 on Cloud SQL for prod.")
        self.assertEqual(turns[0].session_id, "sess_42")
        self.assertEqual(turns[0].role, "user")

    async def test_prompt_hook_empty_prompt_is_noop(self):
        hook = self._generate_and_get_prompt_hook()
        proc = self._run_prompt_hook(hook, json.dumps({"prompt": "   "}))
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(await self.storage.get_unconsolidated_turns(), [])

    async def test_prompt_hook_invalid_json_exits_one_and_persists_nothing(self):
        hook = self._generate_and_get_prompt_hook()
        proc = self._run_prompt_hook(hook, "this is not json")
        self.assertEqual(proc.returncode, 1, "failure must be non-zero but never 2 (would block the prompt)")
        self.assertIn("remagent hook", proc.stderr)
        self.assertEqual(await self.storage.get_unconsolidated_turns(), [])


if __name__ == "__main__":
    unittest.main()
