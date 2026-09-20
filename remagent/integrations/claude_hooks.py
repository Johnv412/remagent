"""
Claude Code Hooks & Settings Generator for RemAgent.
Automatically scaffolds .claude/settings.json and hook scripts to connect Claude Code
with RemAgent's zero-vector memory lifecycle (SessionStart recall & Stop dream consolidation).
"""

import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def detect_native_auto_memory(target_dir: str, home: Optional[Path] = None) -> Tuple[str, str]:
    """
    Best-effort detection of Claude Code's native Auto Memory / Auto Dream
    (per-project markdown memory consolidation, shipped 2026).

    Returns (state, detail) with state one of:
      "disabled" — an explicit off-switch was found
      "active"   — native memory files exist for this project
      "unknown"  — no explicit signal either way (the feature defaults on
                   but writes files lazily, so absence proves nothing)
    """
    home = home or Path.home()

    if os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY"):
        return "disabled", "CLAUDE_CODE_DISABLE_AUTO_MEMORY is set in the environment"

    # Project settings override global; read both before we scaffold anything.
    for settings_path in (
        Path(target_dir).resolve() / ".claude" / "settings.json",
        home / ".claude" / "settings.json",
    ):
        try:
            cfg = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("autoMemoryEnabled") is False:
            return "disabled", f"autoMemoryEnabled=false in {settings_path}"

    # Claude Code keys project memory dirs by a sanitized-path slug.
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path(target_dir).resolve()))
    memory_md = home / ".claude" / "projects" / slug / "memory" / "MEMORY.md"
    if memory_md.exists():
        return "active", f"native memory files present at {memory_md.parent}"

    return "unknown", (
        "no explicit disable found; native auto-memory may be on by default "
        "but has not written memory files for this project yet"
    )


SESSION_START_TEMPLATE = """#!/usr/bin/env python3
\"\"\"RemAgent SessionStart hook for Claude Code.

Claude Code truncates hook output to roughly 2KB. A full recall injection is
far larger, so emitting it directly loses most of the brain silently. This
hook writes the COMPLETE injection to a file and puts a compact digest plus a
pointer to that file into the session context.

Fail-soft by contract: a missing brain, a missing binary, or a rules set that
outgrows the token budget all produce silence and exit 0. A memory fault must
never block a session.
\"\"\"
import json
import os
import shutil
import subprocess
import sys

DB = os.path.expanduser("__REMAGENT_DB__")
AGENT = "__REMAGENT_AGENT__"
CONTEXT_FILE = os.path.expanduser("__REMAGENT_CONTEXT__")
MAX_TOKENS = "__REMAGENT_MAX_TOKENS__"
BUDGET = 1800
NL = chr(10)


def _short(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def main():
    remagent = shutil.which("remagent")
    if not remagent or not os.path.exists(DB):
        return 0
    base = ["--agent", AGENT, "--db", DB, "--max-tokens", MAX_TOKENS]

    try:
        full = subprocess.run([remagent, "recall", "--format", "injection"] + base,
                              capture_output=True, text=True, timeout=60)
        if full.returncode == 0 and full.stdout.strip():
            parent = os.path.dirname(CONTEXT_FILE)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(CONTEXT_FILE, "w", encoding="utf-8") as fh:
                fh.write(full.stdout)
    except Exception:
        pass

    try:
        res = subprocess.run([remagent, "recall", "--format", "json"] + base,
                             capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            return 0
        data = json.loads(res.stdout)
    except Exception:
        return 0

    rules = [r for r in data.get("rules", []) if r.get("is_active", True)]
    facts = data.get("facts", [])
    prefs = [r for r in rules if r.get("category") == "user_preference"]
    p1 = [r for r in rules
          if r.get("category") != "user_preference" and r.get("priority", 3) <= 1]

    dream = str(data.get("last_dream_at") or "never")[:10]
    lines = ["[REMAGENT BRAIN | agent=" + AGENT + "] "
             + str(len(facts)) + " facts, " + str(len(rules))
             + " rules, last dream " + dream]
    if prefs:
        lines.append("OPERATOR PREFERENCES:")
        for r in prefs:
            lines.append("- " + _short(r.get("rule", ""), 150))

    footer = ["FULL BRAIN (" + str(len(rules)) + " rules + " + str(len(facts))
              + " facts): " + CONTEXT_FILE,
              "Read that file when the task touches systems it describes."]
    reserved = len(NL.join(footer)) + 2

    if p1:
        lines.append("CORE DOCTRINE (P1):")
        for r in p1:
            cand = "- " + _short(r.get("rule", ""), 95)
            if len(NL.join(lines)) + len(cand) + reserved > BUDGET:
                break
            lines.append(cand)

    lines.extend(footer)
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": NL.join(lines)[:BUDGET],
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def generate_claude_configuration(
    target_dir: str = ".",
    db_path: str = "remagent_memory.db",
    agent_id: str = "claude_code",
    force: bool = False,
) -> Dict[str, str]:
    """
    Generates .claude/settings.json and executable hook shell scripts in target_dir.
    Returns a mapping of relative file paths to their creation status.
    """
    base_dir = Path(target_dir).resolve()
    claude_dir = base_dir / ".claude"
    hooks_dir = claude_dir / "hooks"

    claude_dir.mkdir(parents=True, exist_ok=True)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    settings_path = claude_dir / "settings.json"
    session_start_hook_path = hooks_dir / "session_start.py"
    prompt_hook_path = hooks_dir / "user_prompt_submit.py"
    db_stem = os.path.splitext(os.path.basename(db_path))[0]
    context_file = str(base_dir / f"{db_stem}_context.md")

    settings_content = {
        "mcpServers": {
            "remagent": {
                "command": "remagent-mcp",
                "args": ["--db", db_path, "--agent", agent_id],
            }
        },
        # Claude Code expects matcher-group objects here. A flat
        # [{"type": "command", ...}] list parses but never fires, which is
        # how earlier scaffolds produced hooks that silently did nothing.
        # Paths are absolute: hooks do not reliably run from the repo root.
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{sys.executable} {session_start_hook_path}",
                        }
                    ]
                }
            ],
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{sys.executable} {prompt_hook_path}",
                        }
                    ]
                }
            ],
            # No Stop -> `remagent dream` hook. Consolidation is a paid LLM
            # call; firing one at the end of every session is expensive and
            # duplicates scheduled consolidation (launchd/cron). Run
            # `remagent dream` on a schedule instead.
        },
    }

    session_start_script = (
        SESSION_START_TEMPLATE
        .replace("__REMAGENT_DB__", db_path)
        .replace("__REMAGENT_AGENT__", agent_id)
        .replace("__REMAGENT_CONTEXT__", context_file)
        .replace("__REMAGENT_MAX_TOKENS__", "8000")
    )

    prompt_hook_script = f'''#!/usr/bin/env python3
"""RemAgent UserPromptSubmit hook for Claude Code.

Reads the hook JSON on stdin and appends the user's prompt to the episodic
memory buffer via the storage API directly (no PATH dependence).

Exit codes: 0 = logged, or nothing to log; 1 = failure (non-blocking:
Claude Code surfaces the error but the prompt still goes through; nothing
is persisted). Never exits 2 — that would block the user's prompt.
"""
import argparse
import asyncio
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default={db_path!r})
    parser.add_argument("--session", default=None)
    args = parser.parse_args()

    try:
        data = json.load(sys.stdin)
    except Exception as exc:
        print(f"remagent hook: invalid hook JSON on stdin: {{exc}}", file=sys.stderr)
        return 1

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return 0

    session_id = data.get("session_id") or args.session or "claude_code"
    db = os.path.expanduser(args.db)

    try:
        parent = os.path.dirname(db)
        if parent:
            os.makedirs(parent, exist_ok=True)

        from remagent.schemas import RawTurnLog
        from remagent.storage.sqlite import SQLiteStorageAdapter

        async def _log() -> None:
            storage = SQLiteStorageAdapter(db_path=db)
            await storage.initialize()
            try:
                await storage.save_turn(
                    RawTurnLog(session_id=session_id, role="user", content=prompt)
                )
            finally:
                await storage.close()

        asyncio.run(_log())
    except Exception as exc:
        print(f"remagent hook: failed to log prompt: {{exc}}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    results = {}

    # Write settings.json
    if not settings_path.exists() or force:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings_content, f, indent=2)
        results[str(settings_path)] = "created"
    else:
        results[str(settings_path)] = "skipped (already exists, use --force to overwrite)"

    # Write session_start.py
    if not session_start_hook_path.exists() or force:
        with open(session_start_hook_path, "w", encoding="utf-8") as f:
            f.write(session_start_script)
        # Make script executable
        current_perms = os.stat(session_start_hook_path).st_mode
        os.chmod(session_start_hook_path, current_perms | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        results[str(session_start_hook_path)] = "created"
    else:
        results[str(session_start_hook_path)] = "skipped (already exists, use --force to overwrite)"

    # Ensure the workspace .gitignore covers local memory artifacts —
    # committing agent memory to git is OPT-IN (it may contain sensitive
    # session content). Appended once, marker-guarded, never overwrites.
    gitignore_path = base_dir / ".gitignore"
    ignore_marker = "# RemAgent local memory"
    db_name = os.path.basename(db_path)
    ignore_block = (
        f"\n{ignore_marker} — committing agent memory to git is opt-in;"
        " it may contain sensitive session content\n"
        f"{db_name}*\n"
        f"{db_stem}_md/\n"
        f"{db_stem}_context.md\n"
    )
    existing_ignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    if ignore_marker in existing_ignore:
        results[str(gitignore_path)] = "skipped (RemAgent block already present)"
    else:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write(ignore_block)
        results[str(gitignore_path)] = "updated (memory db + markdown mirror ignored)"

    # Write user_prompt_submit.py (turn capture — without it, dreams have
    # nothing to consolidate)
    if not prompt_hook_path.exists() or force:
        with open(prompt_hook_path, "w", encoding="utf-8") as f:
            f.write(prompt_hook_script)
        current_perms = os.stat(prompt_hook_path).st_mode
        os.chmod(prompt_hook_path, current_perms | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        results[str(prompt_hook_path)] = "created"
    else:
        results[str(prompt_hook_path)] = "skipped (already exists, use --force to overwrite)"

    return results
