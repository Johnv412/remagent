"""
Command-line interface for the RemAgent framework.
"""

import argparse
import asyncio
import json
import os
import sys
from remagent.daemon import ConsolidationBusyError, DreamDaemon
from remagent.decay import MemoryDecayEngine
from remagent.doctor import run_doctor
from remagent.export import ExportError, default_out_dir, export_markdown
from remagent.schemas import current_utc_iso
from remagent.engine.errors import ProviderConfigError
from remagent.engine.providers import make_backend, resolve_provider
from remagent.engine.synthesizer import DreamSynthesizer
from remagent.schemas import RawTurnLog
from remagent.storage.sqlite import SQLiteStorageAdapter
from remagent.governor import GovernorBudgetError, TokenBudgetGovernor
from remagent.integrations.claude_hooks import detect_native_auto_memory, generate_claude_configuration


def _default_db() -> str:
    """--db default: REMAGENT_DB env var wins, else remagent_memory.db in CWD."""
    return os.environ.get("REMAGENT_DB", "remagent_memory.db")


def _default_agent(fallback: str = "default_agent") -> str:
    """--agent default: REMAGENT_AGENT env var wins, else the command's fallback."""
    return os.environ.get("REMAGENT_AGENT", fallback)


async def run_cli():
    parser = argparse.ArgumentParser(
        prog="remagent",
        description="RemAgent: Autonomous Zero-Vector Memory Framework for AI Agents",
    )
    try:
        from importlib.metadata import version as _pkg_version
        _version = _pkg_version("remagent")
    except Exception:
        _version = "unknown"
    parser.add_argument("--version", action="version", version=f"remagent {_version}")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: dream
    dream_parser = subparsers.add_parser("dream", help="Trigger an immediate REM consolidation cycle")
    dream_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    dream_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")
    dream_parser.add_argument(
        "--export-md", nargs="?", const="", default=None, metavar="DIR",
        help="After a successful dream, regenerate the markdown mirror (default DIR: <db>_md)",
    )

    # Command: status
    status_parser = subparsers.add_parser("status", help="Inspect active memory profile and consolidated facts")
    status_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    status_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")

    # Command: recall
    recall_parser = subparsers.add_parser("recall", help="Recall consolidated memory and active directives")
    recall_parser.add_argument("--format", choices=["injection", "json"], default="injection", help="Output format")
    recall_parser.add_argument("--query", default=None, help="Query context for fact relevance filtering")
    recall_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")
    recall_parser.add_argument("--max-tokens", type=int, default=6000, help="Maximum token budget for prompt injection")
    recall_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")

    # Command: log
    log_parser = subparsers.add_parser("log", help="Append a raw interaction turn into the memory buffer")
    log_parser.add_argument("--role", choices=["user", "assistant", "system", "tool"], required=True, help="Turn role")
    log_content = log_parser.add_mutually_exclusive_group(required=True)
    log_content.add_argument("--content", help="Turn message content")
    log_content.add_argument(
        "--content-file",
        help="Read turn content from a file, or '-' for stdin. Use this instead of "
             "--content for long transcripts, which can exceed the OS argument limit.",
    )
    log_parser.add_argument("--session", default="default_session", help="Session ID")
    log_parser.add_argument("--turn-id", default=None,
                            help="Optional stable id for this turn. Logging the same id twice stores it once "
                                 "(insert-or-ignore; the first copy is never overwritten).")
    log_parser.add_argument("--json", action="store_true", help="Print one JSON line: {status, turn_id, ...}")
    log_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")

    # Command: export
    export_parser = subparsers.add_parser("export", help="Export memory as a human-readable mirror")
    export_parser.add_argument("--markdown", action="store_true", help="Markdown format (currently the only format)")
    export_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    export_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")
    export_parser.add_argument("--out", default=None, help="Output directory (default: <db>_md next to the database)")

    # Command: decay
    decay_parser = subparsers.add_parser("decay", help="Apply Ebbinghaus temporal decay to stored facts")
    decay_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    decay_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")
    decay_parser.add_argument("--half-life-days", type=float, default=30.0, help="Confidence half-life in days")
    decay_parser.add_argument("--floor", type=float, default=0.20, help="Confidence floor below which facts are deactivated")

    # Command: doctor
    doctor_parser = subparsers.add_parser("doctor", help="Read-only self-audit of the memory pipeline")
    doctor_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    doctor_parser.add_argument("--agent", default=_default_agent(), help="Agent identifier (env: REMAGENT_AGENT)")
    doctor_parser.add_argument("--max-queue", type=int, default=100, help="Max acceptable unconsolidated turns")
    doctor_parser.add_argument("--max-dream-age-hours", type=float, default=24.0, help="Max hours since last dream when turns are queued")
    doctor_parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text")

    # Command: soak
    soak_parser = subparsers.add_parser("soak", help="Plain-English verdict on the 7-day soak")
    soak_parser.add_argument("--config", default="~/.remagent/soak_config.json", help="Soak config path")
    soak_parser.add_argument("--today", default=None, help=argparse.SUPPRESS)  # test hook: YYYY-MM-DD

    # Command: init-claude
    init_claude_parser = subparsers.add_parser("init-claude", help="Scaffold Claude Code hooks and settings.json")
    init_claude_parser.add_argument("--dir", default=".", help="Target workspace directory")
    init_claude_parser.add_argument("--db", default=_default_db(), help="SQLite database path (env: REMAGENT_DB)")
    init_claude_parser.add_argument("--agent", default=_default_agent("claude_code"), help="Agent identifier (env: REMAGENT_AGENT)")
    init_claude_parser.add_argument("--force", action="store_true", help="Overwrite existing configuration and hooks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "init-claude":
        # Detect BEFORE scaffolding so pre-existing settings are what's read.
        native_state, native_detail = detect_native_auto_memory(args.dir)
        results = generate_claude_configuration(
            target_dir=args.dir,
            db_path=args.db,
            agent_id=args.agent,
            force=args.force,
        )
        print("🧠 [RemAgent] Claude Code Integration Scaffolding:")
        for path, status in results.items():
            print(f"   • {path}: {status}")
        print("\n✨ Claude Code is now wired to RemAgent memory!")
        print("   - SessionStart hook: Pre-loads active rules & knowledge into context.")
        print("   - UserPromptSubmit hook: Logs each of your prompts as a raw turn (run `remagent dream` to consolidate).")
        print("   - MCP server: Exposes remagent_recall, remagent_log, and remagent_dream tools.")
        if native_state == "disabled":
            print(f"\nℹ️  Native Claude Code auto-memory appears DISABLED ({native_detail}).")
            print("   RemAgent will be the only memory layer for this repo.")
        else:
            qualifier = "is ACTIVE" if native_state == "active" else "may be active"
            print(f"\nℹ️  Native Claude Code auto-memory (Auto Dream) {qualifier} ({native_detail}).")
            print("   The two are complementary: native handles this repo's own markdown memory;")
            print("   RemAgent adds a cross-agent shared brain on top (one database reachable from")
            print("   Claude Code, Gemini, and any MCP host).")
        return

    if args.command == "soak":
        from remagent.soak import run_soak_report
        code, report = run_soak_report(config_path=args.config, today=args.today)
        print(report)
        if code != 0:
            sys.exit(code)
        return

    if args.command == "export":
        # Read-only like doctor: must never create the DB it mirrors.
        if not args.markdown:
            print("❌ FAILED: specify a format — currently only --markdown is supported.", file=sys.stderr)
            sys.exit(2)
        out_dir = args.out or default_out_dir(args.db)
        try:
            written = export_markdown(db_path=args.db, agent_id=args.agent, out_dir=out_dir)
        except ExportError as exc:
            print(f"❌ FAILED: markdown export did not complete: {exc}", file=sys.stderr)
            print("   Nothing was written.", file=sys.stderr)
            sys.exit(1)
        print(f"📝 [RemAgent] Memory mirror written: {len(written)} file(s) in {out_dir}")
        for path in written:
            print(f"   • {os.path.basename(path)}")
        return

    if args.command == "doctor":
        # Runs before any storage initialization: doctor is strictly
        # read-only and must never create the database it is auditing.
        results = run_doctor(
            db_path=args.db,
            agent_id=args.agent,
            max_queue=args.max_queue,
            max_dream_age_hours=args.max_dream_age_hours,
        )
        ok = all(r.passed for r in results)
        if args.json:
            print(json.dumps({
                "ok": ok,
                "timestamp": current_utc_iso(),
                "db": args.db,
                "agent": args.agent,
                "checks": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
            }))
        else:
            print(f"🩺 [RemAgent Doctor] db={args.db} agent={args.agent}")
            for r in results:
                print(f"   {'✅' if r.passed else '❌'} {r.name}: {r.detail}")
            print("   → ALL CHECKS PASSED" if ok else "   → DOCTOR FAILED: pipeline needs attention")
        if not ok:
            sys.exit(1)
        return

    synthesizer = None
    provider = None
    if args.command == "dream":
        # Resolve the LLM provider before touching storage so a missing key or
        # missing SDK extra is one line on stderr, exit 1, no traceback.
        try:
            provider = resolve_provider()
            backend = make_backend(provider)
        except ProviderConfigError as exc:
            print(f"❌ FAILED: {exc}", file=sys.stderr)
            sys.exit(1)
        synthesizer = DreamSynthesizer(provider=provider, backend=backend)

    storage = SQLiteStorageAdapter(db_path=args.db)
    try:
        await storage.initialize()
    except Exception as exc:
        if args.command == "log":  # the log command promises one of: saved, duplicate, failed
            if getattr(args, "json", False):
                print(json.dumps({"status": "failed", "turn_id": getattr(args, "turn_id", None),
                                  "error": f"cannot open {args.db}: {type(exc).__name__}: {exc}"}))
            else:
                print(f"❌ FAILED: cannot open {args.db}: {type(exc).__name__}: {exc}", file=sys.stderr)
                print("   Nothing was logged.", file=sys.stderr)
            sys.exit(1)
        raise

    try:
        if args.command == "dream":
            print(f"🧠 [RemAgent] Initiating REM Sleep Consolidation Cycle via {provider.provider} ({provider.model})...")
            daemon = DreamDaemon(storage=storage, synthesizer=synthesizer, agent_id=args.agent)
            try:
                result = await daemon.consolidate_now()
            except ConsolidationBusyError as exc:
                print(f"⏳ BUSY: {exc}", file=sys.stderr)
                sys.exit(2)
            except Exception as exc:
                print(f"❌ FAILED: REM consolidation did not complete: {exc}", file=sys.stderr)
                print("   No facts were written; unconsolidated turns remain queued for retry.", file=sys.stderr)
                sys.exit(1)
            if result:
                print(f"✨ Consolidation Complete! Run ID: {result.run_id}")
                print(f"   - Added Facts: {len(result.added_facts)}")
                print(f"   - Updated Rules: {len(result.updated_rules)}")
                print(f"   - Pruned Noise Items: {result.pruned_noise_count}")
                print(f"   - Token Savings: ~{result.estimated_token_savings} tokens")
                print(f"   - Cognitive Reasoning: {result.reasoning_summary}")
                if args.export_md is not None:
                    out_dir = args.export_md or default_out_dir(args.db)
                    try:
                        written = export_markdown(db_path=args.db, agent_id=args.agent, out_dir=out_dir)
                        print(f"📝 Memory mirror regenerated: {len(written)} file(s) in {out_dir}")
                    except ExportError as exc:
                        # The consolidation itself persisted; the mirror did not.
                        print(f"❌ FAILED: consolidation succeeded but markdown export failed: {exc}", file=sys.stderr)
                        sys.exit(1)
            else:
                print("💤 No unconsolidated turns found. Agent memory is already fully consolidated.")

        elif args.command == "status":
            profile = await storage.load_memory_profile(agent_id=args.agent)
            print(f"📊 [RemAgent Profile: {profile.agent_id}]")
            print(f"   Last Dream: {profile.last_dream_at or 'Never'}")
            print(f"   Total Pruned Turns: {profile.total_pruned_turns}")
            print(f"\n📌 Active Facts ({len([f for f in profile.facts if f.is_active])}):")
            for f in profile.facts:
                if f.is_active:
                    print(f"   • {f.entity}.{f.attribute} = {f.value} (conf: {f.confidence})")
            print(f"\n📜 Operational Directives & Rules ({len([r for r in profile.rules if r.is_active])}):")
            for r in profile.rules:
                if r.is_active:
                    print(f"   • [{r.category.upper()}] P{r.priority}: {r.rule}")

        elif args.command == "recall":
            profile = await storage.load_memory_profile(agent_id=args.agent)
            if args.format == "json":
                active_facts = [f.model_dump() for f in profile.facts if f.is_active]
                active_rules = [r.model_dump() for r in profile.rules if r.is_active]
                print(json.dumps({
                    "agent_id": args.agent,
                    "facts": active_facts,
                    "rules": active_rules,
                    "last_dream_at": profile.last_dream_at,
                }, indent=2))
            else:
                governor = TokenBudgetGovernor(default_max_tokens=args.max_tokens)
                try:
                    injection = governor.build_budgeted_prompt_injection(
                        profile=profile,
                        query_context=args.query,
                        max_tokens=args.max_tokens,
                    )
                except GovernorBudgetError as exc:
                    print(f"❌ FAILED: recall injection could not be built: {exc}", file=sys.stderr)
                    sys.exit(1)
                if injection:
                    print(injection)
                else:
                    print("[No active memory facts or rules found]")

        elif args.command == "decay":
            try:
                profile = await storage.load_memory_profile(agent_id=args.agent)
                if not profile.facts and not profile.rules and profile.last_dream_at is None:
                    print(
                        f"❌ FAILED: no memory profile found for agent '{args.agent}' in {args.db}.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                engine = MemoryDecayEngine(
                    half_life_days=args.half_life_days,
                    min_confidence_floor=args.floor,
                )
                updated_profile, pruned = engine.apply_decay(profile)
                # Persist only after a fully successful decay pass.
                await storage.save_memory_profile(updated_profile)
            except Exception as exc:
                print(f"❌ FAILED: decay pass did not complete: {exc}", file=sys.stderr)
                print("   No changes were persisted.", file=sys.stderr)
                sys.exit(1)
            active_count = len([f for f in updated_profile.facts if f.is_active])
            print(f"🍂 [RemAgent] Decay pass complete for agent '{args.agent}'.")
            print(f"   - Half-life: {args.half_life_days} days | Confidence floor: {args.floor}")
            print(f"   - Facts deactivated this pass: {len(pruned)}")
            for f in pruned:
                print(f"     • {f.entity}.{f.attribute} (confidence decayed below floor)")
            print(f"   - Active facts remaining: {active_count}")

        elif args.command == "log":
            if args.content_file:
                try:
                    if args.content_file == "-":
                        content = sys.stdin.read()
                    else:
                        with open(os.path.expanduser(args.content_file), "r", encoding="utf-8") as fh:
                            content = fh.read()
                except OSError as exc:
                    print(f"❌ FAILED: cannot read --content-file: {exc}", file=sys.stderr)
                    print("   Nothing was logged.", file=sys.stderr)
                    sys.exit(1)
            else:
                content = args.content

            # An empty turn is a no-op, not an error: hooks that fire on a
            # session with nothing to say must not report failure.
            if not content.strip():
                print("💤 Nothing to log: content was empty.")
                return

            turn_kwargs = {"session_id": args.session, "role": args.role, "content": content}
            if args.turn_id:
                turn_kwargs["turn_id"] = args.turn_id.strip()
            turn = RawTurnLog(**turn_kwargs)
            redacted = turn.content != content
            try:
                stored = await storage.save_turn(turn)
            except Exception as exc:
                if args.json:
                    print(json.dumps({"status": "failed", "turn_id": turn.turn_id, "error": f"{type(exc).__name__}: {exc}"}))
                else:
                    print(f"❌ FAILED: turn {turn.turn_id} was NOT saved: {type(exc).__name__}: {exc}", file=sys.stderr)
                sys.exit(1)
            status = "saved" if stored else "duplicate"
            if args.json:
                print(json.dumps({"status": status, "turn_id": turn.turn_id, "session_id": turn.session_id,
                                  "secrets_redacted": redacted}))
            elif stored:
                print(f"✅ saved: turn {turn.turn_id}" + (" (secrets redacted before saving)" if redacted else ""))
            else:
                print(f"↩️  duplicate: turn {turn.turn_id} was already stored; nothing changed.")
    finally:
        await storage.close()


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
