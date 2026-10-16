"""
Soak verdict: grades the 7-day real-usage soak in plain English.

Reads the soak config, the daily doctor snapshots (soak.jsonl), and the
memory DB, then prints PASS / FAIL / IN PROGRESS with reasons. Exit code:
0 for PASS or healthy-in-progress, 1 for FAIL (or unhealthy-in-progress).

Machine-gradable criteria (from docs/SOAK_CRITERIA.md):
  C1  every captured daily snapshot is ok (no_fallback / no_erasure /
      queue never fail); >=5 of 7 snapshots captured (Macs sleep — 5-6
      passes with a warning, <5 is insufficient evidence).
  C2  at least one organic contradiction was resolved since the baseline,
      and the superseded facts all have active replacements.
  C3  the recall injection builds, is non-empty, and fits the token budget.
      Trimming is expected once the brain is large; it is reported, not failed.
The final criterion — "would this injection genuinely help a fresh
session?" — is John's judgment; the report prints the injection and says
so rather than pretending a script can grade usefulness.
"""

import json
import os
import sqlite3
from datetime import date, datetime
from typing import List, Optional, Tuple

from remagent.governor import GovernorBudgetError, TokenBudgetGovernor
from remagent.schemas import Fact, MemoryProfile, OperationalRule

MIN_SNAPSHOTS = 5


def _snapshot_day(line: dict) -> Optional[str]:
    ts = line.get("timestamp", "")
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def run_soak_report(
    config_path: str = "~/.remagent/soak_config.json",
    today: Optional[str] = None,
) -> Tuple[int, str]:
    """Returns (exit_code, plain-English report)."""
    lines: List[str] = []
    cfg_path = os.path.expanduser(config_path)
    if not os.path.exists(cfg_path):
        return 1, f"❌ No soak is configured ({cfg_path} not found)."
    cfg = json.load(open(cfg_path))

    start = date.fromisoformat(cfg["start_date"])
    end = date.fromisoformat(cfg["end_date"])
    now = date.fromisoformat(today) if today else date.today()
    total_days = cfg.get("days", 7)

    # --- gather snapshots in the soak window ---
    snapshots = []
    log_path = os.path.expanduser(cfg["soak_log"])
    if os.path.exists(log_path):
        for raw in open(log_path):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                return 1, f"❌ SOAK FAILED: {log_path} contains a corrupt line — instrumentation is broken."
            day = _snapshot_day(entry)
            if day and start.isoformat() <= day <= end.isoformat():
                snapshots.append((day, entry))

    days_covered = sorted({d for d, _ in snapshots})
    bad = [(d, e) for d, e in snapshots if not e.get("ok")]

    day_number = (now - start).days + 1
    header = f"🧪 Soak: {start} → {end} (7 days), today is day {min(max(day_number, 0), 99)}"
    lines.append(header)
    lines.append(f"   Snapshots captured: {len(days_covered)} day(s) — {', '.join(days_covered) or 'none yet'}")

    # --- in progress? ---
    if now <= end:
        if bad:
            for d, e in bad:
                failed = [c["name"] + " (" + c["detail"] + ")" for c in e.get("checks", []) if not c.get("passed")]
                lines.append(f"   ❌ {d}: {'; '.join(failed) or 'snapshot not ok'}")
            lines.append("⚠️  SOAK IN PROGRESS but already FAILING — a daily snapshot came back unhealthy.")
            lines.append("   The failing check above names the problem; fix it and reset the soak clock.")
            return 1, "\n".join(lines)
        lines.append(f"⏳ SOAK IN PROGRESS: healthy so far. Come back after {end} 21:00 for the verdict.")
        return 0, "\n".join(lines)

    # --- complete: grade it ---
    problems: List[str] = []
    warnings: List[str] = []

    # C1: snapshot health & coverage
    if bad:
        for d, e in bad:
            failed = [c["name"] for c in e.get("checks", []) if not c.get("passed")]
            problems.append(f"day {d} snapshot failed ({', '.join(failed)})")
    if len(days_covered) < MIN_SNAPSHOTS:
        problems.append(
            f"only {len(days_covered)}/{total_days} daily snapshots captured — insufficient evidence "
            f"(was the Mac asleep at 21:00?)"
        )
    elif len(days_covered) < total_days:
        warnings.append(f"{len(days_covered)}/{total_days} snapshots captured (missed days likely = Mac asleep)")

    # C2: organic supersession since baseline
    db = os.path.expanduser(cfg["db"])
    facts: List[dict] = []
    contradictions_since = 0
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        contradictions_since = conn.execute(
            "SELECT count(*) FROM dream_audit_history "
            "WHERE agent_id = ? AND contradiction_resolutions_json != '[]' AND timestamp > ?",
            (cfg["agent"], cfg["baseline_taken_at"]),
        ).fetchone()[0]
        row = conn.execute(
            "SELECT facts_json, rules_json FROM memory_profiles WHERE agent_id = ?", (cfg["agent"],)
        ).fetchone()
        facts = json.loads(row[0]) if row else []
        rules = json.loads(row[1]) if row else []
        conn.close()
    except Exception as exc:
        problems.append(f"could not read the memory DB: {exc}")
        rules = []

    if contradictions_since == 0:
        problems.append(
            "no organic contradiction was resolved during the soak — the supersession path never ran "
            "on real usage. Extend the soak, or genuinely correct one of your stored facts and re-check."
        )
    else:
        from remagent.doctor import find_erasure_orphans
        orphans = find_erasure_orphans(facts)
        if orphans:
            problems.append(f"supersession left facts with no active replacement: {', '.join(sorted(set(orphans)))}")

    # C3: injection fits budget
    injection = ""
    try:
        profile = MemoryProfile(
            agent_id=cfg["agent"],
            facts=[Fact(**f) for f in facts],
            rules=[OperationalRule(**r) for r in rules],
        )
        governor = TokenBudgetGovernor()
        try:
            injection = governor.build_budgeted_prompt_injection(profile)
        except GovernorBudgetError as exc:
            injection = ""
            problems.append(f"the recall injection cannot be built — {exc}")
        else:
            tokens = governor.estimate_tokens(injection) if injection else 0
            n_rules = injection.count("\n- [")
            n_facts = injection.count(" (conf: ")
            lines.append(
                f"   Injection: ~{tokens} tokens, {n_rules} rule(s) + {n_facts} fact(s), within budget."
            )
            if "omitted by token budget" in injection:
                lines.append("   (lower-priority items trimmed to fit — expected at scale and "
                             "stated in the injection itself)")
            if not injection.strip():
                problems.append("the recall injection came back empty")
    except Exception as exc:
        problems.append(f"could not build the recall injection: {exc}")

    growth = len(facts) - cfg.get("baseline_fact_count", 0)
    lines.append(f"   Memory growth since baseline: {growth:+d} fact(s), "
                 f"{contradictions_since} contradiction resolution(s).")
    for w in warnings:
        lines.append(f"   ⚠️  {w}")

    if problems:
        lines.append("")
        lines.append("❌ SOAK FAILED:")
        for p in problems:
            lines.append(f"   • {p}")
        return 1, "\n".join(lines)

    lines.append("")
    lines.append("✅ SOAK PASSED on every machine-gradable criterion.")
    lines.append("   One human check remains — read this injection and ask: would it genuinely")
    lines.append("   help a fresh session? If yes, the soak is fully passed; proceed to release.")
    lines.append(injection or "   (injection is empty — that alone should give you pause)")
    return 0, "\n".join(lines)
