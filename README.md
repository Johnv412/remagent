

 # RemAgent 🧠💤

**One memory across all your agents.**

A zero-vector, local-first shared brain: the same consolidated memory, reachable from Claude Code, Gemini-powered agents, Hermes, any MCP host, or plain CLI. Modeled on biological sleep / REM memory consolidation.

[![CI](https://github.com/Johnv412/rem-agent-/actions/workflows/ci.yml/badge.svg)](https://github.com/Johnv412/rem-agent-/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/remagent)](https://pypi.org/project/remagent/)

---

## 🔌 One Config Line, Any MCP Host

RemAgent ships an MCP server. One entry in any MCP-capable host — Claude Code, Gemini CLI, an IDE agent, your own framework — attaches it to the same shared memory:

```json
{
  "mcpServers": {
    "remagent": {
      "command": "remagent-mcp",
      "args": ["--db", "/absolute/path/to/memory.db", "--agent", "me"]
    }
  }
}
```

Every host that points at the same database file shares one brain: `remagent_recall` injects the consolidated graph, `remagent_log` captures new facts, `remagent_dream` consolidates. Seven IDE windows, a CLI script, and a Python agent loop can all remember — and correct — the same things.

## ⚡ The Problem: Vector Databases Are Noisy, Brittle & Expensive

Traditional AI agent architectures rely on Vector RAG (Retrieval-Augmented Generation). In long-running, autonomous agent workloads, vector databases fail in predictable ways:

| Failure Mode in Vector RAG | How RemAgent Resolves It |
| --- | --- |
| **Semantic Drift & Chaff Bloat** — RAG stores every pleasantry ("Thanks!"), failed tool trace, and typo as embeddings. | RemAgent prunes ephemeral noise during background sleep cycles, keeping only durable facts and rules. |
| **Contradiction Paralysis** — a user says "Use MySQL" on Day 1 and "Switch to Postgres" on Day 2; RAG retrieves both chunks and the agent guesses. | RemAgent explicitly resolves contradictions and marks obsolete facts as superseded, with a pointer to what replaced them. |
| **Token Bloat & Cost** — injecting ten raw text chunks burns thousands of context tokens per query. | RemAgent injects a tight, deterministic graph of attributed facts, filtered to a configurable token budget. |
| **Zero Cognitive Synthesis** — vector DBs are indexers; they never learn anything. | RemAgent synthesizes behavioral heuristics and operational directives from raw session history. |

The core bet: for a single agent or team, **a small, legible, versioned store of resolved facts beats a large pile of embeddings.** You can `cat` it, `diff` it, and audit it — no retrieval ranking to debug.

## 🆚 Why Not Mem0, Zep, or Letta?

Those are excellent projects, and if you need hosted, multi-tenant, vector-hybrid memory at scale, use them. RemAgent makes different trade-offs on purpose:

- **Zero-vector.** No embeddings, no similarity search, no "why did it rank the wrong chunk third." Facts are structured records you can read.
- **Local-first.** Memory is a SQLite file on your machine by default. Nothing leaves your computer except consolidation calls to the LLM (see Privacy below).
- **Consolidation-centered.** The sleep cycle — pruning, contradiction resolution, heuristic extraction — is the product, not a bolt-on.
- **Small enough to audit.** One Python package, one database file, plain schemas.

If your memory problem is "one developer / one team, long-running agents, facts that change" — that's what this is built for.

## 🤝 And Claude Code's Native Auto Dream?

Claude Code shipped Auto Dream in 2026: native background consolidation of its per-project markdown memory — merging duplicate notes, deleting contradicted ones, pruning stale entries. It's a genuinely good feature, it validates the sleep-consolidation metaphor, and if you only use Claude Code in one project, it may be all you need. RemAgent occupies the ground it deliberately doesn't:

| | Claude Code Auto Dream | RemAgent |
| --- | --- | --- |
| **Reach** | Claude Code, per-project | One shared memory across Claude Code, Gemini, Hermes, any MCP host, CLI, and Python — cross-project if you point them at one DB |
| **Memory form** | Markdown note files with a size-capped index | Structured entity–attribute facts with confidence scores and explicit `superseded_by` pointers — queryable, plus a generated markdown mirror for humans |
| **Contradictions** | Contradicted notes are deleted during consolidation | Old facts are kept, marked inactive, and linked to their replacement; an enforced invariant guarantees an update never erases knowledge |
| **Integrity** | Consolidation runs in the background; its internals aren't exposed for audit (as of this writing) | Every dream writes an audit row (what was added, what superseded what, the model's reasoning); `remagent doctor` self-audits the whole pipeline on demand |

They compose rather than compete: `remagent init-claude` detects native auto-memory and says so — native keeps handling that repo's own notes while RemAgent runs the cross-agent shared brain on top.

## 🧬 How It Works: The REM Sleep Cycle

RemAgent mimics mammalian memory consolidation:

```
[Agent Awake] ──> Logs raw episodic turns (user, tools, LLM) into a buffer
                       │
                       ▼  (idle timer or scheduled trigger — see Dream Modes)
[Dream Daemon] ──> Wakes in background (zero latency to the active user)
                       │
                       ├─► 1. NOISE PRUNING          (discards ephemeral tool traces, chatter)
                       ├─► 2. FACT EXTRACTION         (builds an entity-attribute graph)
                       ├─► 3. CONTRADICTION RESOLUTION (supersedes stale beliefs with new truth)
                       └─► 4. OPERATIONAL HEURISTICS   (extracts durable directives)
                       │
                       ▼
[Structured Memory Graph]  (zero-vector, deterministic — SQLite or Firestore)
                       │
                       ▼  (next session / next query)
[Agent Recall] ──> Deterministic memory injection into the prompt, within token budget
```

### Dream Modes

RemAgent consolidates in two ways, and you can use either or both:

1. **Idle-trigger (embedded).** The `DreamDaemon` runs inside your Python process and dreams after a configurable period of agent inactivity. Best for long-running agent loops.
2. **Scheduled (system-level).** For Claude Code and desktop use, `remagent init-claude` can wire dreams to session lifecycle hooks, and consolidation can also run on a schedule (launchd on macOS, cron elsewhere). Best for capturing work across many short sessions.

A note on cadence: consolidation benefits from seeing *batches* of sessions. Dreaming too frequently over a young memory store mostly re-processes the same entries. Nightly is a sensible default for scheduled mode; tune from there.

## 🔒 Privacy & Data Handling

Read this before installing. RemAgent captures interaction turns — which for developers can include code, file paths, client names, and anything else you type.

- **What's stored:** raw turns and consolidated facts, in a local SQLite database (`memory.db`) you own. Nothing is stored by RemAgent anywhere else.
- **What leaves your machine:** during a dream cycle, buffered turns are sent to your selected LLM provider (Gemini, Anthropic, OpenAI-compatible, or xAI) for consolidation. That is the only network egress. If your sessions may contain secrets or client-confidential material, treat this the same way you'd treat any LLM API usage — and don't log what you can't send.
- **Purging:** delete the database file, or use `remagent decay` to age out low-confidence facts. Superseded facts retain history until purged.
- **Git:** the memory database and the markdown mirror (`remagent export --markdown`) are added to `.gitignore` by the `init-claude` scaffold. **Committing agent memory to git is opt-in** — it may contain sensitive session content (code, paths, client names), so only remove those ignore rules deliberately.
- **Scope:** single-node SQLite is the product today — one database file, one owner. The storage layer is a small adapter interface designed to extend to multi-tenant backends; an experimental Firestore adapter ships in the `[firestore]` extra, but treat anything beyond local SQLite as unproven until documented otherwise.

## 📦 Installation

```bash
# Standard installation (SQLite + Gemini support included)
pip install remagent

# Add an LLM provider for the dream synthesizer (pick the one you have a key for)
pip install "remagent[anthropic]"   # Anthropic Claude
pip install "remagent[openai]"      # OpenAI, any OpenAI-compatible server, and xAI (Grok)
pip install "remagent[gemini]"      # Google Gemini (alias — already in the base install for 1.x)

# Claude Code terminal & IDE integration (MCP server + hooks)
pip install "remagent[claude]"

# Experimental Google Cloud Firestore storage adapter (single-node SQLite
# is the supported product today)
pip install "remagent[firestore]"
```

Or from source: `git clone https://github.com/Johnv412/rem-agent-.git && cd rem-agent- && pip install -e ".[claude]"`

### Choose an LLM provider — four keys, pick one

Dream consolidation needs exactly one API key. Export the one you have:

```bash
export ANTHROPIC_API_KEY="..."   # Anthropic Claude        (pip install "remagent[anthropic]")
export OPENAI_API_KEY="..."      # OpenAI                  (pip install "remagent[openai]")
export XAI_API_KEY="..."         # xAI Grok                (pip install "remagent[openai]")
export GEMINI_API_KEY="..."      # Google Gemini           (included in the base install)
```

With no `REMAGENT_PROVIDER` set, RemAgent auto-detects the first key present in this order: `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → `XAI_API_KEY` → `GEMINI_API_KEY`.

| Variable | Purpose |
|---|---|
| `REMAGENT_PROVIDER` | Force a provider: `gemini`, `anthropic`, `openai`, or `xai`. Overrides auto-detection. |
| `REMAGENT_MODEL` | Override the selected provider's default model (e.g. `claude-opus-5`, `gpt-5.5`, `grok-4.5`, `gemini-2.5-pro`). |
| `OPENAI_BASE_URL` | Point the `openai` provider at any OpenAI-compatible server (Ollama, vLLM, LM Studio, a proxy). Applies to `openai` only — `xai` is always `https://api.x.ai/v1`. |

Default models: Gemini `gemini-2.5-flash`, Anthropic `claude-sonnet-5`, OpenAI `gpt-6-astra`, xAI `grok-4.6`. Set `REMAGENT_MODEL=claude-opus-5` (or any other model id) to override.

xAI is not a separate backend: it is the OpenAI-compatible backend with `XAI_API_KEY` and the xAI base URL, so it needs the `[openai]` extra and nothing else.

**Verified against live APIs:** Gemini, Anthropic, and xAI. OpenAI is wired and unit-tested against a mocked SDK, but no request has ever been made to the real OpenAI API — treat that path as unproven until you exercise it yourself.

Every provider receives the same consolidation prompt and goes through the same parser, so the supersession invariant (old fact inactive with a `superseded_by` pointer to the new active fact) is enforced identically. If a provider's response cannot be parsed, the dream fails loudly naming the provider and writes nothing. `remagent doctor` reports the active provider, which keys are present (names only), and whether that provider's SDK is installed.

## 🚀 Quickstart (Python 3.11+)

```python
import asyncio
from remagent import DreamDaemon, SQLiteStorageAdapter, DreamSynthesizer, RawTurnLog

async def main():
    # 1. Initialize local SQLite storage
    storage = SQLiteStorageAdapter("my_agent_memory.db")
    await storage.initialize()

    # 2. Start the autonomous background Dream Daemon
    daemon = DreamDaemon(
        storage=storage,
        idle_threshold_seconds=15.0,  # dreams after 15s of agent inactivity
    )
    await daemon.start()

    # 3. Log agent interaction turns
    await storage.save_turn(RawTurnLog(
        role="user",
        content="Hey! For this project, let's use PostgreSQL instead of SQLite, and enable strict TypeScript mode."
    ))
    daemon.record_activity()

    # 4. Trigger an immediate dream cycle (or let the daemon dream during idle)
    result = await daemon.consolidate_now()

    print(f"✨ Consolidation: {result.reasoning_summary}")
    print(f"   Added facts: {len(result.added_facts)}")
    print(f"   Updated rules: {len(result.updated_rules)}")
    print(f"   Estimated token savings: ~{result.estimated_token_savings} tokens")

    # 5. Inspect consolidated memory
    profile = await storage.load_memory_profile()
    for fact in profile.facts:
        if fact.is_active:
            print(f"📌 {fact.entity}.{fact.attribute} = {fact.value} (conf: {fact.confidence})")

    await daemon.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🤖 Hermes Agent Framework Integration

RemAgent provides a first-class connector and tool integration for autonomous agent loops:

```python
from remagent.integrations.hermes import HermesMemoryConnector, RemAgentTool

connector = HermesMemoryConnector(agent_id="coding_assistant")
await connector.initialize()

# 1. Before generating a response, inject recalled memory into the system prompt
memory_context = await connector.get_system_prompt_injection(
    query_context="database configuration"
)

# 2. When the user or agent speaks, log the turn (resets the idle clock)
await connector.log_interaction(
    role="user",
    content="Remember to always run tests with pytest before committing."
)

# 3. Expose memory to the LLM as a tool
tool = RemAgentTool(connector)
tool_definition = tool.get_tool_schema()
```

## 🧠 Claude Code Integration

RemAgent has native support for Claude Code (Anthropic's agentic CLI). Developers running Claude Code gain autonomous zero-vector memory, deterministic memory injection at session start, and background consolidation across sessions.

### 2-Step Setup

```bash
# 1. Install RemAgent with Claude MCP support
pip install "remagent[claude]"

# 2. Scaffold .claude/settings.json and lifecycle hooks
remagent init-claude
```

By default `init-claude` configures the current repository. To share one memory across every Claude Code session on the machine, install the hooks globally (see docs) so all projects write to the same store.

### What init-claude configures

```json
{
  "mcpServers": {
    "remagent": {
      "command": "remagent-mcp",
      "args": ["--db", "<db>", "--agent", "<agent>"]
    }
  },
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command", "command": "<python> <repo>/.claude/hooks/session_start.py" } ] }
    ],
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "<python> <repo>/.claude/hooks/user_prompt_submit.py" } ] }
    ]
  }
}
```

Three details matter, and each one was a real bug:

- **Matcher-group shape.** Claude Code expects `[{ "hooks": [ ... ] }]`. A flat
  `[{ "type": "command", ... }]` list parses without error but never fires.
- **Absolute paths.** Hooks do not reliably run from the repository root, so a
  relative `.claude/hooks/...` command silently fails to resolve.
- **No `Stop` hook.** Consolidation is a paid LLM call. Firing one at the end of
  every session is expensive and duplicates scheduled consolidation — run
  `remagent dream` from launchd or cron instead.

`SessionStart` writes the full recall injection to `<db_stem>_context.md` and puts
a compact digest plus a pointer to that file into the session, because Claude Code
truncates hook output to roughly 2KB — well under a real brain's size.

### Exposed MCP Tools

- **`remagent_recall`** — recalls active entity facts and prioritized operational directives, filtered to the prompt token budget.
- **`remagent_log`** — appends raw developer or agent interaction turns to the unconsolidated buffer.
- **`remagent_dream`** — triggers an immediate consolidation pass: extract facts, resolve contradictions, prune noise.

## 🛠️ CLI Usage

```bash
# Scaffold Claude Code hooks and settings in the current repository
remagent init-claude

# Recall consolidated memory context for prompt injection
remagent recall --format injection

# Trigger an immediate consolidation pass
remagent dream --db my_agent_memory.db

# Inspect active facts, directives, and estimated token savings
remagent status --db my_agent_memory.db

# Run the built-in observation-period check: prints a plain-English
# PASSED / FAILED / IN PROGRESS verdict on memory health over time
remagent soak

# Append a raw turn from bash/scripts
remagent log --role user --content "Deploying to us-west2 region."

# Apply Ebbinghaus temporal decay to long-dormant, low-confidence facts
remagent decay --db my_agent_memory.db --half-life-days 30 --floor 0.2

# Export active memory as a readable markdown mirror (one file per entity,
# rules in rules.md) — cat/grep/diff your agent's memory. The database stays
# the source of truth; add --export-md to `remagent dream` to regenerate the
# mirror after every dream cycle.
remagent export --markdown --db my_agent_memory.db

# Self-audit the installation (hooks, jobs, database integrity)
remagent doctor
```

## 🏛️ Architecture & Schemas

### Fact

| Field | Description |
| --- | --- |
| `entity` | Target entity (e.g. `User`, `ProjectBackend`, `AuthService`) |
| `attribute` | Specific property (e.g. `database`, `port`, `framework`) |
| `value` | Ground-truth value (e.g. `PostgreSQL`, `3000`, `Express`) |
| `confidence` | Confidence score, 0.0–1.0 |
| `superseded_by` | Optional ID of the fact that invalidated this record |
| `is_active` | Boolean state flag |

### OperationalRule

| Field | Description |
| --- | --- |
| `category` | `user_preference` \| `coding_standard` \| `architecture_heuristic` \| `operational_directive` \| `domain_constraint` |
| `rule` | Concise, actionable heuristic |
| `rationale` | Reasoning behind the rule |
| `priority` | 1 (critical) to 5 (minor) |

Every fact carries provenance (where it came from), a supersession chain (what replaced it), and decay (how it ages out). Memory you can't audit is just configuration with extra steps.

## 🖥️ Demo Dashboard

The repo contains a web dashboard (Vite + React + Express) that visualizes the dream cycle. Run it locally with `npm run dev`. **It is a demo playground running on seeded, in-memory simulation data** — not a live view of a real RemAgent database, and nothing it shows is read from your `memory.db`. Its Gemini-backed features (consolidation, agent chat) require `GEMINI_API_KEY` and return explicit errors without it.

There is no public hosted instance. The dashboard has no authentication of any kind, so exposing it publicly would let anyone spend your LLM quota — run it locally only.

## 🧪 Testing status

What is actually covered, and what is not. These gaps are known and deliberate rather than
oversights, so they are written down instead of discovered later.

**Covered:** the unit suite runs on every push and mocks each provider at the SDK boundary, so
all four backends are checked for identical parsing and the same supersession invariant. A
nightly canary exercises the **real** Gemini API end to end — logging `$40`, then `$52`, and
asserting the old fact ends inactive with `superseded_by` pointing at the new active fact.

**Known gaps:**

- **Anthropic canary skips.** The nightly workflow has an Anthropic job, but `ANTHROPIC_API_KEY`
  is not in repo secrets, so it skips cleanly (green, not red) on every run. Anthropic was
  verified live once by hand; it is not verified continuously.
- **No OpenAI canary.** The OpenAI path has never been run against the real API at all.
- **The live-Gemini unit test never runs in CI.** It is gated behind `RUN_LIVE_GEMINI=1`, which CI
  does not set. The nightly canary covers that path instead, which is why the gate is left alone.

A green suite therefore means "the parsing and supersession logic is sound for every provider,"
not "every provider has been proven against its live API."

## 🗺️ Roadmap

- [x] PyPI release (`pip install remagent`)
- [x] Additional LLM providers for the synthesizer (Anthropic, OpenAI-compatible, xAI)
- [ ] Global (machine-wide) Claude Code hook install as a first-class command
- [ ] Team-shared memory stores with per-agent namespaces
- [ ] Reproducible benchmark suite for pruning and recall quality

## 👤 Maintainer

Built by **John Vincent** ([@Johnv412](https://github.com/Johnv412)) — JV.AI Systems. I build production AI voice and automation systems for service businesses; RemAgent is the memory layer extracted from that work. Issues and PRs welcome.

Built with Claude Code.

## 📄 License

Apache-2.0.
