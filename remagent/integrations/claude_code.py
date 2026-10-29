"""
Claude Code MCP (Model Context Protocol) Server for RemAgent.
Enables Claude Code in terminals and IDEs to leverage zero-vector biological memory,
deterministic prompt injections, and background REM consolidation.
"""

import json
from typing import Any, Dict, List, Optional
from remagent.schemas import RawTurnLog, MemoryProfile, Fact, OperationalRule
from remagent.storage.sqlite import SQLiteStorageAdapter
from remagent.engine.synthesizer import DreamSynthesizer
from remagent.daemon import ConsolidationBusyError, DreamDaemon
from remagent.governor import TokenBudgetGovernor

try:
    from mcp.server import MCPServer
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        MCPServer = None  # type: ignore


def create_claude_mcp_server(
    name: str = "remagent",
    default_db_path: str = "remagent_memory.db",
    default_agent_id: str = "claude_code",
) -> Any:
    """
    Creates and configures the RemAgent MCP server with recall, log, and dream tools.
    """
    if MCPServer is None:
        raise ImportError(
            "The 'mcp' package is required to run the RemAgent Claude Code MCP server.\n"
            "Please install it using: pip install 'remagent[claude]' or pip install mcp"
        )

    server = MCPServer(name)

    @server.tool(
        name="remagent_recall",
        description=(
            "Recalls active consolidated entity facts, operational directives, and prioritized "
            "coding rules from RemAgent's zero-vector memory within a strict token budget."
        ),
    )
    async def remagent_recall(
        query_context: str = "",
        max_tokens: int = 6000,
        agent_id: str = default_agent_id,
        db_path: str = default_db_path,
    ) -> str:
        """Recall high-signal consolidated facts and rules for prompt injection."""
        storage = SQLiteStorageAdapter(db_path=db_path)
        await storage.initialize()
        try:
            profile: MemoryProfile = await storage.load_memory_profile(agent_id=agent_id)
            governor = TokenBudgetGovernor(default_max_tokens=max_tokens)
            prompt_injection = governor.build_budgeted_prompt_injection(
                profile=profile,
                query_context=query_context if query_context else None,
                max_tokens=max_tokens,
            )

            active_facts = [f for f in profile.facts if f.is_active]
            active_rules = [r for r in profile.rules if r.is_active]
            active_rules.sort(key=lambda r: r.priority)

            response_data = {
                "status": "success",
                "agent_id": agent_id,
                "query_context": query_context,
                "prompt_injection": prompt_injection,
                "facts": [f.model_dump() for f in active_facts],
                "rules": [r.model_dump() for r in active_rules],
                "total_active_facts": len(active_facts),
                "total_active_rules": len(active_rules),
                "last_dream_at": profile.last_dream_at,
            }
            return json.dumps(response_data, indent=2)
        finally:
            await storage.close()

    @server.tool(
        name="remagent_log",
        description=(
            "Appends a raw developer or agent interaction turn to RemAgent's episodic memory buffer "
            "for subsequent REM sleep consolidation."
        ),
    )
    async def remagent_log(
        role: str,
        content: str,
        session_id: str = "claude_code",
        agent_id: str = default_agent_id,
        db_path: str = default_db_path,
        turn_id: str = "",
    ) -> str:
        """Log a conversational or tool execution turn into the episodic buffer.
        turn_id (optional): a stable id; logging the same id twice stores it once and reports "duplicate"."""
        valid_roles = ["user", "assistant", "system", "tool"]
        normalized_role = role.lower()
        if normalized_role not in valid_roles:
            # Never silently coerce bad input into a "logged" success.
            raise ValueError(
                f"Invalid role {role!r}; must be one of {valid_roles}. The turn was NOT logged."
            )

        storage = SQLiteStorageAdapter(db_path=db_path)
        await storage.initialize()
        try:
            turn_kwargs = {"session_id": session_id, "role": normalized_role, "content": content}
            if turn_id:
                turn_kwargs["turn_id"] = str(turn_id).strip()
            turn = RawTurnLog(**turn_kwargs)  # type: ignore[arg-type]
            stored = await storage.save_turn(turn)
            response_data = {
                "status": "logged" if stored else "duplicate",
                "secrets_redacted": turn.content != content,
                "turn_id": turn.turn_id,
                "session_id": session_id,
                "role": normalized_role,
                "timestamp": str(turn.timestamp),
            }
            return json.dumps(response_data, indent=2)
        finally:
            await storage.close()

    @server.tool(
        name="remagent_dream",
        description=(
            "Triggers an immediate background REM sleep consolidation pass to prune noise, "
            "extract discrete entity facts, resolve contradictions, and synthesize rules."
        ),
    )
    async def remagent_dream(
        agent_id: str = default_agent_id,
        db_path: str = default_db_path,
    ) -> str:
        """Trigger an immediate REM consolidation cycle."""
        storage = SQLiteStorageAdapter(db_path=db_path)
        await storage.initialize()
        try:
            synthesizer = DreamSynthesizer()
            daemon = DreamDaemon(storage=storage, synthesizer=synthesizer, agent_id=agent_id)
            try:
                result = await daemon.consolidate_now()
            except ConsolidationBusyError as exc:
                # Busy is NOT "up to date": turns remain queued for retry.
                return json.dumps({"status": "busy", "message": str(exc)}, indent=2)
            if result:
                response_data = {
                    "status": "consolidated",
                    "run_id": result.run_id,
                    "added_facts_count": len(result.added_facts),
                    "updated_rules_count": len(result.updated_rules),
                    "pruned_noise_count": result.pruned_noise_count,
                    "estimated_token_savings": result.estimated_token_savings,
                    "reasoning_summary": result.reasoning_summary,
                    "timestamp": str(result.timestamp),
                }
            else:
                response_data = {
                    "status": "skipped",
                    "message": "No unconsolidated turns to process. Memory is up to date.",
                }
            return json.dumps(response_data, indent=2)
        finally:
            await storage.close()

    return server


def main():
    """CLI entrypoint for running the RemAgent MCP server over stdio."""
    import argparse
    parser = argparse.ArgumentParser(description="RemAgent Model Context Protocol (MCP) Server for Claude Code")
    parser.add_argument("--db", default="remagent_memory.db", help="Path to SQLite memory database")
    parser.add_argument("--agent", default="claude_code", help="Agent identifier")
    args = parser.parse_args()

    server = create_claude_mcp_server(
        name="remagent",
        default_db_path=args.db,
        default_agent_id=args.agent,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
