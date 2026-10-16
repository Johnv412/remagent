"""
DreamSynthesizer: Autonomous cognitive consolidation engine.
Emulates biological sleep/REM consolidation: prunes conversational noise, extracts discrete
entity facts, resolves contradictions, and updates operational heuristics without vector databases.

The LLM behind a dream is chosen by remagent.engine.providers (Gemini, Anthropic,
OpenAI-compatible incl. xAI). Prompt and parser are shared across all backends
(remagent.engine.prompting), so every provider must yield the same structured type.
"""

import asyncio
import logging
from typing import List, Optional

from remagent.engine.errors import (  # noqa: F401  (re-exported)
    DreamSynthesisError,
    NoProviderKeyError,
    ProviderConfigError,
    ProviderNotInstalledError,
)
from remagent.engine.prompting import (  # noqa: F401  (re-exported)
    DREAM_PROMPT_SYSTEM,
    DreamSynthesisOutput,
    SynthesizedContradiction,
    SynthesizedFact,
    SynthesizedRule,
    build_user_prompt,
    dream_output_json_schema,
    parse_dream_output,
)
from remagent.engine.providers import (
    ProviderConfig,
    legacy_gemini_config,
    make_backend,
    resolve_provider,
    with_model,
)
from remagent.schemas import (
    Fact,
    OperationalRule,
    RawTurnLog,
    MemoryProfile,
    DreamConsolidationResult,
    ContradictionResolution,
    generate_uuid,
    current_utc_iso,
)


logger = logging.getLogger("remagent.synthesizer")

RULE_CATEGORIES = [
    "user_preference",
    "coding_standard",
    "architecture_heuristic",
    "operational_directive",
    "domain_constraint",
]


class DreamSynthesizer:
    """
    Consolidates raw turns into structured memory using the selected LLM provider.

    Construction never touches the network or imports an SDK. The provider is
    resolved lazily on first use unless a ProviderConfig / backend is injected:
      - DreamSynthesizer()                      -> resolve from environment
      - DreamSynthesizer(api_key="...")         -> Gemini with that key (1.0.x behaviour)
      - DreamSynthesizer(provider=cfg, backend=b) -> explicit (used by the CLI)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        provider: Optional[ProviderConfig] = None,
        backend=None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff_seconds = retry_backoff_seconds
        self._config: Optional[ProviderConfig] = provider
        self._backend = backend
        if provider is not None and model_name:
            self._config = with_model(provider, model_name)

    # -- provider resolution -------------------------------------------------

    def _resolve_config(self) -> ProviderConfig:
        if self._config is None:
            if self.api_key:
                self._config = legacy_gemini_config(self.api_key, self.model_name)
            else:
                cfg = resolve_provider()
                self._config = with_model(cfg, self.model_name) if self.model_name else cfg
        return self._config

    def _get_backend(self):
        if self._backend is None:
            self._backend = make_backend(self._resolve_config())
        return self._backend

    @property
    def provider_name(self) -> Optional[str]:
        return self._config.provider if self._config else None

    @property
    def _client(self):
        """SDK client of the active backend (kept for tests that inject fakes)."""
        return self._get_backend()._client

    @_client.setter
    def _client(self, value) -> None:
        self._get_backend()._client = value

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Retry only genuinely transient transport failures: timeouts,
        connection drops, 429 rate limits, and 5xx server errors."""
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
            return True
        transient_names = ("Timeout", "ConnectionError", "ConnectionReset", "ServiceUnavailable", "DeadlineExceeded")
        return any(name in type(exc).__name__ for name in transient_names)

    async def _generate_and_parse(self, backend, provider: str, user_prompt: str, json_schema):
        """One model call with bounded retries for transient faults, then the
        shared parse. Any failure surfaces as DreamSynthesisError so the caller
        can decide about a fallback model; parse errors are never retried."""
        raw_text = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_text = backend.generate(DREAM_PROMPT_SYSTEM, user_prompt, DreamSynthesisOutput, json_schema)
                break
            except DreamSynthesisError:
                raise
            except Exception as exc:
                if attempt < self.max_attempts and self._is_transient_error(exc):
                    delay = self.retry_backoff_seconds * attempt
                    logger.warning(
                        "Transient %s error on attempt %d/%d: %s — retrying in %.1fs",
                        provider, attempt, self.max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Dream synthesis failed; no memory will be written: %s", exc, exc_info=True)
                raise DreamSynthesisError(f"{provider} dream synthesis failed: {exc}") from exc

        try:
            return parse_dream_output(raw_text, provider)
        except DreamSynthesisError as exc:
            logger.error("%s; no memory will be written", exc, exc_info=True)
            raise

    # -- consolidation -------------------------------------------------------

    async def consolidate_window(
        self,
        unconsolidated_turns: List[RawTurnLog],
        existing_profile: MemoryProfile,
    ) -> DreamConsolidationResult:
        """
        Executes a single REM sleep consolidation pass over unconsolidated turns.
        """
        if not unconsolidated_turns:
            return DreamConsolidationResult(
                run_id=generate_uuid(),
                added_facts=[],
                updated_rules=[],
                contradiction_resolutions=[],
                pruned_noise_count=0,
                pruned_noise_reasons=[],
                reasoning_summary="No unconsolidated turns in queue. Dream cycle skipped.",
                consolidated_turn_ids=[],
                timestamp=current_utc_iso(),
                estimated_token_savings=0,
            )

        # Raises ProviderConfigError (a DreamSynthesisError) when no key / no SDK.
        backend = self._get_backend()
        provider = self._resolve_config().provider

        user_prompt, raw_char_count = build_user_prompt(unconsolidated_turns, existing_profile)
        json_schema = dream_output_json_schema()

        primary_model = backend.model
        fallback_model = self._resolve_config().fallback_model
        fallback_from: Optional[str] = None

        try:
            parsed = await self._generate_and_parse(backend, provider, user_prompt, json_schema)
        except DreamSynthesisError as primary_exc:
            if not fallback_model or fallback_model == primary_model:
                raise
            # The code default failed outright — model retired, rejected, or
            # returned nothing usable. Try the configured fallback once, and
            # say so loudly. An operator-set REMAGENT_MODEL never reaches here
            # (resolve_provider leaves fallback_model unset for it).
            logger.warning(
                "%s dream on default model %r failed (%s); retrying once on fallback %r",
                provider, primary_model, primary_exc, fallback_model,
            )
            backend.model = fallback_model
            try:
                parsed = await self._generate_and_parse(backend, provider, user_prompt, json_schema)
            except DreamSynthesisError as fallback_exc:
                raise DreamSynthesisError(
                    f"{provider} dream synthesis failed on default model {primary_model!r} "
                    f"({primary_exc}) and on fallback {fallback_model!r} ({fallback_exc})"
                ) from fallback_exc
            fallback_from = primary_model

        # Map to domain entities
        turn_ids = [t.turn_id for t in unconsolidated_turns]

        added_facts: List[Fact] = []
        for f in parsed.added_facts:
            added_facts.append(
                Fact(
                    entity=f.entity,
                    attribute=f.attribute,
                    value=f.value,
                    confidence=max(0.0, min(1.0, f.confidence)),
                    source_turn_ids=turn_ids,
                    is_active=True,
                )
            )

        updated_rules: List[OperationalRule] = []
        for r in parsed.updated_rules:
            cat = r.category if r.category in RULE_CATEGORIES else "operational_directive"
            updated_rules.append(
                OperationalRule(
                    category=cat,
                    rule=r.rule,
                    rationale=r.rationale,
                    priority=max(1, min(5, r.priority)),
                    is_active=True,
                )
            )

        contradiction_resolutions: List[ContradictionResolution] = []
        for c in parsed.contradictions:
            contradiction_resolutions.append(
                ContradictionResolution(
                    prior_fact_id=None,
                    entity=c.entity,
                    attribute=c.attribute,
                    prior_value=c.prior_value,
                    new_value=c.new_value,
                    resolution_reasoning=c.resolution_reasoning,
                )
            )

        # Estimate token savings (approx 4 chars/token)
        estimated_token_savings = max(0, (raw_char_count // 4) - 80)

        return DreamConsolidationResult(
            run_id=generate_uuid(),
            added_facts=added_facts,
            updated_rules=updated_rules,
            contradiction_resolutions=contradiction_resolutions,
            pruned_noise_count=parsed.pruned_noise_count,
            pruned_noise_reasons=parsed.pruned_noise_categories,
            reasoning_summary=parsed.reasoning_summary,
            consolidated_turn_ids=turn_ids,
            timestamp=current_utc_iso(),
            estimated_token_savings=estimated_token_savings,
            model_name=backend.model,
            fallback_from=fallback_from,
        )
