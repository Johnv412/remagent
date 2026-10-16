"""
LLM provider selection and backends for dream consolidation.

Exactly three backends exist: gemini (google-genai, base dependency), anthropic
(official Anthropic SDK, extra remagent[anthropic]) and openai (official OpenAI
SDK, extra remagent[openai]). xAI is NOT a separate backend: it is the openai
backend pointed at https://api.x.ai/v1 with XAI_API_KEY.

Selection (see resolve_provider):
  1. REMAGENT_PROVIDER, if set: gemini | anthropic | openai | xai
  2. else the first present key in ANTHROPIC_API_KEY -> OPENAI_API_KEY ->
     XAI_API_KEY -> GEMINI_API_KEY (GOOGLE_API_KEY accepted as a Gemini alias)
  3. OPENAI_BASE_URL applies to the openai provider only; xai is pinned.
  4. REMAGENT_MODEL overrides the selected provider's default model.

SDKs are imported lazily inside the backend that needs them so a base install
never requires anthropic or openai.
"""

import importlib.util
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

from remagent.engine.errors import (
    DreamSynthesisError,
    NoProviderKeyError,
    ProviderConfigError,
    ProviderNotInstalledError,
)

XAI_BASE_URL = "https://api.x.ai/v1"

PROVIDERS = ("gemini", "anthropic", "openai", "xai")
AUTODETECT_ORDER = ("anthropic", "openai", "xai", "gemini")

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
ALL_KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY")

BACKEND_FOR_PROVIDER = {
    "gemini": "gemini",
    "anthropic": "anthropic",
    "openai": "openai",
    "xai": "openai",
}

# Default chat models, verified in-session (2026-09-06) against provider docs
# and the installed SDK model literals. See the provider report for sources.
DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-sonnet-5",   # claude-opus-5 available via REMAGENT_MODEL
    "openai": "gpt-6-astra",
    "xai": "grok-4.6",
}

# Tried once if the default model fails outright. Only applies to the code
# default: a model the operator set via REMAGENT_MODEL is never swapped behind
# their back. Mirrors Dream Engine, which records fallback_from on the audit row.
FALLBACK_MODELS = {
    "gemini": "gemini-2.5-flash",
}

SDK_MODULE = {"gemini": "google.genai", "anthropic": "anthropic", "openai": "openai"}
EXTRA_NAME = {"gemini": "gemini", "anthropic": "anthropic", "openai": "openai"}

_KEY_HINT = (
    "set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, "
    "or set REMAGENT_PROVIDER to choose a provider explicitly"
)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str            # gemini | anthropic | openai | xai
    backend: str             # gemini | anthropic | openai
    api_key: str
    model: str
    key_env: str             # name of the env var the key came from (never the value)
    base_url: Optional[str] = None
    fallback_model: Optional[str] = None   # set only when `model` is the code default


def _gemini_key(env: Mapping[str, str]) -> Optional[tuple]:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if env.get(name):
            return name, env[name]
    return None


def _key_for(provider: str, env: Mapping[str, str]) -> Optional[tuple]:
    if provider == "gemini":
        return _gemini_key(env)
    name = KEY_ENV[provider]
    return (name, env[name]) if env.get(name) else None


def present_keys(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Names (never values) of the provider keys present, in autodetect order."""
    env = os.environ if env is None else env
    found = []
    for provider in AUTODETECT_ORDER:
        hit = _key_for(provider, env)
        if hit:
            found.append(hit[0])
    return found


def resolve_provider(env: Optional[Mapping[str, str]] = None) -> ProviderConfig:
    """Pick the active provider from the environment. Raises ProviderConfigError
    (a DreamSynthesisError) with a one-line, secret-free message on failure."""
    env = os.environ if env is None else env
    explicit = (env.get("REMAGENT_PROVIDER") or "").strip().lower()

    if explicit:
        if explicit not in PROVIDERS:
            raise ProviderConfigError(
                f"REMAGENT_PROVIDER={explicit!r} is not supported; use one of gemini, anthropic, openai, xai"
            )
        hit = _key_for(explicit, env)
        if not hit:
            raise NoProviderKeyError(
                f"REMAGENT_PROVIDER={explicit} but {KEY_ENV[explicit]} is not set; {_KEY_HINT}"
            )
        provider = explicit
    else:
        provider = None
        hit = None
        for candidate in AUTODETECT_ORDER:
            hit = _key_for(candidate, env)
            if hit:
                provider = candidate
                break
        if provider is None:
            raise NoProviderKeyError(f"no LLM API key found; {_KEY_HINT}")

    key_env, api_key = hit
    explicit_model = (env.get("REMAGENT_MODEL") or "").strip()
    model = explicit_model or DEFAULT_MODELS[provider]
    fallback = None if explicit_model else FALLBACK_MODELS.get(provider)
    base_url: Optional[str] = None
    if provider == "xai":
        base_url = XAI_BASE_URL
    elif provider == "openai":
        base_url = (env.get("OPENAI_BASE_URL") or "").strip() or None

    return ProviderConfig(
        provider=provider,
        backend=BACKEND_FOR_PROVIDER[provider],
        api_key=api_key,
        model=model,
        key_env=key_env,
        base_url=base_url,
        fallback_model=fallback,
    )


def legacy_gemini_config(api_key: str, model: Optional[str] = None) -> ProviderConfig:
    """DreamSynthesizer(api_key=...) has always meant 'Gemini with this key'."""
    return ProviderConfig(
        provider="gemini", backend="gemini", api_key=api_key,
        model=model or DEFAULT_MODELS["gemini"], key_env="api_key",
        fallback_model=None if model else FALLBACK_MODELS.get("gemini"),
    )


def sdk_importable(backend: str) -> bool:
    module = SDK_MODULE[backend]
    try:
        top = module.split(".")[0]
        if importlib.util.find_spec(top) is None:
            return False
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _not_installed(backend: str) -> ProviderNotInstalledError:
    return ProviderNotInstalledError(
        f"the {backend} provider needs its SDK extra: pip install \"remagent[{EXTRA_NAME[backend]}]\""
    )


class _Backend:
    """Common shape: one call per dream, returning the raw text the shared parser consumes."""
    name: str = ""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model = config.model
        self.api_key = config.api_key
        self._client: Any = None

    def ensure_sdk(self) -> None:
        if not sdk_importable(self.config.backend):
            raise _not_installed(self.config.backend)

    def generate(self, system_prompt: str, user_prompt: str, schema_model, json_schema: Dict[str, Any]) -> str:
        raise NotImplementedError


class GeminiBackend(_Backend):
    """Existing Gemini path. Behaviour is unchanged from 1.0.x."""
    name = "gemini"

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as exc:
                raise _not_installed("gemini") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, system_prompt, user_prompt, schema_model, json_schema) -> str:
        client = self._get_client()
        # Send a sanitised schema, not the Pydantic model. google-genai rejects
        # any `default` in a response schema, and the model carries four of
        # them, so passing it directly fails before the request leaves.
        # Imported here to keep this module free of a prompting dependency at
        # import time.
        from remagent.engine.prompting import gemini_response_schema

        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": gemini_response_schema(),
            },
        )
        return response.text


class AnthropicBackend(_Backend):
    name = "anthropic"
    max_tokens = 16000

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise _not_installed("anthropic") from exc
            # Retries are owned by the synthesizer's bounded loop.
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=0)
        return self._client

    def generate(self, system_prompt, user_prompt, schema_model, json_schema) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": json_schema}},
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise DreamSynthesisError("anthropic declined the consolidation request (stop_reason=refusal)")
        parts = [
            getattr(block, "text", "")
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)


class OpenAIBackend(_Backend):
    """OpenAI, any OpenAI-compatible server (OPENAI_BASE_URL), and xAI."""
    name = "openai"

    def _get_client(self):
        if self._client is None:
            try:
                import openai  # type: ignore
            except ImportError as exc:
                raise _not_installed("openai") from exc
            kwargs: Dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def generate(self, system_prompt, user_prompt, schema_model, json_schema) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "dream_synthesis", "schema": json_schema, "strict": True},
            },
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise DreamSynthesisError(f"{self.config.provider} returned no choices")
        message = choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise DreamSynthesisError(f"{self.config.provider} declined the consolidation request: {refusal}")
        return message.content or ""


_BACKENDS = {
    "gemini": GeminiBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}


def make_backend(config: ProviderConfig) -> _Backend:
    """Construct the backend for a config. Verifies the SDK extra is importable
    up front so the CLI can fail with a one-line install hint."""
    backend = _BACKENDS[config.backend](config)
    backend.ensure_sdk()
    return backend


def with_model(config: ProviderConfig, model: str) -> ProviderConfig:
    return replace(config, model=model)
