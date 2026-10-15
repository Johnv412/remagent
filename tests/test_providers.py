"""
Provider-layer tests: selection rules, cross-backend parity (mocked at the SDK
boundary, never at the parser), xAI routing, CLI one-line failures, and the
doctor's secret-free provider report.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from remagent.engine.errors import (
    DreamSynthesisError,
    NoProviderKeyError,
    ProviderConfigError,
    ProviderNotInstalledError,
)
from remagent.engine.prompting import DREAM_PROMPT_SYSTEM
from remagent.engine.providers import DEFAULT_MODELS, XAI_BASE_URL, present_keys, resolve_provider
from remagent.engine.synthesizer import DreamSynthesizer
from remagent.schemas import Fact, MemoryProfile, RawTurnLog

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None

PROVIDER_ENV_VARS = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
    "OPENAI_BASE_URL", "REMAGENT_PROVIDER", "REMAGENT_MODEL",
)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in PROVIDER_ENV_VARS}
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Shared fixture: the MySQL -> Postgres contradiction
# ---------------------------------------------------------------------------

def fixture_turns():
    return [RawTurnLog(role="user", content="Decision: we are migrating the project database from MySQL to PostgreSQL.")]


def fixture_profile():
    return MemoryProfile(agent_id="a", facts=[Fact(entity="Project", attribute="database", value="MySQL")])


PAYLOAD = {
    "added_facts": [
        {"entity": "Project", "attribute": "database", "value": "PostgreSQL", "confidence": 0.95},
    ],
    "updated_rules": [
        {"category": "architecture_heuristic", "rule": "Use PostgreSQL for the project database",
         "rationale": "explicit migration decision", "priority": 2},
    ],
    "contradictions": [
        {"entity": "Project", "attribute": "database", "prior_value": "MySQL", "new_value": "PostgreSQL",
         "resolution_reasoning": "newer turn supersedes"},
    ],
    "pruned_noise_count": 0,
    "pruned_noise_categories": [],
    "reasoning_summary": "Database migrated from MySQL to PostgreSQL.",
}
PAYLOAD_TEXT = json.dumps(PAYLOAD)


def normalise(result):
    """Provider-independent view of a DreamConsolidationResult (ids/timestamps stripped)."""
    return {
        "facts": [(f.entity, f.attribute, f.value, f.confidence, f.is_active) for f in result.added_facts],
        "rules": [(r.category, r.rule, r.rationale, r.priority) for r in result.updated_rules],
        "contradictions": [
            (c.entity, c.attribute, c.prior_value, c.new_value, c.resolution_reasoning)
            for c in result.contradiction_resolutions
        ],
        "pruned": (result.pruned_noise_count, list(result.pruned_noise_reasons)),
        "summary": result.reasoning_summary,
    }


# --- SDK-boundary fakes -----------------------------------------------------

class _GeminiModels:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class _GeminiClient:
    def __init__(self, text):
        self.models = _GeminiModels(text)


class _AnthropicMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=self.text)],
        )


def anthropic_fake(text, stop_reason="end_turn"):
    return SimpleNamespace(messages=_AnthropicMessages(text, stop_reason))


class _OpenAICompletions:
    def __init__(self, text, refusal=None):
        self.text, self.refusal, self.calls = text, refusal, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text, refusal=self.refusal))])


def openai_fake(text, refusal=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=_OpenAICompletions(text, refusal)))


# ---------------------------------------------------------------------------
# Selection rules (pure, env dict in -> config out)
# ---------------------------------------------------------------------------

class TestProviderSelection(unittest.TestCase):

    def test_autodetect_order_anthropic_openai_xai_gemini(self):
        env = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "XAI_API_KEY": "x", "GEMINI_API_KEY": "g"}
        self.assertEqual(resolve_provider(env).provider, "anthropic")
        del env["ANTHROPIC_API_KEY"]
        self.assertEqual(resolve_provider(env).provider, "openai")
        del env["OPENAI_API_KEY"]
        self.assertEqual(resolve_provider(env).provider, "xai")
        del env["XAI_API_KEY"]
        self.assertEqual(resolve_provider(env).provider, "gemini")

    def test_explicit_provider_wins_over_autodetect(self):
        env = {"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g", "REMAGENT_PROVIDER": "gemini"}
        cfg = resolve_provider(env)
        self.assertEqual((cfg.provider, cfg.backend, cfg.key_env), ("gemini", "gemini", "GEMINI_API_KEY"))
        self.assertEqual(cfg.model, DEFAULT_MODELS["gemini"])

    def test_xai_is_openai_backend_with_pinned_base_url(self):
        env = {"XAI_API_KEY": "x", "OPENAI_BASE_URL": "http://should-be-ignored.local/v1"}
        cfg = resolve_provider(env)
        self.assertEqual(cfg.backend, "openai")
        self.assertEqual(cfg.provider, "xai")
        self.assertEqual(cfg.base_url, XAI_BASE_URL)
        self.assertEqual(cfg.base_url, "https://api.x.ai/v1")
        self.assertEqual(cfg.model, DEFAULT_MODELS["xai"])

    def test_openai_base_url_applies_only_to_openai(self):
        cfg = resolve_provider({"OPENAI_API_KEY": "o", "OPENAI_BASE_URL": "http://localhost:11434/v1"})
        self.assertEqual(cfg.base_url, "http://localhost:11434/v1")
        self.assertIsNone(resolve_provider({"OPENAI_API_KEY": "o"}).base_url)

    def test_default_models_pinned(self):
        self.assertEqual(resolve_provider({"ANTHROPIC_API_KEY": "a"}).model, "claude-sonnet-5")
        self.assertEqual(resolve_provider({"OPENAI_API_KEY": "o"}).model, "gpt-6-astra")
        self.assertEqual(resolve_provider({"XAI_API_KEY": "x"}).model, "grok-4.6")
        self.assertEqual(resolve_provider({"GEMINI_API_KEY": "g"}).model, "gemini-3.8-flash")

    def test_gemini_default_carries_fallback_but_explicit_model_does_not(self):
        """The fallback exists to survive a retired default. An operator who set
        REMAGENT_MODEL chose deliberately and must never be swapped silently."""
        default = resolve_provider({"GEMINI_API_KEY": "g"})
        self.assertEqual(default.fallback_model, "gemini-2.5-flash")
        explicit = resolve_provider({"GEMINI_API_KEY": "g", "REMAGENT_MODEL": "gemini-3.8-pro"})
        self.assertIsNone(explicit.fallback_model)
        self.assertIsNone(resolve_provider({"ANTHROPIC_API_KEY": "a"}).fallback_model)

    def test_remagent_model_overrides_default(self):
        cfg = resolve_provider({"ANTHROPIC_API_KEY": "a", "REMAGENT_MODEL": "claude-opus-5"})
        self.assertEqual(cfg.model, "claude-opus-5")

    def test_google_api_key_alias_still_selects_gemini(self):
        cfg = resolve_provider({"GOOGLE_API_KEY": "g"})
        self.assertEqual((cfg.provider, cfg.key_env), ("gemini", "GOOGLE_API_KEY"))

    def test_no_keys_raises_naming_every_key_and_provider_var(self):
        with self.assertRaises(NoProviderKeyError) as ctx:
            resolve_provider({})
        msg = str(ctx.exception)
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "REMAGENT_PROVIDER"):
            self.assertIn(name, msg)
        self.assertNotIn("\n", msg)

    def test_explicit_provider_without_its_key_raises(self):
        with self.assertRaises(NoProviderKeyError) as ctx:
            resolve_provider({"REMAGENT_PROVIDER": "anthropic", "GEMINI_API_KEY": "g"})
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_unknown_provider_raises(self):
        with self.assertRaises(ProviderConfigError):
            resolve_provider({"REMAGENT_PROVIDER": "mistral", "OPENAI_API_KEY": "o"})

    def test_present_keys_reports_names_never_values(self):
        names = present_keys({"GEMINI_API_KEY": "top-secret", "XAI_API_KEY": "also-secret"})
        self.assertEqual(names, ["XAI_API_KEY", "GEMINI_API_KEY"])
        self.assertNotIn("secret", " ".join(names))


# ---------------------------------------------------------------------------
# Parity: same fixture, mocked SDK payloads, identical facts/rules
# ---------------------------------------------------------------------------

class TestBackendParity(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # One fixture instance shared by every backend in a test, so turn ids,
        # timestamps and fact ids are identical and the prompts must match byte-for-byte.
        self.turns = fixture_turns()
        self.profile = fixture_profile()

    async def _gemini_result(self):
        s = DreamSynthesizer(api_key="gemini-test-key")
        s._client = _GeminiClient(PAYLOAD_TEXT)
        result = await s.consolidate_window(self.turns, self.profile)
        return result, s._client.models.calls[0]

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic SDK not installed (pip install 'remagent[anthropic]')")
    async def test_anthropic_and_gemini_payloads_parse_to_same_facts_and_rules(self):
        gemini_result, gemini_call = await self._gemini_result()

        fake = anthropic_fake(PAYLOAD_TEXT)
        with mock.patch.dict(os.environ, clean_env(ANTHROPIC_API_KEY="sk-ant-test"), clear=True), \
             mock.patch("anthropic.Anthropic", return_value=fake) as ctor:
            s = DreamSynthesizer()
            anthropic_result = await s.consolidate_window(self.turns, self.profile)

        self.assertEqual(ctor.call_args.kwargs["api_key"], "sk-ant-test")
        call = fake.messages.calls[0]
        self.assertEqual(call["model"], DEFAULT_MODELS["anthropic"])
        self.assertEqual(call["system"], DREAM_PROMPT_SYSTEM)
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        # Shared prompt builder: both backends were sent the identical user prompt.
        self.assertEqual(call["messages"][0]["content"], gemini_call["contents"])
        self.assertEqual(gemini_call["config"]["system_instruction"], DREAM_PROMPT_SYSTEM)

        self.assertEqual(normalise(gemini_result), normalise(anthropic_result))
        # The supersession invariant survives both parsers identically.
        self.assertEqual(normalise(anthropic_result)["contradictions"][0][2:4], ("MySQL", "PostgreSQL"))
        self.assertEqual(normalise(anthropic_result)["facts"][0][:3], ("Project", "database", "PostgreSQL"))

    @unittest.skipUnless(HAS_OPENAI, "openai SDK not installed (pip install 'remagent[openai]')")
    async def test_openai_payload_matches_gemini(self):
        gemini_result, _ = await self._gemini_result()
        fake = openai_fake(PAYLOAD_TEXT)
        with mock.patch.dict(os.environ, clean_env(OPENAI_API_KEY="sk-openai-test"), clear=True), \
             mock.patch("openai.OpenAI", return_value=fake) as ctor:
            result = await DreamSynthesizer().consolidate_window(self.turns, self.profile)
        self.assertNotIn("base_url", ctor.call_args.kwargs)
        call = fake.chat.completions.calls[0]
        self.assertEqual(call["model"], DEFAULT_MODELS["openai"])
        self.assertEqual(call["messages"][0], {"role": "system", "content": DREAM_PROMPT_SYSTEM})
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertEqual(normalise(gemini_result), normalise(result))

    @unittest.skipUnless(HAS_OPENAI, "openai SDK not installed (pip install 'remagent[openai]')")
    async def test_xai_key_selects_openai_backend_with_xai_base_url(self):
        fake = openai_fake(PAYLOAD_TEXT)
        env = clean_env(XAI_API_KEY="xai-test-key", OPENAI_BASE_URL="http://must-not-apply.local/v1")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("openai.OpenAI", return_value=fake) as ctor:
            s = DreamSynthesizer()
            result = await s.consolidate_window(fixture_turns(), fixture_profile())
        self.assertEqual(s.provider_name, "xai")
        self.assertEqual(ctor.call_args.kwargs["base_url"], "https://api.x.ai/v1")
        self.assertEqual(ctor.call_args.kwargs["api_key"], "xai-test-key")
        self.assertEqual(fake.chat.completions.calls[0]["model"], DEFAULT_MODELS["xai"])
        self.assertEqual(len(result.added_facts), 1)

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic SDK not installed")
    async def test_unparseable_anthropic_response_raises_naming_provider(self):
        fake = anthropic_fake("definitely not json")
        with mock.patch.dict(os.environ, clean_env(ANTHROPIC_API_KEY="k"), clear=True), \
             mock.patch("anthropic.Anthropic", return_value=fake):
            with self.assertRaises(DreamSynthesisError) as ctx:
                await DreamSynthesizer().consolidate_window(fixture_turns(), fixture_profile())
        self.assertIn("anthropic", str(ctx.exception))

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic SDK not installed")
    async def test_anthropic_refusal_raises_not_fabricates(self):
        fake = anthropic_fake("", stop_reason="refusal")
        with mock.patch.dict(os.environ, clean_env(ANTHROPIC_API_KEY="k"), clear=True), \
             mock.patch("anthropic.Anthropic", return_value=fake):
            with self.assertRaises(DreamSynthesisError):
                await DreamSynthesizer().consolidate_window(fixture_turns(), fixture_profile())

    async def test_missing_sdk_extra_raises_install_hint(self):
        with mock.patch.dict(os.environ, clean_env(ANTHROPIC_API_KEY="k"), clear=True), \
             mock.patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(ProviderNotInstalledError) as ctx:
                await DreamSynthesizer().consolidate_window(fixture_turns(), fixture_profile())
        self.assertIn('pip install "remagent[anthropic]"', str(ctx.exception))


# ---------------------------------------------------------------------------
# CLI: one line, exit 1, no traceback
# ---------------------------------------------------------------------------

def run_cli(args, env, prelude=""):
    code = f"import sys; {prelude}sys.argv = {['remagent', *args]!r}; from remagent.cli import main; main()"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)


class TestDreamCliProviderErrors(unittest.TestCase):

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.db_path)  # provider errors must fire before storage is touched

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _assert_one_line_failure(self, proc):
        self.assertEqual(proc.returncode, 1, msg=proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(len(proc.stderr.strip().splitlines()), 1, msg=proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertFalse(os.path.exists(self.db_path), "must not create the DB when no provider is usable")

    def test_no_keys_exits_1_with_one_line_naming_all_keys(self):
        proc = run_cli(["dream", "--db", self.db_path], clean_env())
        self._assert_one_line_failure(proc)
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "REMAGENT_PROVIDER"):
            self.assertIn(name, proc.stderr)

    def test_provider_anthropic_without_extra_tells_user_to_install(self):
        proc = run_cli(["dream", "--db", self.db_path],
                       clean_env(REMAGENT_PROVIDER="anthropic", ANTHROPIC_API_KEY="k"),
                       prelude="sys.modules['anthropic'] = None; ")
        self._assert_one_line_failure(proc)
        self.assertIn('pip install "remagent[anthropic]"', proc.stderr)

    def test_provider_openai_without_extra_tells_user_to_install(self):
        proc = run_cli(["dream", "--db", self.db_path],
                       clean_env(REMAGENT_PROVIDER="openai", OPENAI_API_KEY="k"),
                       prelude="sys.modules['openai'] = None; ")
        self._assert_one_line_failure(proc)
        self.assertIn('pip install "remagent[openai]"', proc.stderr)

    def test_explicit_provider_missing_key_is_one_line(self):
        proc = run_cli(["dream", "--db", self.db_path], clean_env(REMAGENT_PROVIDER="xai", GEMINI_API_KEY="g"))
        self._assert_one_line_failure(proc)
        self.assertIn("XAI_API_KEY", proc.stderr)


# ---------------------------------------------------------------------------
# Doctor: provider report never leaks a secret
# ---------------------------------------------------------------------------

class TestDoctorProviderReport(unittest.TestCase):

    def _doctor_json(self, env):
        proc = run_cli(["doctor", "--db", "/nonexistent/remagent-doctor.db", "--json"], env)
        return json.loads(proc.stdout), proc

    def test_reports_active_provider_and_key_names_only(self):
        secret_a, secret_g = "sk-ant-SECRET-1234567890", "AIza-SECRET-0987654321"
        data, proc = self._doctor_json(clean_env(ANTHROPIC_API_KEY=secret_a, GEMINI_API_KEY=secret_g))
        checks = {c["name"]: c for c in data["checks"]}
        self.assertTrue(checks["api_key"]["passed"])
        self.assertIn("ANTHROPIC_API_KEY", checks["api_key"]["detail"])
        self.assertIn("GEMINI_API_KEY", checks["api_key"]["detail"], "Gemini key must be reported even when not selected")
        self.assertTrue(checks["provider"]["passed"])
        self.assertIn("anthropic", checks["provider"]["detail"])
        self.assertIn("provider_sdk", checks)
        self.assertNotIn(secret_a, proc.stdout + proc.stderr)
        self.assertNotIn(secret_g, proc.stdout + proc.stderr)

    def test_explicit_gemini_still_lists_other_keys(self):
        data, _ = self._doctor_json(clean_env(ANTHROPIC_API_KEY="a", GEMINI_API_KEY="g", REMAGENT_PROVIDER="gemini"))
        checks = {c["name"]: c for c in data["checks"]}
        self.assertIn("gemini via GEMINI_API_KEY", checks["provider"]["detail"])
        self.assertIn("ANTHROPIC_API_KEY", checks["api_key"]["detail"])
        self.assertTrue(checks["provider_sdk"]["passed"], "google-genai is a base dependency")

    def test_no_keys_reports_no_provider(self):
        data, proc = self._doctor_json(clean_env())
        checks = {c["name"]: c for c in data["checks"]}
        self.assertFalse(checks["api_key"]["passed"])
        self.assertFalse(checks["provider"]["passed"])
        self.assertNotIn("provider_sdk", checks)
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
