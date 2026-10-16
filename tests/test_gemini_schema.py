"""
Regression tests for the Gemini response-schema contract.

google-genai rejects any `default` in a response schema:

    ValueError: Default value is not supported in the response schema
                for the Gemini API.

Passing the Pydantic model straight through as `response_schema` sends those
defaults along, which broke every dream from 2026-09-27 onward. The dreams
failed loudly and wrote nothing, so no memory was corrupted — but 257 turns
queued up behind the failure.

These tests run entirely offline: `google.genai._transformers.t_schema` is the
same validation the SDK applies before any network call, so a fake client with
`vertexai = False` exercises the real constraint without an API key.
"""

import sys
import unittest

from google.genai import _transformers as genai_transformers

from remagent.engine.prompting import (
    DreamSynthesisOutput,
    dream_output_json_schema,
    gemini_response_schema,
)


class _FakeGeminiClient:
    """t_schema only consults `vertexai`; Gemini API rules apply when False."""
    vertexai = False


def _defaults_in(node, path=""):
    """Every path in a schema that carries a `default` key."""
    found = []
    if isinstance(node, dict):
        if "default" in node:
            found.append(path or "<root>")
        for key, value in node.items():
            found += _defaults_in(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _defaults_in(value, f"{path}[{i}]")
    return found


class TestGeminiSchemaContract(unittest.TestCase):

    def test_raw_pydantic_model_behaviour_is_sdk_dependent(self):
        """Documents why the sanitiser exists without pinning the suite to one
        SDK generation. google-genai 1.x rejects any `default` in a response
        schema; 2.4+ accepts it. Either way, the sanitised schema we actually
        send is covered by the tests below — this one only records which
        behaviour the installed SDK has."""
        try:
            genai_transformers.t_schema(_FakeGeminiClient(), DreamSynthesisOutput)
        except ValueError as exc:
            self.assertIn("Default value is not supported", str(exc))
            return
        import google.genai
        self.skipTest(
            f"google-genai {google.genai.__version__} accepts defaults; "
            "the sanitiser is forward/backward-compat hardening here"
        )

    def test_four_known_fields_carry_the_offending_defaults(self):
        """Pins the exact fields, so a new defaulted field cannot slip in
        unnoticed and silently break dreaming again."""
        offenders = set(_defaults_in(DreamSynthesisOutput.model_json_schema()))
        self.assertEqual(
            offenders,
            {
                "/$defs/SynthesizedFact/properties/confidence",
                "/$defs/SynthesizedRule/properties/rationale",
                "/$defs/SynthesizedRule/properties/priority",
                "/properties/pruned_noise_count",
            },
        )

    def test_gemini_response_schema_is_accepted(self):
        """The whole point: what we actually send must survive t_schema."""
        schema = gemini_response_schema()
        result = genai_transformers.t_schema(_FakeGeminiClient(), schema)
        self.assertIsNotNone(result)

    def test_gemini_response_schema_has_no_defaults_anywhere(self):
        self.assertEqual(_defaults_in(gemini_response_schema()), [])

    def test_gemini_schema_preserves_every_output_field(self):
        """Stripping defaults must not quietly drop structure — a schema that
        validates but omits added_facts would produce empty dreams."""
        props = gemini_response_schema()["properties"]
        self.assertEqual(
            set(props),
            {"added_facts", "updated_rules", "contradictions",
             "pruned_noise_count", "pruned_noise_categories", "reasoning_summary"},
        )
        fact = props["added_facts"]["items"]["properties"]
        self.assertEqual(set(fact), {"entity", "attribute", "value", "confidence"})
        rule = props["updated_rules"]["items"]["properties"]
        self.assertEqual(set(rule), {"category", "rule", "rationale", "priority"})
        contra = props["contradictions"]["items"]["properties"]
        self.assertEqual(
            set(contra),
            {"entity", "attribute", "prior_value", "new_value", "resolution_reasoning"},
        )

    def test_gemini_schema_is_self_contained(self):
        """Gemini gets an inlined schema: no $ref/$defs to resolve, and no
        additionalProperties, which it does not model."""
        import json
        blob = json.dumps(gemini_response_schema())
        self.assertNotIn("$ref", blob)
        self.assertNotIn("$defs", blob)
        self.assertNotIn("additionalProperties", blob)

    def test_reasoning_summary_stays_required(self):
        """A dream with no reasoning_summary is indistinguishable from a
        fabricated one, so the field must stay mandatory."""
        self.assertIn("reasoning_summary", gemini_response_schema()["required"])

    def test_strict_schema_for_other_providers_is_not_weakened(self):
        """Anthropic and OpenAI use strict structured outputs, which need
        additionalProperties:false and full required lists. Fixing Gemini must
        not relax them."""
        strict = dream_output_json_schema()
        self.assertEqual(_defaults_in(strict), [])
        self.assertIs(strict["additionalProperties"], False)
        self.assertIn("reasoning_summary", strict["required"])


if __name__ == "__main__":
    unittest.main()
