"""
Bounded-recall regression tests: recall must keep working as the brain grows.

This exists because of a real outage. At 74 active rules recall fit the budget.
At 744 it refused outright — the rule tier was all-or-nothing, so every session
on the machine silently lost its memory context the moment the brain got big.

The fix selects within the budget instead of refusing, and states what it left
out. These tests use a profile an order of magnitude past the size that broke
it, so the failure cannot come back quietly.
"""

import logging
import unittest

from remagent.governor import GovernorBudgetError, TokenBudgetGovernor
from remagent.schemas import Fact, MemoryProfile, OperationalRule

RULE_COUNT = 800
FACT_COUNT = 2000
BUDGETS = (8000, 4000, 2000, 1000, 500)


def big_profile() -> MemoryProfile:
    """A brain larger than the one that broke, spread across priorities."""
    rules = [
        OperationalRule(
            id=f"rule-{i}",
            category=["operational_directive", "coding_standard",
                      "architecture_heuristic", "user_preference"][i % 4],
            rule=f"Directive {i}: " + ("detail " * 12),
            rationale="synthesized",
            priority=(i % 5) + 1,
        )
        for i in range(RULE_COUNT)
    ]
    facts = [
        Fact(
            entity=f"Service{i % 50}",
            attribute=f"attribute_{i % 17}",
            value=f"value-{i}",
            confidence=0.5 + (i % 50) / 100.0,
        )
        for i in range(FACT_COUNT)
    ]
    return MemoryProfile(agent_id="big_brain", facts=facts, rules=rules)


class TestBoundedRecall(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.profile = big_profile()
        cls.governor = TokenBudgetGovernor()

    def test_recall_fits_every_budget(self):
        """The hard cap is the whole point: no budget may be exceeded, and no
        budget may produce an empty injection."""
        for budget in BUDGETS:
            with self.subTest(budget=budget):
                text = self.governor.build_budgeted_prompt_injection(
                    self.profile, max_tokens=budget
                )
                self.assertTrue(text.strip(), "recall must never be silently empty")
                self.assertLessEqual(self.governor.estimate_tokens(text), budget)

    def test_recall_does_not_refuse_merely_because_the_brain_is_large(self):
        """The exact regression: 800 rules against 8000 tokens used to raise."""
        text = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=8000)
        self.assertIn("REMAGENT DETERMINISTIC MEMORY CONTEXT", text)

    def test_both_rules_and_facts_survive(self):
        """A huge rule set must not crowd facts out entirely — the rules exist
        to operate on the entity graph, so an injection of pure rules is half
        a memory."""
        text = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=8000)
        self.assertIn("OPERATIONAL RULES & POLICIES:", text)
        self.assertIn("ACTIVE ENTITY KNOWLEDGE GRAPH:", text)
        self.assertGreater(text.count("\n- ["), 0)
        self.assertGreater(text.count(" (conf: "), 0)

    def test_highest_priority_rules_are_kept_first(self):
        """When most rules must go, the ones that survive should be the ones
        marked unbreakable."""
        text = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=2000)
        included = [ln for ln in text.splitlines() if ln.startswith("- [")]
        self.assertTrue(included)
        self.assertTrue(all("(P1)" in ln for ln in included),
                        f"expected only P1 rules at a tight budget, got: {included[:3]}")

    def test_omissions_are_stated_not_hidden(self):
        text = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=2000)
        self.assertIn("rule(s) omitted by token budget", text)
        self.assertIn(f"of {RULE_COUNT} shown", text)
        self.assertIn("fact(s) omitted by token budget", text)

    def test_trimming_is_logged(self):
        """Operators need to know recall is lossy without reading the output."""
        with self.assertLogs("remagent.governor", level=logging.WARNING) as captured:
            self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=2000)
        self.assertTrue(any("recall trimmed to fit" in m for m in captured.output))

    def test_selection_is_deterministic(self):
        """Same brain, same budget, same injection — otherwise sessions would
        drift for reasons nobody can reproduce."""
        a = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=3000)
        b = self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=3000)
        self.assertEqual(a, b)

    def test_query_context_steers_fact_selection(self):
        """With 2000 facts and room for a few dozen, relevance to the request
        is what makes recall useful rather than arbitrary."""
        text = self.governor.build_budgeted_prompt_injection(
            self.profile, query_context="Service7 attribute_3", max_tokens=2000
        )
        self.assertIn("Service7.", text)

    def test_larger_budget_never_returns_less(self):
        sizes = [
            self.governor.estimate_tokens(
                self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=b)
            )
            for b in (1000, 2000, 4000, 8000)
        ]
        self.assertEqual(sizes, sorted(sizes), f"recall shrank as budget grew: {sizes}")

    def test_absurd_budget_fails_loudly(self):
        """Refusing is still correct for a misconfigured budget — it just must
        not be triggered by brain size."""
        with self.assertRaises(GovernorBudgetError):
            self.governor.build_budgeted_prompt_injection(self.profile, max_tokens=3)


if __name__ == "__main__":
    unittest.main()
