"""
TokenBudgetGovernor: Precision token budgeting & priority compression for RemAgent.
Ensures zero-vector context injections strictly respect LLM token boundaries (e.g. 200 - 1000 tokens)
by prioritizing P1 operational rules, high-confidence entity facts, and active relationships.
"""

import logging
from typing import Any, Dict, List, Optional
from remagent.schemas import Fact, OperationalRule, MemoryProfile


logger = logging.getLogger("remagent.governor")


class GovernorBudgetError(RuntimeError):
    """
    Raised only when the budget is too small to hold even a minimal injection
    — a misconfigured budget, not a large brain. A big rule set is handled by
    bounded selection with an explicit omission note, never by refusing.
    """


class TokenBudgetGovernor:
    """
    Manages prompt token allocation for consolidated memory injections.

    Recall has to keep working as the brain grows. An all-or-nothing rule tier
    does the opposite: at 74 rules it fit, at 744 it refused outright and every
    session lost its memory context. So rules are now selected rather than
    guaranteed — highest priority and most relevant first — and anything left
    out is stated in the output instead of being dropped silently.
    """

    def __init__(
        self,
        default_max_tokens: int = 6000,
        char_to_token_ratio: float = 3.8,
        fact_reserve_share: float = 0.3,
    ):
        self.default_max_tokens = default_max_tokens
        self.char_to_token_ratio = char_to_token_ratio
        # Facts keep a slice of the budget so a huge rule set cannot crowd out
        # the entity knowledge the rules are supposed to operate on.
        self.fact_reserve_share = min(max(fact_reserve_share, 0.0), 0.9)

    def estimate_tokens(self, text: str) -> int:
        """Heuristic character-to-token estimator."""
        return max(1, int(len(text) / self.char_to_token_ratio))

    # -- ranking ------------------------------------------------------------

    @staticmethod
    def _relevance(text: str, query_lower: str) -> int:
        return sum(1 for token in set(text.lower().split()) if token in query_lower)

    def _rank_rules(
        self, rules: List[OperationalRule], query_context: Optional[str]
    ) -> List[OperationalRule]:
        """Priority first, then query relevance, then shorter rules — shorter
        means more of the operating doctrine survives the same budget."""
        q = (query_context or "").lower()
        return sorted(
            rules,
            key=lambda r: (
                r.priority,
                -(self._relevance(r.rule, q) if q else 0),
                len(r.rule),
            ),
        )

    def _rank_facts(self, facts: List[Fact], query_context: Optional[str]) -> List[Fact]:
        if not query_context:
            return sorted(facts, key=lambda f: f.confidence, reverse=True)
        q = query_context.lower()
        scored = []
        for f in facts:
            score = 0
            if f.entity.lower() in q:
                score += 4
            if f.attribute.lower() in q:
                score += 3
            if str(f.value).lower() in q:
                score += 2
            scored.append((score, f.confidence, f))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored]

    # -- selection ----------------------------------------------------------

    def build_budgeted_prompt_injection(
        self,
        profile: MemoryProfile,
        query_context: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Compile the highest-signal memory context that fits within max_tokens.

        Selection order: P1 rules, then query-relevant facts, then remaining
        rules by priority, then remaining facts by confidence. Whatever does
        not fit is reported in the output and logged.
        """
        budget = max_tokens or self.default_max_tokens

        ranked_rules = self._rank_rules([r for r in profile.rules if r.is_active], query_context)
        ranked_facts = self._rank_facts([f for f in profile.facts if f.is_active], query_context)
        total_rules, total_facts = len(ranked_rules), len(ranked_facts)

        # Cost each line once instead of re-rendering the whole injection per
        # candidate: at 744 rules the quadratic version is both slow and the
        # reason this path was never exercised at scale.
        overhead = self.estimate_tokens(self._format_injection([], [], None, None))
        rule_costs = [self.estimate_tokens(self._rule_line(r)) for r in ranked_rules]
        fact_costs = [self.estimate_tokens(self._fact_line(f)) for f in ranked_facts]

        if budget <= overhead:
            raise GovernorBudgetError(
                f"token budget {budget} cannot hold even an empty injection "
                f"(~{overhead} tokens of structure). Raise max_tokens."
            )

        # Greedy selection costs only the content; the omission notes are
        # accounted for exactly by the render-and-shrink pass below. Reserving
        # for them up front starves small budgets of the very rules they are
        # meant to describe.
        spend = overhead
        fact_reserve = int((budget - overhead) * self.fact_reserve_share) if ranked_facts else 0

        n_rules = 0
        for cost in rule_costs:
            if spend + cost > budget - fact_reserve:
                break
            spend += cost
            n_rules += 1

        n_facts = 0
        for cost in fact_costs:
            if spend + cost > budget:
                break
            spend += cost
            n_facts += 1

        # Facts may not have used their reserve; give the remainder back to rules.
        for cost in rule_costs[n_rules:]:
            if spend + cost > budget:
                break
            spend += cost
            n_rules += 1

        selected_rules = ranked_rules[:n_rules]
        selected_facts = ranked_facts[:n_facts]

        # Estimation is a heuristic, so verify the rendered result and shrink
        # if it overshoots. Facts go first; rules are the higher tier.
        while True:
            text = self._format_injection(
                selected_rules,
                selected_facts,
                self._rules_omitted_note(total_rules, len(selected_rules)),
                self._facts_omitted_note(total_facts - len(selected_facts)),
            )
            if self.estimate_tokens(text) <= budget:
                break
            if selected_facts:
                selected_facts.pop()
            elif selected_rules:
                selected_rules.pop()
            else:
                # Nothing left to drop and still over budget: the frame alone
                # does not fit. Fall through to the loud failure below rather
                # than return something that busts the caller's budget.
                break

        if not selected_rules and not selected_facts and (total_rules or total_facts):
            raise GovernorBudgetError(
                f"token budget {budget} is too small to include a single rule or fact "
                f"({total_rules} rules, {total_facts} facts available). Raise max_tokens."
            )

        omitted_rules = total_rules - len(selected_rules)
        omitted_facts = total_facts - len(selected_facts)
        if omitted_rules or omitted_facts:
            logger.warning(
                "recall trimmed to fit %d-token budget: %d/%d rules, %d/%d facts included "
                "(omitted %d rules, %d facts)",
                budget, len(selected_rules), total_rules,
                len(selected_facts), total_facts, omitted_rules, omitted_facts,
            )

        return text

    @staticmethod
    def _rule_line(rule: OperationalRule) -> str:
        return f"- [{rule.category.upper()}] (P{rule.priority}): {rule.rule}"

    @staticmethod
    def _fact_line(fact: Fact) -> str:
        return f"- {fact.entity}.{fact.attribute} = {str(fact.value)} (conf: {fact.confidence:.2f})"

    @staticmethod
    def _rules_omitted_note(total: int, included: int) -> Optional[str]:
        omitted = total - included
        if omitted <= 0:
            return None
        return (
            f"(+{omitted} lower-priority rule(s) omitted by token budget; "
            f"{included} of {total} shown, highest priority first)"
        )

    @staticmethod
    def _facts_omitted_note(omitted_count: int) -> Optional[str]:
        if omitted_count <= 0:
            return None
        return f"(+{omitted_count} lower-priority fact(s) omitted by token budget)"

    def _format_injection(
        self,
        rules: List[OperationalRule],
        facts: List[Fact],
        rules_note: Optional[str] = None,
        facts_note: Optional[str] = None,
    ) -> str:
        sections = ["\n[REMAGENT DETERMINISTIC MEMORY CONTEXT]"]
        if rules:
            sections.append("OPERATIONAL RULES & POLICIES:")
            for r in rules:
                sections.append(self._rule_line(r))
        if rules_note:
            sections.append(rules_note)

        if facts:
            sections.append("\nACTIVE ENTITY KNOWLEDGE GRAPH:")
            for f in facts:
                sections.append(self._fact_line(f))
        if facts_note:
            sections.append(f"\n{facts_note}")

        sections.append("[END MEMORY CONTEXT]\n")
        return "\n".join(sections)
