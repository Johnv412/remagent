"""Secret hiding for logged turns (pre-launch, 1.1.2).

Every turn is scanned for API keys, tokens, passwords, private-key blocks and payment-card numbers BEFORE it is
saved, and again when it is rendered into a dream prompt (so rows written by older versions are covered too).
Matches are replaced with a category tag such as [REDACTED:api_key]; the original value is never stored, logged
or sent to a model. The detector is adapted from Dream Engine's sensitive.py and reports category names only.
"""

import re
from typing import Iterable

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("api_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b")),           # OpenAI / Anthropic style
    ("api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")),                           # Google API key
    ("api_key", re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}\b")),                           # Google express key
    ("api_key", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),                       # GitHub tokens
    ("api_key", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),                     # Slack tokens
    ("api_key", re.compile(r"\bsb_(?:secret|publishable)_[A-Za-z0-9_-]{10,}\b")),     # Supabase keys
    ("api_key", re.compile(r"\b(?:flr|flw)_[A-Za-z0-9_-]{20,}\b")),                   # floor keys
    ("token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),  # JWT
    ("token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("api_key", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|client[_-]?secret)"
                           r"(\s*[:=]\s*|\s+)['\"]?[A-Za-z0-9._~+/=-]{8,}['\"]?")),
    ("password", re.compile(r"(?i)\b(?:password|passwd|pwd)(\s*[:=]\s*|\s+is\s+)['\"]?\S+['\"]?")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----|"
                               r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
_DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn(candidate: str) -> bool:
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _keep_label(match: re.Match, category: str) -> str:
    """For 'api_key = VALUE' keep the label so the sentence still reads; drop only the value."""
    text = match.group(0)
    if category in ("api_key", "password", "token") and match.lastindex:
        sep_end = match.end(1) - match.start(0)
        return text[:sep_end] + f"[REDACTED:{category}]"
    return f"[REDACTED:{category}]"


def redact(content: str) -> tuple[str, list[str]]:
    """(clean content, sorted category names found). Never returns the matched value."""
    if not content:
        return content, []
    found: set[str] = set()
    clean = content
    for category, pattern in _PATTERNS:
        def _sub(m: re.Match, _c=category) -> str:
            found.add(_c)
            return _keep_label(m, _c)
        clean = pattern.sub(_sub, clean)

    def _card(m: re.Match) -> str:
        if _luhn(m.group(0)):
            found.add("payment_card")
            return "[REDACTED:payment_card]"
        return m.group(0)
    clean = _DIGIT_RUN.sub(_card, clean)
    return clean, sorted(found)


def detect_sensitive_categories(content: str) -> list[str]:
    """Category names only, for callers that just want to know."""
    return redact(content)[1]


def redact_all(items: Iterable[str]) -> list[str]:
    return [redact(s)[0] for s in items]
