"""Redacts secrets from trace text before it is written to disk (M2-T1)."""

import re

_PATTERNS = {
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{20,})\b"
    ),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "account": re.compile(r"\b\d{4}(?:[ -]?\d{4}){2,3}\b"),
}


def redact_text(text: str) -> tuple[str, int]:
    """Replaces API keys, emails, and account/card numbers with [REDACTED:<kind>] markers."""
    count = 0

    def _replace(kind, pattern, s):
        nonlocal count

        def _sub(_match):
            nonlocal count
            count += 1
            return f"[REDACTED:{kind}]"

        return pattern.sub(_sub, s)

    for kind, pattern in _PATTERNS.items():
        text = _replace(kind, pattern, text)

    return text, count
