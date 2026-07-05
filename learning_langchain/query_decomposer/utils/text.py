from typing import Any, List
import re

WIKI_KEYWORDS = re.compile(
    r"\b(who|biography|history|invented|founded|born|died|country|capital|"
    r"population|language|culture|war|empire|dynasty|theory|definition|"
    r"what is|what are|explain|overview)\b",
    re.IGNORECASE
)


def _coerce_text(content: Any) -> str:
    """Convert varied LLM message content payloads into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return str(content)
    return str(content)

def format_scratchpad(scratchpad: List[str]) -> str:
    if not scratchpad:
        return "  (empty — no sub-questions answered yet)"
    return "\n".join(f"  {entry}" for entry in scratchpad)