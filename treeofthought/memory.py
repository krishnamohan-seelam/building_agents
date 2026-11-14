"""Shared durable conversation-memory helpers for the Tree of Thought agent."""

from pathlib import Path
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage

DEFAULT_THREAD_ID = "local_tot_session"
DEFAULT_DATABASE_PATH = Path(__file__).with_name("tot_memory.sqlite3")
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS_PER_MESSAGE = 500
TRUNCATION_MARKER = "... [truncated]"


def format_conversation_history(messages: Sequence[BaseMessage]) -> str:
    """Format recent user and assistant messages for an LLM prompt.

    Each message's content is capped at MAX_HISTORY_CHARS_PER_MESSAGE so a single
    long turn (e.g. a prior synthesized answer) can't blow up the prompt sent to
    every node on later turns in the same thread.
    """
    recent_messages = list(messages)[-MAX_HISTORY_MESSAGES:]
    if not recent_messages:
        return "None yet."

    lines = []
    for message in recent_messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = str(message.content).strip()
        if content:
            if len(content) > MAX_HISTORY_CHARS_PER_MESSAGE:
                content = content[:MAX_HISTORY_CHARS_PER_MESSAGE] + TRUNCATION_MARKER
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "None yet."
