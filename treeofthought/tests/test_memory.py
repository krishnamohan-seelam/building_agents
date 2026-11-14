from langchain_core.messages import AIMessage, HumanMessage

from treeofthought.memory import (
    MAX_HISTORY_CHARS_PER_MESSAGE,
    MAX_HISTORY_MESSAGES,
    format_conversation_history,
)


def test_empty_history_returns_placeholder():
    assert format_conversation_history([]) == "None yet."


def test_short_messages_are_unchanged():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = format_conversation_history(messages)
    assert result == "User: hi\nAssistant: hello"


def test_long_message_is_truncated_per_message():
    long_content = "x" * (MAX_HISTORY_CHARS_PER_MESSAGE * 3)
    messages = [HumanMessage(content=long_content)]
    result = format_conversation_history(messages)

    assert result.startswith("User: ")
    body = result[len("User: "):]
    assert len(body) <= MAX_HISTORY_CHARS_PER_MESSAGE + len("... [truncated]")
    assert body.endswith("... [truncated]")


def test_history_caps_total_size_even_with_many_long_messages():
    long_content = "y" * (MAX_HISTORY_CHARS_PER_MESSAGE * 5)
    messages = [HumanMessage(content=long_content) for _ in range(50)]
    result = format_conversation_history(messages)

    per_message_cap = MAX_HISTORY_CHARS_PER_MESSAGE + len("... [truncated]") + len("User: ")
    max_total = MAX_HISTORY_MESSAGES * per_message_cap + (MAX_HISTORY_MESSAGES - 1)  # newline joins
    assert len(result) <= max_total
