import json
from unittest.mock import MagicMock, patch
from query_decomposer.utils.text import _coerce_text
from query_decomposer.utils.tools import web_search, wikipedia_lookup
from query_decomposer.agent import (
    router,
    decomposer_node,
    research_node,
    synthesizer_node,
    log_event,
    logger
)


def test_coerce_text_handles_list_content() -> None:
    assert _coerce_text([{"type": "text", "text": "Hello"}, " world"]) == "Hello world"


def test_router_falls_back_to_synthesize_for_unrecognized_output() -> None:
    state = {"final_answer": "unexpected output"}
    assert router(state) == "synthesize"


@patch("query_decomposer.utils.tools.wikipedia.summary")
def test_wikipedia_lookup(mock_summary) -> None:
    mock_summary.return_value = "Python is a programming language."
    result = wikipedia_lookup.invoke({"topic": "Python"})
    assert result == "Python is a programming language."
    mock_summary.assert_called_once_with("Python", sentences=5, auto_suggest=True)


@patch("query_decomposer.utils.tools.DuckDuckGoSearchRun.run")
def test_web_search(mock_run) -> None:
    mock_run.return_value = "Search result details"
    result = web_search.invoke({"query": "current news"})
    assert result == "Search result details"
    mock_run.assert_called_once_with("current news")


def test_log_event_formats_json_correctly() -> None:
    with patch.object(logger, "info") as mock_info:
        log_event("test_event", {"key": "value"})
        mock_info.assert_called_once()
        log_str = mock_info.call_args[0][0]
        log_json = json.loads(log_str)
        assert log_json["event_type"] == "test_event"
        assert log_json["data"] == {"key": "value"}
        assert "timestamp" in log_json


@patch("query_decomposer.agent.get_llm")
def test_decomposer_node_logging(mock_get_llm) -> None:
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Follow-up [web_search]: What is the weather?"
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    state = {
        "question": "What is the weather in Paris and Rome?",
        "scratchpad": [],
        "final_answer": "",
        "iterations": 0
    }

    with patch.object(logger, "info") as mock_info:
        res = decomposer_node(state)
        assert res["iterations"] == 1
        assert "Follow-up" in res["final_answer"]

        log_types = []
        for call in mock_info.call_args_list:
            log_entry = json.loads(call[0][0])
            log_types.append(log_entry["event_type"])
        
        assert "node_start" in log_types
        assert "llm_start" in log_types
        assert "llm_end" in log_types
        assert "node_end" in log_types


@patch("query_decomposer.agent.TOOLS")
def test_research_node_logging(mock_tools) -> None:
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Sunny, 25C"
    mock_tools.__getitem__.return_value = mock_tool

    state = {
        "question": "What is the weather?",
        "scratchpad": [],
        "final_answer": "Follow-up [web_search]: What is the weather?",
        "iterations": 1
    }

    with patch.object(logger, "info") as mock_info:
        res = research_node(state)
        assert "scratchpad" in res
        assert "Q: What is the weather?\nA: Sunny, 25C" in res["scratchpad"]

        log_types = []
        for call in mock_info.call_args_list:
            log_entry = json.loads(call[0][0])
            log_types.append(log_entry["event_type"])

        assert "node_start" in log_types
        assert "tool_start" in log_types
        assert "tool_end" in log_types
        assert "node_end" in log_types


@patch("query_decomposer.agent.get_llm")
def test_synthesizer_node_logging_inline_answer(mock_get_llm) -> None:
    state = {
        "question": "What is the capital of France?",
        "scratchpad": [],
        "final_answer": "Final Answer: Paris",
        "iterations": 1
    }

    with patch.object(logger, "info") as mock_info:
        res = synthesizer_node(state)
        assert res["final_answer"] == "Paris"

        log_types = []
        for call in mock_info.call_args_list:
            log_entry = json.loads(call[0][0])
            log_types.append(log_entry["event_type"])

        assert "node_start" in log_types
        assert "node_end" in log_types
        assert "llm_start" not in log_types

