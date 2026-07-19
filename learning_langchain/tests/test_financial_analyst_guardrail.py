"""
Tests for financial_analyst_workflow_with_guardrail.py

Covers:
    1. test_guardrail_blocks_offtopic         — off-topic query is rejected pre-LLM
    2. test_guardrail_passes_financial        — financial query passes through
    3. test_tool_structured_error_stock       — bad ticker returns structured error dict
    4. test_tool_structured_error_search      — DuckDuckGo failure returns structured dict
    5. test_graph_compiles                    — graph builds without error using FakeListChatModel
    6. test_state_violation_count_increments  — two off-topic messages increment counter to 2
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Make the learning_langchain package importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.fake import FakeListChatModel

from financial_analyst_workflow_with_guardrail import (
    AgentState,
    guardrail_node,
    _guardrail_router,
    _is_financial_query,
    _REJECTION_MSG,
    get_stock_price_results,
    get_duckduckgo_results,
    build_agent_graph,
    CustomToolNode,
    AssistantNode,
    END,
)


# ─────────────────────────────────────────────
# Helper — build minimal AgentState
# ─────────────────────────────────────────────

def make_state(user_text: str, violation_count: int = 0) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_text)],
        "current_ticker": "",
        "sources_used": [],
        "violation_count": violation_count,
    }


# ─────────────────────────────────────────────
# 1. Guardrail blocks off-topic queries
# ─────────────────────────────────────────────

def test_guardrail_blocks_offtopic():
    state = make_state("Tell me a joke about penguins")
    result = guardrail_node(state)

    assert result.get("violation_count") == 1, "violation_count should be incremented to 1"
    assert len(result.get("messages", [])) == 1, "A rejection message should be appended"
    rejection = result["messages"][0]
    assert isinstance(rejection, AIMessage), "Rejection must be an AIMessage"
    assert _REJECTION_MSG in rejection.content


def test_guardrail_routes_offtopic_to_end():
    """After guardrail fires, the router should return END."""
    state = make_state("What is the capital of France?")
    result = guardrail_node(state)
    # Merge result into state to simulate LangGraph state merging
    merged_state = {**state, **result, "messages": state["messages"] + result.get("messages", [])}
    route = _guardrail_router(merged_state)
    assert route == END


# ─────────────────────────────────────────────
# 2. Guardrail passes financial queries
# ─────────────────────────────────────────────

def test_guardrail_passes_financial():
    state = make_state("What is the stock price of Reliance?")
    result = guardrail_node(state)

    # No violation, no rejection message added
    assert result.get("violation_count", 0) == 0 or result == {}
    assert len(result.get("messages", [])) == 0


def test_guardrail_passes_ticker_query():
    state = make_state("Show me TCS.NS earnings data")
    result = guardrail_node(state)
    assert result == {} or result.get("violation_count", 0) == 0


def test_is_financial_query_positive():
    assert _is_financial_query("What is the NIFTY index today?") is True
    assert _is_financial_query("Tell me about HDFC stock dividend") is True
    assert _is_financial_query("Apple earnings per share last quarter") is True


def test_is_financial_query_negative():
    assert _is_financial_query("How do I bake a chocolate cake?") is False
    assert _is_financial_query("What is the weather in Mumbai?") is False
    assert _is_financial_query("Tell me about history of Rome") is False


# ─────────────────────────────────────────────
# 3. Structured errors — stock tool
# ─────────────────────────────────────────────

def test_tool_structured_error_stock_invalid_ticker():
    """An invalid ticker should return a structured error dict, not raise."""
    result = get_stock_price_results.invoke({"ticker_symbol": "INVALID###TICKER"})
    assert isinstance(result, dict), "Tool must always return a dict"
    assert result.get("status") in ("error", "success"), "status must be 'error' or 'success'"


def test_tool_structured_error_stock_empty():
    """A ticker that returns no data should give status=error with code=NO_DATA."""
    # Patch yfinance to return empty DataFrame
    import pandas as pd
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("financial_analyst_workflow_with_guardrail.yf.Ticker", return_value=mock_ticker):
        result = get_stock_price_results.invoke({"ticker_symbol": "FAKE.NS"})

    assert result["status"] == "error"
    assert result["code"] == "NO_DATA"
    assert "FAKE.NS" in result["message"]


def test_tool_structured_error_stock_exception():
    """A yfinance exception should be caught and returned as status=error."""
    with patch("financial_analyst_workflow_with_guardrail.yf.Ticker", side_effect=RuntimeError("network error")):
        result = get_stock_price_results.invoke({"ticker_symbol": "TCS.NS"})

    assert result["status"] == "error"
    assert result["code"] == "FETCH_FAILED"
    assert "network error" in result["message"]


# ─────────────────────────────────────────────
# 4. Structured errors — search tool
# ─────────────────────────────────────────────

def test_tool_structured_error_search():
    """A DuckDuckGo exception should return status=error, not raise."""
    with patch(
        "financial_analyst_workflow_with_guardrail.DuckDuckGoSearchRun.invoke",
        side_effect=RuntimeError("rate limited"),
    ):
        result = get_duckduckgo_results.invoke({"query": "Reliance stock news"})

    assert result["status"] == "error"
    assert result["code"] == "SEARCH_FAILED"
    assert "rate limited" in result["message"]


def test_tool_structured_success_search():
    """A successful search should return status=success and a results key."""
    mock_wrapper = MagicMock(spec=["max_results"])
    mock_search = MagicMock()
    mock_search.invoke.return_value = "Reliance hits all time high..."

    with patch(
        "financial_analyst_workflow_with_guardrail.DuckDuckGoSearchAPIWrapper",
        return_value=mock_wrapper,
    ):
        with patch(
            "financial_analyst_workflow_with_guardrail.DuckDuckGoSearchRun",
            return_value=mock_search,
        ):
            result = get_duckduckgo_results.invoke({"query": "Reliance stock news"})

    assert result["status"] == "success"
    assert "results" in result


# ─────────────────────────────────────────────
# 5. Graph compilation
# ─────────────────────────────────────────────

def test_graph_compiles():
    """The graph should compile without errors using a MagicMock LLM."""
    # FakeListChatModel doesn't implement bind_tools; use a MagicMock instead.
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    tools = []
    graph = build_agent_graph(mock_llm, tools, system_prompt="Test prompt")
    assert graph is not None, "Graph should compile successfully"


def test_graph_has_guardrail_node():
    """The compiled graph should include a 'guardrail' node."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    graph = build_agent_graph(mock_llm, [])
    node_names = list(graph.get_graph().nodes.keys())
    assert "guardrail" in node_names, f"Expected 'guardrail' node, found: {node_names}"


# ─────────────────────────────────────────────
# 6. violation_count increments across calls
# ─────────────────────────────────────────────

def test_state_violation_count_increments():
    """Two sequential off-topic calls should sum violation_count to 2."""
    state = make_state("Who wrote Romeo and Juliet?", violation_count=0)
    result1 = guardrail_node(state)
    count1 = result1.get("violation_count", 0)
    assert count1 == 1, f"Expected 1 after first violation, got {count1}"

    # Simulate second off-topic message with persisted count
    state2 = make_state("What is the best pizza recipe?", violation_count=count1)
    result2 = guardrail_node(state2)
    count2 = result2.get("violation_count", 0)
    assert count2 == 2, f"Expected 2 after second violation, got {count2}"


def test_violation_count_not_incremented_on_financial():
    """A valid financial query must NOT increment violation_count."""
    state = make_state("What is the PE ratio of TCS?", violation_count=0)
    result = guardrail_node(state)
    assert result.get("violation_count", 0) == 0, "No violation should be recorded for financial queries"
