"""Unit and integration tests for Smart Query Router module."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from smart_query_router.state import RouterState, IntentClassification
from smart_query_router.prompts import (
    build_intent_detection_messages,
    build_direct_response_messages,
    build_web_search_response_messages,
)
from smart_query_router.nodes import (
    format_search_results,
    intent_detection_node,
    web_search_node,
    direct_response_node,
    _get_default_llm,
    _get_default_search_tool,
)
from smart_query_router.graph import route_intent, build_router_graph
from smart_query_router.cli import run_cli, EXIT_KEYWORDS


# ============================================================================
# 1. State & Schema Tests
# ============================================================================

def test_router_state_schema():
    """Verify RouterState TypedDict structure."""
    state: RouterState = {
        "query": "What is quantum computing?",
        "intent": "direct",
        "reasoning": "General scientific topic",
        "search_results": None,
        "response": "Quantum computing is...",
    }
    assert state["query"] == "What is quantum computing?"
    assert state["intent"] == "direct"
    assert state["response"] == "Quantum computing is..."


def test_intent_classification_pydantic_model():
    """Verify IntentClassification Pydantic model validation."""
    model_search = IntentClassification(
        intent="search", reasoning="Requires live stock market price"
    )
    assert model_search.intent == "search"
    assert "stock market" in model_search.reasoning

    model_direct = IntentClassification(
        intent="direct", reasoning="Basic calculus rule"
    )
    assert model_direct.intent == "direct"

    with pytest.raises(ValueError):
        IntentClassification(intent="invalid_intent", reasoning="Invalid")


# ============================================================================
# 2. Prompt Management Tests (prompt_helper)
# ============================================================================

def test_prompt_builder_integration():
    """Verify prompt generation via prompt_helper PromptBuilder."""
    intent_msgs = build_intent_detection_messages("What is the weather in Tokyo today?")
    assert len(intent_msgs) == 2
    assert "Tokyo today" in intent_msgs[1].content

    direct_msgs = build_direct_response_messages("Explain Python list comprehension.")
    assert len(direct_msgs) == 2
    assert "list comprehension" in direct_msgs[1].content

    search_msgs = build_web_search_response_messages(
        "Who won the match?", "Match results: Team A won 2-1"
    )
    assert len(search_msgs) == 2
    assert "Team A won 2-1" in search_msgs[1].content


# ============================================================================
# 3. Helper & Internal Function Tests
# ============================================================================

def test_format_search_results():
    """Verify search result formatting utility."""
    # List of dicts
    results_list = [
        {"title": "News 1", "content": "Headline 1", "url": "https://example.com/1"},
        {"title": "News 2", "content": "Headline 2", "url": "https://example.com/2"},
    ]
    formatted = format_search_results(results_list)
    assert "[1] News 1" in formatted
    assert "https://example.com/1" in formatted

    # List of non-dicts
    assert "[1] Simple String" in format_search_results(["Simple String"])

    # Plain string
    assert format_search_results("Raw text results") == "Raw text results"

    # Empty list
    assert format_search_results([]) == "No relevant search results found."


def test_get_default_llm_and_search_tool():
    """Verify default LLM and search tool instantiation with and without API keys."""
    with patch.dict("os.environ", {}, clear=True):
        llm = _get_default_llm()
        assert llm is not None

        tool = _get_default_search_tool()
        assert tool is not None


# ============================================================================
# 4. Node Unit Tests
# ============================================================================

def test_intent_detection_node_search():
    """Verify intent_detection_node returns search intent when LLM classifies as search."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_structured_llm.invoke.return_value = IntentClassification(
        intent="search", reasoning="Requires live updates"
    )

    state: RouterState = {"query": "Latest stock price of Apple"}
    result = intent_detection_node(state, llm=mock_llm)

    assert result["intent"] == "search"
    assert result["reasoning"] == "Requires live updates"


def test_intent_detection_node_default_llm():
    """Verify intent_detection_node uses default LLM when llm parameter is None."""
    with patch("smart_query_router.nodes._get_default_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = IntentClassification(
            intent="direct", reasoning="General fact"
        )
        mock_get_llm.return_value = mock_llm

        state: RouterState = {"query": "What is 10 + 10?"}
        result = intent_detection_node(state, llm=None)
        assert result["intent"] == "direct"


def test_web_search_node():
    """Verify web_search_node invokes search tool and LLM."""
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = [
        {"title": "Search Result", "content": "Latest news content", "url": "http://test.org"}
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Augmented answer based on news.")

    state: RouterState = {"query": "What is happening today?"}
    result = web_search_node(state, llm=mock_llm, search_tool=mock_tool)

    assert result["search_results"] == mock_tool.invoke.return_value
    assert result["response"] == "Augmented answer based on news."
    mock_tool.invoke.assert_called_once_with({"query": "What is happening today?"})


def test_web_search_node_default_instantiation():
    """Verify web_search_node uses defaults when llm and search_tool are None."""
    with patch("smart_query_router.nodes._get_default_llm") as mock_get_llm, \
         patch("smart_query_router.nodes._get_default_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Search context"
        mock_get_tool.return_value = mock_tool

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Search result response")
        mock_get_llm.return_value = mock_llm

        state: RouterState = {"query": "Weather today"}
        result = web_search_node(state, llm=None, search_tool=None)
        assert result["response"] == "Search result response"


def test_web_search_node_exception_handling():
    """Verify web_search_node handles search tool exception gracefully."""
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = RuntimeError("API key invalid")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Fallback answer.")

    state: RouterState = {"query": "Failed search query"}
    result = web_search_node(state, llm=mock_llm, search_tool=mock_tool)

    assert "Search execution failed" in str(result["search_results"])
    assert result["response"] == "Fallback answer."


def test_direct_response_node():
    """Verify direct_response_node generates answer using LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Direct answer from model knowledge.")

    state: RouterState = {"query": "What is 2 + 2?"}
    result = direct_response_node(state, llm=mock_llm)

    assert result["response"] == "Direct answer from model knowledge."


def test_direct_response_node_default_llm():
    """Verify direct_response_node uses default LLM when llm is None."""
    with patch("smart_query_router.nodes._get_default_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Direct response default")
        mock_get_llm.return_value = mock_llm

        state: RouterState = {"query": "Tell me a joke"}
        result = direct_response_node(state, llm=None)
        assert result["response"] == "Direct response default"


# ============================================================================
# 5. Routing Logic Tests
# ============================================================================

def test_route_intent():
    """Verify route_intent decision logic."""
    assert route_intent({"intent": "search"}) == "web_search_node"
    assert route_intent({"intent": "direct"}) == "direct_response_node"
    assert route_intent({}) == "direct_response_node"


# ============================================================================
# 6. End-to-End Graph Execution Tests
# ============================================================================

def test_graph_execution_search_path():
    """Verify graph execution routes to web_search for search intent."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    mock_structured_llm.invoke.return_value = IntentClassification(
        intent="search", reasoning="Requires web data"
    )
    mock_llm.invoke.return_value = AIMessage(content="Web augmented answer.")

    mock_search_tool = MagicMock()
    mock_search_tool.invoke.return_value = [{"title": "Info", "content": "Details"}]

    graph = build_router_graph(llm=mock_llm, search_tool=mock_search_tool)
    final_state = graph.invoke({"query": "What is the current price of Gold?"})

    assert final_state["intent"] == "search"
    assert final_state["response"] == "Web augmented answer."
    assert final_state["search_results"] == [{"title": "Info", "content": "Details"}]


def test_graph_execution_direct_path():
    """Verify graph execution routes to direct_response for direct intent."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    mock_structured_llm.invoke.return_value = IntentClassification(
        intent="direct", reasoning="Internal knowledge topic"
    )
    mock_llm.invoke.return_value = AIMessage(content="Direct answer.")

    mock_search_tool = MagicMock()

    graph = build_router_graph(llm=mock_llm, search_tool=mock_search_tool)
    final_state = graph.invoke({"query": "What is Photosynthesis?"})

    assert final_state["intent"] == "direct"
    assert final_state["response"] == "Direct answer."
    mock_search_tool.invoke.assert_not_called()


def test_graph_execution_with_runtime_config():
    """Verify graph execution with runtime config={"configurable": {...}} dictionary."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_structured_llm.invoke.return_value = IntentClassification(
        intent="search", reasoning="Configurable test reasoning"
    )
    mock_llm.invoke.return_value = AIMessage(content="Configured response.")

    mock_search_tool = MagicMock()
    mock_search_tool.invoke.return_value = [{"title": "Config Item", "content": "Data"}]

    # Unconfigured graph base
    graph = build_router_graph()
    final_state = graph.invoke(
        {"query": "Test configurable query"},
        config={"configurable": {"llm": mock_llm, "search_tool": mock_search_tool}},
    )

    assert final_state["intent"] == "search"
    assert final_state["reasoning"] == "Configurable test reasoning"
    assert final_state["response"] == "Configured response."
    assert final_state["search_results"] == [{"title": "Config Item", "content": "Data"}]


def test_nodes_with_runnable_config():
    """Verify individual top-level node functions handle RunnableConfig directly."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Direct node via RunnableConfig.")

    state: RouterState = {"query": "Test query"}
    config = {"configurable": {"llm": mock_llm}}

    result = direct_response_node(state, config=config)
    assert result["response"] == "Direct node via RunnableConfig."


# ============================================================================
# 7. CLI REPL & Rich Rendering Tests

# ============================================================================

from smart_query_router.cli import render_search_results_table


def test_render_search_results_table():
    """Verify render_search_results_table constructs a Rich Table."""
    results_list = [
        {"title": "Test Title 1", "url": "https://example.com/1", "content": "Sample content 1"},
        {"title": "Test Title 2", "url": "https://example.com/2", "snippet": "Sample snippet 2"},
    ]
    table = render_search_results_table(results_list)
    assert table is not None
    assert table.title == "[bold green]Tavily Web Search Context[/bold green]"

    # Test string input
    table_str = render_search_results_table("Raw text result")
    assert table_str is not None

    # Test other data type
    table_obj = render_search_results_table({"raw": "dict"})
    assert table_obj is not None


def test_cli_exit_keywords():
    """Verify CLI exits cleanly on termination keywords using rich Console input."""
    for keyword in ["exit", "quit", "bye", "stop"]:
        with patch("rich.console.Console.input", return_value=keyword), \
             patch("smart_query_router.cli.build_router_graph"):
            run_cli()


def test_cli_interactive_query_processing():
    """Verify CLI handles interactive queries, search results, and errors properly."""
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {
            "intent": "search",
            "reasoning": "Live query",
            "search_results": [{"title": "News", "url": "http://news.com", "content": "Text"}],
            "response": "Search response",
        },
        {
            "intent": "direct",
            "reasoning": "Simple math",
            "response": "2 + 2 = 4",
        },
        RuntimeError("Test execution error"),
    ]

    inputs = ["Weather today", "What is 2 + 2?", "Trigger Error", "", "exit"]
    with patch("rich.console.Console.input", side_effect=inputs), \
         patch("smart_query_router.cli.build_router_graph", return_value=mock_graph):
        run_cli()

    assert mock_graph.invoke.call_count == 3

