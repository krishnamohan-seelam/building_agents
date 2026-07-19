import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from learning_langchain.financial_analyst_workflow import (
    get_stock_price_results,
    get_duckduckgo_results,
    get_tools,
    get_llm,
    get_llm_with_tools,
    AssistantNode,
    build_agent_graph,
    AgentState
)

class TestFinancialAnalystWorkflow:

    @patch("learning_langchain.financial_analyst_workflow.yf.Ticker")
    def test_get_stock_price_results_success(self, mock_ticker):
        # Setup mock ticker and history DataFrame
        mock_stock = MagicMock()
        # Create a mock DataFrame with a 'Close' column
        dates = pd.date_range(start="2026-07-01", periods=5)
        history_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
        mock_stock.history.return_value = history_df
        mock_ticker.return_value = mock_stock

        result = get_stock_price_results.invoke({"ticker_symbol": "AAPL"})
        
        assert isinstance(result, dict)
        assert len(result) == 5
        assert "2026-07-05" in result
        assert result["2026-07-05"] == 104.0

    @patch("learning_langchain.financial_analyst_workflow.yf.Ticker")
    def test_get_stock_price_results_empty(self, mock_ticker):
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock

        result = get_stock_price_results.invoke({"ticker_symbol": "INVALID"})
        assert result == "No data found for this ticker."

    @patch("learning_langchain.financial_analyst_workflow.yf.Ticker")
    def test_get_stock_price_results_error(self, mock_ticker):
        mock_ticker.side_effect = Exception("API connection error")
        result = get_stock_price_results.invoke({"ticker_symbol": "AAPL"})
        assert "Error fetching historical data" in result

    @patch("learning_langchain.financial_analyst_workflow.DuckDuckGoSearchRun")
    @patch("learning_langchain.financial_analyst_workflow.DuckDuckGoSearchAPIWrapper")
    def test_get_duckduckgo_results_success(self, mock_wrapper_class, mock_search_class):
        mock_search_inst = MagicMock()
        mock_search_inst.invoke.return_value = "Search Results Content"
        mock_search_class.return_value = mock_search_inst

        result = get_duckduckgo_results.invoke({"query": "AAPL news"})
        assert result == "Search Results Content"
        mock_search_inst.invoke.assert_called_once_with("AAPL news")

    @patch("learning_langchain.financial_analyst_workflow.DuckDuckGoSearchRun")
    @patch("learning_langchain.financial_analyst_workflow.DuckDuckGoSearchAPIWrapper")
    def test_get_duckduckgo_results_error(self, mock_wrapper_class, mock_search_class):
        mock_search_inst = MagicMock()
        mock_search_inst.invoke.side_effect = Exception("DDG rate limit")
        mock_search_class.return_value = mock_search_inst

        result = get_duckduckgo_results.invoke({"query": "AAPL news"})
        assert result == "Failed to fetch results."

    def test_get_tools(self):
        tools = get_tools()
        assert len(tools) == 2
        assert tools[0].name == "get_duckduckgo_results"
        assert tools[1].name == "get_stock_price_results"

    @patch("learning_langchain.financial_analyst_workflow.ChatOpenAI")
    def test_get_llm(self, mock_chat_openai):
        get_llm("test-api-key")
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0, api_key="test-api-key")

    def test_get_llm_with_tools(self):
        mock_llm = MagicMock()
        tools = ["tool1", "tool2"]
        get_llm_with_tools(mock_llm, tools)
        mock_llm.bind_tools.assert_called_once_with(tools)

    def test_assistant_node_call(self):
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Response message")
        mock_llm.invoke.return_value = mock_response

        node = AssistantNode(llm_with_tools=mock_llm, system_prompt="You are a helper")
        state: AgentState = {"messages": [HumanMessage(content="Hello")]}
        
        result = node(state)
        
        assert result == {"messages": [mock_response]}
        # Verify the system prompt was prepended in the call
        args, _ = mock_llm.invoke.call_args
        called_messages = args[0]
        assert len(called_messages) == 2
        assert isinstance(called_messages[0], SystemMessage)
        assert called_messages[0].content == "You are a helper"
        assert isinstance(called_messages[1], HumanMessage)
        assert called_messages[1].content == "Hello"

    @patch("learning_langchain.financial_analyst_workflow.StateGraph")
    def test_build_agent_graph(self, mock_state_graph):
        mock_builder = MagicMock()
        mock_state_graph.return_value = mock_builder
        mock_llm = MagicMock()
        mock_tools = []
        
        build_agent_graph(mock_llm, mock_tools, system_prompt="Test Prompt")
        
        mock_builder.add_node.assert_any_call("assistant", ANY)
        mock_builder.add_node.assert_any_call("tools", ANY)
        mock_builder.add_conditional_edges.assert_called_once()
        mock_builder.add_edge.assert_called_once_with("tools", "assistant")
        mock_builder.set_entry_point.assert_called_once_with("assistant")
        mock_builder.compile.assert_called_once()
