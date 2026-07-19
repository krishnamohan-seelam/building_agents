"""
Example 6: Financial Analyst Workflow — With Guardrails

Deepening of financial_analyst_workflow.py (Example 5):

New additions in this module
────────────────────────────
1. Expanded AgentState
   Three fields are added alongside `messages` to give each node a
   structured slice of shared state it owns and updates explicitly,
   rather than forcing the LLM to reconstruct context from history:

   • current_ticker  (str)  – set by AssistantNode when it detects a
                              ticker in the LLM's tool-call arguments.
   • sources_used    (list) – set by CustomToolNode to record which
                              data sources were called (yfinance /
                              duckduckgo) on each turn.
   • violation_count (int)  – set by guardrail_node; incremented each
                              time the user sends an off-topic message.

2. Guardrail node  (guardrail_node)
   A dedicated LangGraph node inserted *before* the AssistantNode.
   It runs a zero-cost keyword/regex heuristic on the latest user
   message.  If the message is off-topic it appends a polite rejection
   AIMessage, increments violation_count, and routes straight to END —
   the LLM is never invoked and no API tokens are consumed.

3. Structured tool errors
   Both tools now return a dict  {"status": "error", "message": …,
   "code": …}  on failure instead of a plain error string.  This gives
   the LLM a consistent, machine-readable signal and prevents it from
   retrying identically or hallucinating parameters.

4. CustomToolNode
   A thin subclass of ToolNode that inspects each tool call and
   appends the tool name to state["sources_used"] before the parent
   execute logic runs.

Graph topology (new)
──────────────────────────────────────────────────
  START
    │
    ▼
  guardrail ──(off-topic)──► END
    │ (financial)
    ▼
  assistant ──(tool calls)──► tools ──► assistant
    │ (no tool calls)
    ▼
   END

Usage:
    python financial_analyst_workflow_with_guardrail.py --thread-id session_g1
"""

import logging
import re

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from load_env import configure_environment

import argparse
import os
import sys
import yfinance as yf
from typing import Literal, TypedDict, Annotated, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_ticker: str          # Tracks the active stock ticker explicitly
    sources_used: list           # Records data sources used this turn
    violation_count: int         # Counts off-topic attempts over the session


# ─────────────────────────────────────────────
# Guardrail Configuration
# ─────────────────────────────────────────────

# Keywords that indicate a financially-relevant query.
# Any word in the user's message matching one of these (case-insensitive)
# is treated as in-scope.
_FINANCIAL_KEYWORDS = re.compile(
    r"\b("
    r"stock|ticker|price|market|share|equity|nse|bse|nasdaq|nyse|"
    r"financial|invest|portfolio|company|revenue|earnings|dividend|"
    r"fund|bond|crypto|currency|forex|trade|trading|analyst|"
    r"profit|loss|quarter|annual|report|valuation|pe|eps|"
    r"reliance|tcs|infosys|wipro|hdfc|icici|sbi|apple|google|"
    r"microsoft|tesla|amazon|meta|nvidia|sensex|nifty"
    r")\b",
    re.IGNORECASE,
)

_REJECTION_MSG = (
    "I'm sorry, but I'm strictly a **Financial Analyst Assistant**. "
    "I can only help with questions about **stock prices and the reasons "
    "for their performance**. Could you please ask a finance-related question?"
)


def _is_financial_query(text: str) -> bool:
    """Return True if the text contains at least one financial keyword."""
    return bool(_FINANCIAL_KEYWORDS.search(text))


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@tool
def get_stock_price_results(
    ticker_symbol: str,
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "1mo",
) -> dict:
    """
    Fetch historical stock price data using Yahoo Finance.
    For Indian stocks, append '.NS' to the symbol (e.g. 'RELIANCE.NS' or 'TCS.NS').
    Useful for when you need to get stock prices and trends.

    Returns a dict with 'status': 'success' and closing prices,
    or 'status': 'error' with a structured error payload on failure.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period=period)

        if history.empty:
            return {
                "status": "error",
                "message": f"No historical data found for ticker '{ticker_symbol}'. "
                           "Verify the symbol is correct (append '.NS' for Indian stocks).",
                "code": "NO_DATA",
            }

        closing_prices = {
            str(k.date()): round(v, 2)
            for k, v in history["Close"].tail(5).to_dict().items()
        }
        return {"status": "success", "ticker": ticker_symbol, "data": closing_prices}

    except Exception as e:
        logger.error("Error fetching stock data for %s: %s", ticker_symbol, e)
        return {
            "status": "error",
            "message": str(e),
            "code": "FETCH_FAILED",
        }


@tool
def get_duckduckgo_results(query: str, num_results: int = 5) -> dict:
    """
    Fetch search results from DuckDuckGo.
    Useful for when you need to answer questions about current events
    or find recent news about a stock or company.

    Returns a dict with 'status': 'success' and search text,
    or 'status': 'error' with a structured error payload on failure.
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=num_results)
        search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        results = search_tool.invoke(query)
        return {"status": "success", "source": "duckduckgo", "results": results}
    except Exception as e:
        logger.error("DuckDuckGo search error: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "code": "SEARCH_FAILED",
        }


def get_tools():
    return [get_duckduckgo_results, get_stock_price_results]


# ─────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────

def get_llm(api_key: str, model: str = "gpt-4o", temperature: int = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def get_llm_with_tools(llm, tools):
    """Bind tools to the LLM."""
    return llm.bind_tools(tools)


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def guardrail_node(state: AgentState) -> dict:
    """
    Pre-LLM guardrail node.

    Checks whether the latest user message is finance-related using a
    keyword/regex heuristic.  If not, returns a rejection message and
    increments violation_count.  The conditional edge then routes this
    state to END — the LLM is never called.

    State fields written:
        • violation_count  (incremented on every off-topic message)
        • messages         (rejection AIMessage appended on violation)
    """
    last_message = state["messages"][-1]
    text = last_message.content if hasattr(last_message, "content") else ""

    if _is_financial_query(text):
        # Pass through — no state changes needed
        return {}

    # Off-topic: build rejection and increment counter
    current_count = state.get("violation_count", 0) or 0
    return {
        "messages": [AIMessage(content=_REJECTION_MSG)],
        "violation_count": current_count + 1,
    }


def _guardrail_router(state: AgentState) -> str:
    """
    Conditional edge function after guardrail_node.
    If the last message is already an AIMessage (rejection was appended),
    route to END.  Otherwise continue to the assistant.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.content == _REJECTION_MSG:
        return END
    return "assistant"


class AssistantNode:
    """
    LLM reasoning node.

    Prepends the system prompt on every invocation, calls the bound LLM,
    and extracts the ticker symbol from any stock-price tool-call arguments
    to update current_ticker in state.

    State fields written:
        • messages          (LLM response appended)
        • current_ticker    (updated when a stock tool call is detected)
    """

    def __init__(self, llm_with_tools, system_prompt: Optional[str] = None):
        self.llm_with_tools = llm_with_tools
        self.system_prompt = system_prompt

    def __call__(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = self.llm_with_tools.invoke(messages)

        # Extract ticker from tool call arguments if present
        updates: dict = {"messages": [response]}
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc.get("name") == "get_stock_price_results":
                    ticker = tc.get("args", {}).get("ticker_symbol", "")
                    if ticker:
                        updates["current_ticker"] = ticker
                        break

        return updates


class CustomToolNode(ToolNode):
    """
    Subclass of ToolNode that records which tools were invoked into
    state["sources_used"] before delegating execution to the parent.

    State fields written:
        • sources_used  (tool names appended after each invocation)
    """

    def __call__(self, state: AgentState, **kwargs) -> dict:
        # Collect tool names from the last AI message's tool_calls
        last = state["messages"][-1]
        called_tools = []
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get("name", "")
                if name == "get_stock_price_results":
                    called_tools.append("Yahoo Finance (yfinance)")
                elif name == "get_duckduckgo_results":
                    called_tools.append("DuckDuckGo Web Search")
                else:
                    called_tools.append(name)

        # Run parent ToolNode logic
        parent_result = super().__call__(state, **kwargs)

        # Merge sources_used
        existing = state.get("sources_used") or []
        merged = list(dict.fromkeys(existing + called_tools))  # dedup, preserve order
        parent_result["sources_used"] = merged
        return parent_result


# ─────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────

def build_agent_graph(llm, tools, system_prompt=None, checkpointer=None):
    """
    Build the guarded financial analyst agent graph.

    Graph topology:
        START → guardrail → [END on violation | assistant]
                             assistant → tools_condition → [tools | END]
                                                tools → assistant

    Args:
        llm:          The language model to use.
        tools:        List of tools available to the agent.
        system_prompt: Optional system prompt prepended on each turn.
        checkpointer: Optional LangGraph checkpointer for durable memory.
    """
    llm_with_tools = get_llm_with_tools(llm, tools)

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("assistant", AssistantNode(llm_with_tools, system_prompt))
    builder.add_node("tools", CustomToolNode(tools))

    # Edges
    builder.set_entry_point("guardrail")
    builder.add_conditional_edges("guardrail", _guardrail_router)
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    configure_environment()

    # Fix for Windows console UnicodeEncodeError (e.g. for the Rupee symbol)
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    llm = get_llm(openai_api_key)
    tools = get_tools()

    system_prompt = (
        "You are an expert financial analyst assistant. "
        "You have access to financial data tools and a web-search tool. "
        "Think step-by-step and use the tools effectively to answer complex financial questions.\n"
        "When a user asks about a stock, you MUST use the websearch tool to find the top 5 "
        "recent news articles about that stock. Then analyse those results to reason about WHY "
        "the stock is performing the way it is, and clearly explain those reasons.\n"
        "TRANSPARENCY RULE: Explicitly cite your sources in every response. State which data "
        "came from 'Yahoo Finance' and which from 'DuckDuckGo Web Search'.\n"
        "RESTRICTION: You are strictly restricted to stock prices and the reasons for their "
        "performance. If asked anything else, politely decline.\n"
        "Always use tools to get the most up-to-date information."
    )

    parser = argparse.ArgumentParser(description="Financial Analyst Agent (with Guardrails)")
    parser.add_argument(
        "--thread-id",
        type=str,
        default="fa_guardrail_session_1",
        help="Thread ID for conversation checkpointing",
    )
    args = parser.parse_args()

    db_path = "financial_analyst_workflow_guardrail.sqlite3"

    with SqliteSaver.from_conn_string(db_path) as memory:
        agent = build_agent_graph(llm, tools, system_prompt, checkpointer=memory)

        console = Console()

        welcome_text = (
            f"Welcome to the **Financial Analyst Agent** _(with Guardrails)_!\n\n"
            f"This agent is restricted to stock prices and performance analysis.\n"
            f"Off-topic queries are rejected **before** reaching the LLM — no tokens wasted.\n"
            f"State is persisted to **{db_path}**.\n"
            f"Current Thread ID: **{args.thread_id}**\n"
            f"Type **'exit'** or **'quit'** to end the chat."
        )
        console.print(
            Panel(
                Markdown(welcome_text),
                title="[bold green]AI Assistant[/bold green]",
                expand=False,
            )
        )

        config = {"configurable": {"thread_id": args.thread_id}}

        while True:
            try:
                console.print()
                user_input = Prompt.ask("[bold blue]You[/bold blue]")

                if user_input.lower() in ["quit", "exit"]:
                    console.print("[bold red]Goodbye![/bold red]")
                    break

                if not user_input.strip():
                    continue

                with console.status("[bold yellow]Agent is thinking...[/bold yellow]", spinner="dots"):
                    result = agent.invoke(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "current_ticker": "",
                            "sources_used": [],
                            "violation_count": 0,
                        },
                        config=config,
                    )

                agent_response = result["messages"][-1].content
                ticker = result.get("current_ticker", "")
                sources = result.get("sources_used", [])
                violations = result.get("violation_count", 0)

                console.print()
                console.print(
                    Panel(
                        Markdown(agent_response),
                        title="[bold green]Financial Analyst[/bold green]",
                        expand=False,
                    )
                )

                # Debug sidebar
                if ticker or sources:
                    meta_parts = []
                    if ticker:
                        meta_parts.append(f"Ticker: [cyan]{ticker}[/cyan]")
                    if sources:
                        meta_parts.append(f"Sources: [yellow]{', '.join(sources)}[/yellow]")
                    console.print("  " + "  |  ".join(meta_parts), style="dim")

                if violations:
                    console.print(
                        f"  [dim red]⚠ Off-topic attempts this session: {violations}[/dim red]"
                    )

            except KeyboardInterrupt:
                console.print("\n[bold red]Goodbye![/bold red]")
                break
            except Exception as e:
                console.print(f"\n[bold red]An error occurred: {e}[/bold red]\n")


if __name__ == "__main__":
    main()
