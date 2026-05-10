"""
Example 5:

Difference from financial_analyst.py (Example 4):
While financial_analyst.py relies on a high-level `create_agent` wrapper to generate the agent, 
this module explicitly constructs the workflow from scratch using LangGraph's `StateGraph`. 
It manually defines the `AgentState`, the nodes (`AssistantNode` and `ToolNode`), and the 
conditional edges. This approach exposes the underlying ReAct architecture and offers 
greater flexibility and customization.

Checkpointing (Example 5 addition):
Uses SqliteSaver to persist conversation state in 'langgraph.sqlite3'. 
This replaces the manual in-memory chat_history list, giving the agent durable 
cross-session memory via LangGraph's built-in checkpointing mechanism.

usage:
python financial_analyst_workflow.py --thread-id session_1

"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from load_env import configure_environment

import argparse
import os
import sys
import yfinance as yf
from typing import Literal, TypedDict, Annotated

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt





class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_stock_price_results(ticker_symbol: str, period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = "1mo"):
    """
    Fetch historical stock price data using Yahoo Finance.
    For Indian stocks, append '.NS' to the symbol (e.g., 'RELIANCE.NS' or 'TCS.NS').
    Useful for when you need to get stock prices and trends.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period=period)
        
        if history.empty:
            return "No data found for this ticker."
            
        # Return the recent closing prices as a dictionary
        # Convert Timestamp keys to strings so it serializes cleanly to the LLM
        return {str(k.date()): v for k, v in history['Close'].tail(5).to_dict().items()}
    except Exception as e:
        return f"Error fetching historical data: {e}"

@tool
def get_duckduckgo_results(query: str, num_results: int = 5):
    """
    Fetch search results from DuckDuckGo using LangChain's DuckDuckGoSearchRun tool.
    Useful for when you need to answer questions about current events or general knowledge.
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=num_results)
        search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        results = search_tool.invoke(query)
        return results
    except Exception as e:
        print(f"Error fetching results: {e}")
        return "Failed to fetch results."

def get_tools():
    return [get_duckduckgo_results, get_stock_price_results]

def get_llm(apikey, model="gpt-4o", temperature=0):
    return ChatOpenAI(model=model, temperature=temperature, api_key=apikey)


def get_llm_with_tools(llm, tools):
    """Bind tools to the LLM"""
    
    return llm.bind_tools(tools)


class AssistantNode:
    """
    Chatbot node that processes messages and generates responses.
    """
    def __init__(self, llm_with_tools, system_prompt=None):
        self.llm_with_tools = llm_with_tools
        self.system_prompt = system_prompt

    def __call__(self, state: AgentState) -> dict:
        messages = state["messages"]
        # Always inject the system prompt at the beginning if it exists
        # We don't return it in the state update, we just pass it to the LLM
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}


def build_agent_graph(llm, tools, system_prompt=None, checkpointer=None):
    """Build the financial analyst agent graph.
    
    Args:
        llm: The language model to use.
        tools: List of tools available to the agent.
        system_prompt: Optional system prompt prepended on the first turn.
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for
                      persistent, cross-session conversation memory.
    """

    llm_with_tools = get_llm_with_tools(llm, tools)

    builder = StateGraph(AgentState)
    builder.add_node("assistant", AssistantNode(llm_with_tools, system_prompt))
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.set_entry_point("assistant")
    
    return builder.compile(checkpointer=checkpointer)


def main():
    configure_environment()
    
    # Fix for Windows console UnicodeEncodeError (e.g. for the Rupee symbol)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set")

    llm = get_llm(openai_api_key)
    tools = get_tools()

    system_prompt = """You are an expert financial analyst assistant.
You have access to financial data tools and a websearch tool.
You should always think step-by-step and use the tools effectively to answer complex financial questions.
When a user asks about a stock, you MUST use the websearch tool to find the top 5 recent news articles about that stock.
You must then analyze those news results to reason about WHY the stock is performing the way it is, and clearly explain those reasons in your final response to the user.
TRANSPARENCY RULE: You MUST explicitly cite your sources in your final response. Clearly state which data came from "Yahoo Finance" and which insights came from the "DuckDuckGo Web Search Tool" so the user knows exactly where the information originated.
RESTRICTION: You are strictly restricted to answering questions about stock prices and the reasons for their performance. If a user asks about anything else, politely decline and remind them of your restriction.
Always use the tools to get the most up-to-date information."""

    parser = argparse.ArgumentParser(description="Financial Analyst Agent")
    parser.add_argument(
        "--thread-id", 
        type=str, 
        default="financial_analyst_workflow_session_1",
        help="Thread ID for conversation checkpointing (change to start a fresh conversation)"
    )
    args = parser.parse_args()

    # SqliteSaver persists all checkpoints to 'financial_analyst_workflow.sqlite3'.
    # The context manager ensures the DB connection is cleanly closed on exit.
    with SqliteSaver.from_conn_string("financial_analyst_workflow.sqlite3") as memory:
        # Build the agent graph wired to the SQLite checkpointer
        agent = build_agent_graph(llm, tools, system_prompt, checkpointer=memory)

        console = Console()

        welcome_text = (
            f"Welcome to the **Financial Analyst Agent**!\n\n"
            f"Please note: This agent is restricted to providing stock prices and analyzing the reasons for their performance.\n"
            f"Conversation history is persisted automatically to **financial_analyst_workflow.sqlite3**.\n"
            f"Current Thread ID: **{args.thread_id}**\n"
            f"Type **'exit'** or **'quit'** to end the chat."
        )
        console.print(Panel(Markdown(welcome_text), title="[bold green]AI Assistant[/bold green]", expand=False))

        # A fixed thread_id ties all turns in this session to the same checkpoint thread.
        # Change this value (or make it a CLI arg) to start a fresh conversation.
        config = {"configurable": {"thread_id": args.thread_id}}

        while True:
            try:
                console.print()
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold red]Goodbye![/bold red]")
                    break
                
                if not user_input.strip():
                    continue

                with console.status("[bold yellow]Agent is thinking...[/bold yellow]", spinner="dots"):
                    # The checkpointer automatically loads prior state for this thread
                    # and saves the new state after each invocation.
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                    )
                
                agent_response = result["messages"][-1].content
                
                console.print()
                console.print(Panel(Markdown(agent_response), title="[bold green]Financial Analyst[/bold green]", expand=False))
                
            except KeyboardInterrupt:
                console.print("\n[bold red]Goodbye![/bold red]")
                break
            except Exception as e:
                console.print(f"\n[bold red]An error occurred: {e}[/bold red]\n")

if __name__ == "__main__":
    main()
