"""
Example 4:

This module demonstrates how to build a specialized ReAct agent using LangGraph and LangChain. 
It creates a "Financial Analyst" persona that can autonomously fetch historical stock prices 
using Yahoo Finance (yfinance) and search for recent market news using DuckDuckGo.
The agent uses these tools to reason about stock performance and present insights to the user.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from load_env import configure_environment

import os
import sys
import yfinance as yf
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

def get_llm(apikey, model="gpt-4o", temperature=0):
    return ChatOpenAI(model=model, temperature=temperature, api_key=apikey)

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

    # Execute the specific agent logic
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    
    console = Console()

    welcome_text = (
        "Welcome to the **Financial Analyst Agent**!\n\n"
        "Please note: This agent is restricted to providing stock prices and analyzing the reasons for their performance.\n"
        "Type **'exit'** or **'quit'** to end the chat."
    )
    console.print(Panel(Markdown(welcome_text), title="[bold green]AI Assistant[/bold green]", expand=False))

    chat_history = []

    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit']:
                console.print("[bold red]Goodbye![/bold red]")
                break
            
            if not user_input.strip():
                continue

            chat_history.append(("user", user_input))
            
            with console.status("[bold yellow]Agent is thinking...[/bold yellow]", spinner="dots"):
                result = agent.invoke({"messages": chat_history})
            
            agent_response = result["messages"][-1].content
            
            console.print()
            console.print(Panel(Markdown(agent_response), title="[bold green]Financial Analyst[/bold green]", expand=False))
            
            chat_history.append(("assistant", agent_response))
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Goodbye![/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occurred: {e}[/bold red]\n")

if __name__ == "__main__":
    main()
