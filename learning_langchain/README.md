# Learning LangChain

This directory contains introductory examples and exercises for building AI agents using LangChain and LangGraph.

## Modules

* [Example 1: Hello Agent (`hello_agent.py`)](./hello_agent.py) - A very basic LangChain agent setup using a simple prompt and LLM combination.
* [Example 2: Agent with Memory (`agent_with_memory.py`)](./agent_with_memory.py) - Demonstrates how to add conversational memory to an agent so it remembers previous turns.
* [Example 3: Chatbot with Tools (`chatbot.py`)](./chatbot.py) - A "smart" agent that uses the Tavily search API to retrieve up-to-date information from the web.
* [Example 4: Financial Analyst (`financial_analyst.py`)](./financial_analyst.py) - A specialized ReAct agent using LangGraph that fetches stock data via Yahoo Finance (`yfinance`) and reasons about market news using DuckDuckGo search.

## Utilities

* [`load_env.py`](./load_env.py) - A utility script to load environment variables from the `.env` file securely.
