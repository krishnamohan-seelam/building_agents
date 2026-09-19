"""
Trip Planner Chat Bot (Based on Planner-Executor Agent)
=========================================================
Pattern: Planner LLM → Plan List → Executor LLM (with tools) → Supervisor → loop or end

Free APIs used:
  - Groq (free tier): https://console.groq.com  → set GROQ_API_KEY
  - Wikipedia (no key needed)
  - DuckDuckGo search (no key needed)
  - OpenWeatherMap (free tier): https://openweathermap.org/api → set OPENWEATHER_API_KEY
"""

import ast
import json
import operator
import os
import re
import sys
from typing import Annotated, Any, List, TypedDict

import warnings

import requests
import wikipedia
from dotenv import load_dotenv

# Suppress the GuessedAtParserWarning from BeautifulSoup used within the wikipedia library
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
from pydantic import BaseModel, Field, SecretStr
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

try:
    from .memory import (
        DEFAULT_DATABASE_PATH,
        DEFAULT_THREAD_ID,
        format_conversation_history,
        latest_human_message,
    )
except ImportError:
    from memory import (
        DEFAULT_DATABASE_PATH,
        DEFAULT_THREAD_ID,
        format_conversation_history,
        latest_human_message,
    )

load_dotenv()
console = Console()

# ──────────────────────────────────────────────
# 1. GRAPH STATE
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    goal: str                               # The original user goal
    messages: Annotated[list, add_messages] # Persisted conversation messages
    plan: List[str]                         # Ordered list of steps from the Planner
    current_step_index: int                 # Which step we're currently executing
    results: List[str]                       # Research results for the current turn
    final_answer: str                       # Set when done


# ──────────────────────────────────────────────
# 2. TOOLS
# ──────────────────────────────────────────────

_ddg = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns a brief summary of results. Useful for finding travel blogs, hotel prices, and current events."""
    try:
        return _ddg.run(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia and return a 3-sentence summary. Useful for destination history and facts."""
    try:
        return wikipedia.summary(topic, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        if not e.options:
            return f"Wikipedia lookup for '{topic}' was ambiguous and returned no options."
        options_str = ", ".join(e.options[:3])
        return f"Wikipedia lookup for '{topic}' was ambiguous. Did you mean one of these? {options_str}"
    except wikipedia.exceptions.PageError:
        return f"Wikipedia lookup failed: Page '{topic}' does not exist."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    Requires OPENWEATHER_API_KEY environment variable (free tier works).
    Falls back to a mock if key is missing. Useful for packing advice.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        console.print(f"[bold yellow][WARNING][/bold yellow] OPENWEATHER_API_KEY not found. Using mock weather data for {city}.")
        return (
            f"[Mock weather — set OPENWEATHER_API_KEY for real data] "
            f"Weather in {city}: 22°C, partly cloudy, humidity 60%."
        )
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if r.status_code != 200:
            return f"Weather API error: {data.get('message', 'unknown error')}"
        weather_list = data.get("weather", [])
        desc = weather_list[0].get("description", "no description") if weather_list else "no description"
        main_data = data.get("main", {})
        temp = main_data.get("temp", "unknown")
        feels = main_data.get("feels_like", "unknown")
        humid = main_data.get("humidity", "unknown")
        return (
            f"Weather in {city}: {desc}, {temp}°C "
            f"(feels like {feels}°C), humidity {humid}%."
        )
    except requests.exceptions.RequestException as e:
        return f"Weather API network error: {e}"
    except json.JSONDecodeError as e:
        return f"Weather API invalid response: {e}"
    except Exception as e:
        return f"Weather fetch failed: {e}"


TOOLS = [web_search, wikipedia_lookup, get_weather]
TOOL_NODE = ToolNode(TOOLS)


# ──────────────────────────────────────────────
# 3. LLM Setup
# ──────────────────────────────────────────────

def get_llm(bind_tools: bool = False) -> Any:
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2, # slightly more creative for a trip advisor
            api_key=SecretStr(groq_api_key),
        )
    elif google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.2,
            api_key=google_api_key,
        )
    elif openai_api_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=SecretStr(openai_api_key),
        )
    else:
        raise ValueError(
            "No valid API key found. Please set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in your environment or .env file."
        )

    if bind_tools:
        return llm.bind_tools(TOOLS)
    return llm


# ──────────────────────────────────────────────
# 4. PLANNER NODE
# ──────────────────────────────────────────────

class Plan(BaseModel):
    name: str = Field(description="A brief name for the travel plan.", default="")
    steps: List[str] = Field(description="Ordered list of concise, actionable steps to research the travel request.")

PLANNER_PROMPT = SystemMessage(content="""You are a world-class Trip Advisor planning agent.
Given a user's travel request, produce a list of concise, actionable research steps.
Each step should be specific enough for another agent to execute using these tools:
  - web_search(query)       → Find attractions, hotels, travel tips
  - wikipedia_lookup(topic) → Get history and cultural facts
  - get_weather(city)       → Check current weather for packing advice
Ensure you gather enough info to create an amazing itinerary or travel recommendation.

RESTRICTION: You are strictly restricted to answering questions about trip planning and travel advice. If a user asks about anything else (e.g., coding, math, general chat), you must produce a plan with a single step that tells the executor to politely decline the request and remind them of your restriction. Do not generate research steps for non-travel queries.
""")


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    goal = latest_human_message(messages) or state.get("goal", "")
    history = format_conversation_history(messages)
    console.print(f"\n[bold magenta][TRAVEL PLANNER][/bold magenta] Analyzing request: {goal}")
    llm = get_llm().with_structured_output(Plan)
    
    try:
        response = llm.invoke([
            PLANNER_PROMPT,
            HumanMessage(content=(
                f"Previous conversation:\n{history}\n\nCurrent request: {goal}"
            ))
        ])
        plan = response.steps
    except Exception as e:
        console.print(f"[bold red]Planner error:[/bold red] {e}")
        plan = []

    if not plan:
        plan = [f"Address the user travel request directly: {goal}"]

    console.print(f"[bold magenta][TRAVEL PLANNER][/bold magenta] Formulated {len(plan)} research steps:")
    for i, step in enumerate(plan, 1):
        console.print(f"  {i}. {step}")

    return {
        "goal": goal,
        "plan": plan,
        "current_step_index": 0,
        "results": [],
    }


# ──────────────────────────────────────────────
# 5. EXECUTOR NODE
# ──────────────────────────────────────────────

EXECUTOR_PROMPT = """You are a diligent Travel Research agent. Your job is to complete ONE specific step of a travel research plan.

Overall travel request: {goal}
Completed research so far:
{completed}

Current task to execute: {step}

Use the available tools to complete this task.
After using tools, provide a concise result summary of what you found (2-3 sentences max), focusing on useful travel information.
"""

def executor_node(state: AgentState) -> dict:
    idx   = state["current_step_index"]
    step  = state["plan"][idx]
    goal  = state["goal"]

    completed_summary = ""
    for i, r in enumerate(state.get("results", [])):
        completed_summary += f"  Step {i+1}: {r}\n"
    if not completed_summary:
        completed_summary = "  None yet.\n"

    console.print(f"\n[bold blue][RESEARCHER][/bold blue] Executing step {idx + 1}/{len(state['plan'])}: {step}")

    llm = get_llm(bind_tools=True)
    messages = [
        SystemMessage(content=EXECUTOR_PROMPT.format(
            goal=goal, completed=completed_summary, step=step
        )),
        HumanMessage(content=f"Execute this task: {step}")
    ]

    max_iterations = 3
    iterations = 0
    while iterations < max_iterations:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_results = TOOL_NODE.invoke({"messages": messages})
        messages.extend(tool_results["messages"])
        iterations += 1
    else:
        console.print(f"[bold yellow][Warning][/bold yellow] Step execution reached max iterations ({max_iterations}). Proceeding to results.")

    content = response.content
    if isinstance(content, list):
        # Handle cases where content is a list of blocks
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        )
    result_text = content.strip() if content else f"Completed: {step}"
    console.print(f"[bold blue][RESEARCHER][/bold blue] Found: {result_text[:120]}...")

    return {
        "results": [f"Step {idx + 1} ({step}): {result_text}"],
        "current_step_index": idx + 1,
    }


# ──────────────────────────────────────────────
# 6. SYNTHESIZER NODE
# ──────────────────────────────────────────────

SYNTHESIZER_PROMPT = """You are an expert Trip Advisor and Travel Agent.
The user submitted a travel request, and your research team gathered information.

User Request: {goal}

Research Results:
{results}

Write an engaging, well-structured travel itinerary or advice response.
Use Markdown to format headings, bullet points, and bold text.
Include:
- A warm greeting as a Trip Advisor.
- Cultural/Historical context (if gathered).
- Top recommendations (attractions, food, etc.).
- Weather advice and packing tips (if gathered).
Make it exciting and helpful!

RESTRICTION: You are strictly restricted to trip planning and travel advice. If the user's request is not related to travel, simply state that you are a Trip Advisor and can only help with trip planning.
"""


def synthesizer_node(state: AgentState) -> dict:
    console.print("\n[bold green][TRIP ADVISOR][/bold green] Crafting the final travel itinerary...")
    llm = get_llm()
    results_text = "\n".join(state["results"])
    response = llm.invoke([
        HumanMessage(content=SYNTHESIZER_PROMPT.format(
            goal=state["goal"],
            results=results_text,
        )),
        HumanMessage(content=(
            "Use this prior conversation for continuity when relevant:\n"
            f"{format_conversation_history(state.get('messages', []))}"
        ))
    ])
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        )
    answer = content.strip() if content else ""
    console.print("[bold green][TRIP ADVISOR][/bold green] Done.")
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# ──────────────────────────────────────────────
# 7. SUPERVISOR
# ──────────────────────────────────────────────

def supervisor_decision(state: AgentState) -> str:
    if state["current_step_index"] < len(state["plan"]):
        console.print(f"[bold cyan][SUPERVISOR][/bold cyan] More research needed ({state['current_step_index']}/{len(state['plan'])}). Continuing.")
        return "continue"
    console.print("[bold cyan][SUPERVISOR][/bold cyan] Research complete. Handing over to Trip Advisor.")
    return "done"


# ──────────────────────────────────────────────
# 8. BUILD THE LANGGRAPH
# ──────────────────────────────────────────────

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("planner",     planner_node)
    graph.add_node("executor",    executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "executor")

    graph.add_conditional_edges(
        "executor",
        supervisor_decision,
        {
            "continue": "executor",
            "done":     "synthesizer",
        }
    )

    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# 9. MAIN — RUNNER
# ──────────────────────────────────────────────

def run_trip_advisor(
    goal: str,
    app: Any,
    thread_id: str = DEFAULT_THREAD_ID,
) -> str:
    console.print(Panel(f"[bold]TRAVEL REQUEST:[/bold] {goal}", title="[bold yellow]Trip Advisor Agent[/bold yellow]", expand=False))

    final_state = app.invoke(
        {"messages": [HumanMessage(content=goal)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    console.print()
    console.print(Panel(Markdown(final_state["final_answer"]), title="[bold green]YOUR TRIP ADVISOR PLAN[/bold green]", expand=False))
    return final_state["final_answer"]


def main():
    with SqliteSaver.from_conn_string(str(DEFAULT_DATABASE_PATH)) as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        console.print("[bold green]Welcome to the Trip Advisor Chat Bot! (Type 'exit' or 'quit' to end)[/bold green]")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                run_trip_advisor(user_input, app)
            except (KeyboardInterrupt, EOFError):
                break

        console.print("\n[bold green]Safe travels! Goodbye![/bold green]")


if __name__ == "__main__":
    main()
    sys.exit(0)
