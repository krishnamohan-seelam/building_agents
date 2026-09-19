"""
Planner-Executor Agent with LangGraph
======================================
Pattern: Planner LLM → Plan List → Executor LLM (with tools) → Supervisor → loop or end

Free APIs used:
  - Groq (free tier): https://console.groq.com  → set GROQ_API_KEY
  - Wikipedia (no key needed)
  - DuckDuckGo search (no key needed)
  - OpenWeatherMap (free tier): https://openweathermap.org/api → set OPENWEATHER_API_KEY

Install:
  pip install langgraph langchain langchain-groq \
              langchain-community duckduckgo-search \
              wikipedia requests python-dotenv
"""

import ast
import json
import operator
import os
import re
from typing import Annotated, Any, List, TypedDict

import requests
import wikipedia
from dotenv import load_dotenv
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
# 2. TOOLS (all free, no paid keys for basic use)
# ──────────────────────────────────────────────

_ddg = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns a brief summary of results."""
    try:
        return _ddg.run(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia and return a 3-sentence summary."""
    try:
        return wikipedia.summary(topic, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        if not e.options:
            return f"Wikipedia lookup for '{topic}' was ambiguous and returned no options."
        try:
            return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
        except Exception as inner_e:
            return f"Wikipedia lookup for '{topic}' was ambiguous, and resolving '{e.options[0]}' failed: {inner_e}"
    except wikipedia.exceptions.PageError:
        return f"Wikipedia lookup failed: Page '{topic}' does not exist."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    Requires OPENWEATHER_API_KEY environment variable (free tier works).
    Falls back to a mock if key is missing.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
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


def _evaluate_ast(node):
    operators = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg
    }
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num): # Fallback for Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_evaluate_ast(node.left), _evaluate_ast(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_evaluate_ast(node.operand))
    else:
        raise TypeError(f"Unsupported expression type: {type(node)}")

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression. E.g. '22 * 7 + 14'"""
    try:
        if len(expression) > 100:
            return "Error: Expression too long."
        tree = ast.parse(expression, mode='eval')
        result = _evaluate_ast(tree.body)
        return str(result)
    except SyntaxError:
        return "Calculation error: Invalid syntax."
    except TypeError as e:
        return f"Calculation error: {e}"
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = [web_search, wikipedia_lookup, get_weather, calculator]
TOOL_NODE = ToolNode(TOOLS)


# ──────────────────────────────────────────────
# 3. LLM (Groq free tier — very fast)
# ──────────────────────────────────────────────

def get_llm(bind_tools: bool = False) -> Any:
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",    # free & fast; swap to llama-3.3-70b for more power
            temperature=0,
            api_key=SecretStr(groq_api_key),
        )
    elif google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=google_api_key,
        )
    elif openai_api_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
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
    steps: List[str] = Field(description="Ordered list of concise, actionable steps to achieve the user's goal.")

PLANNER_PROMPT = SystemMessage(content="""You are a planning agent.
Given a user goal, produce a list of concise, actionable steps.
Each step should be specific enough for another agent to execute using these tools:
  - web_search(query)       → DuckDuckGo web search
  - wikipedia_lookup(topic) → Wikipedia summary
  - get_weather(city)       → current weather
  - calculator(expression)  → arithmetic
""")


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    goal = latest_human_message(messages) or state.get("goal", "")
    history = format_conversation_history(messages)
    console.print(f"\n[bold magenta][PLANNER][/bold magenta] Creating plan for goal: {goal}")
    llm = get_llm().with_structured_output(Plan)
    
    try:
        response = llm.invoke([
            PLANNER_PROMPT,
            HumanMessage(content=(
                f"Previous conversation:\n{history}\n\nCurrent goal: {goal}"
            ))
        ])
        plan = response.steps
    except Exception as e:
        console.print(f"[bold red]Planner error:[/bold red] {e}")
        plan = []

    # Fail-safe: if the plan is empty, fall back to executing the goal directly
    if not plan:
        plan = [f"Address the user goal directly: {goal}"]

    console.print(f"[bold magenta][PLANNER][/bold magenta] Generated {len(plan)} steps:")
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

EXECUTOR_PROMPT = """You are an execution agent. Your job is to complete ONE specific step.

Overall goal: {goal}
Completed steps so far:
{completed}

Current step to execute: {step}

Use the available tools to complete this step.
After using tools, provide a concise result summary (2-3 sentences max).
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

    console.print(f"\n[bold blue][EXECUTOR][/bold blue] Running step {idx + 1}/{len(state['plan'])}: {step}")

    llm = get_llm(bind_tools=True)
    messages = [
        SystemMessage(content=EXECUTOR_PROMPT.format(
            goal=goal, completed=completed_summary, step=step
        )),
        HumanMessage(content=f"Execute this step: {step}")
    ]

    # Agentic loop with safety limit to prevent infinite tool-calling loops
    max_iterations = 3
    iterations = 0
    while iterations < max_iterations:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No more tool calls — executor is done with this step
            break

        # Run the requested tools
        tool_results = TOOL_NODE.invoke({"messages": messages})
        messages.extend(tool_results["messages"])
        iterations += 1
    else:
        console.print(f"[bold yellow][Warning][/bold yellow] Step execution reached max iterations ({max_iterations}). Proceeding to results.")

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    result_text = content.strip() if content else f"Completed: {step}"
    console.print(f"[bold blue][EXECUTOR][/bold blue] Result: {result_text[:120]}...")

    return {
        "results": [f"Step {idx + 1} ({step}): {result_text}"],
        "current_step_index": idx + 1,
    }


# ──────────────────────────────────────────────
# 6. SYNTHESIZER NODE  (runs once at the end)
# ──────────────────────────────────────────────

SYNTHESIZER_PROMPT = """You are a synthesis agent.
The user had a goal and an execution agent completed a series of steps.

Goal: {goal}

Step results:
{results}

Write a clear, well-structured final answer that directly addresses the goal.
Be concise but thorough. Use the actual data from the results.
"""


def synthesizer_node(state: AgentState) -> dict:
    console.print("\n[bold green][SYNTHESIZER][/bold green] Compiling final answer...")
    llm = get_llm()
    results_text = "\n".join(state["results"])
    response = llm.invoke([
        HumanMessage(content=SYNTHESIZER_PROMPT.format(
            goal=state["goal"],
            results=results_text
        )),
        HumanMessage(content=(
            "Use this prior conversation for continuity when relevant:\n"
            f"{format_conversation_history(state.get('messages', []))}"
        ))
    ])
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    answer = content.strip() if content else ""
    console.print("[bold green][SYNTHESIZER][/bold green] Done.")
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# ──────────────────────────────────────────────
# 7. SUPERVISOR (conditional edge logic)
# ──────────────────────────────────────────────

def supervisor_decision(state: AgentState) -> str:
    """
    Conditional edge: if there are more steps, loop back to executor.
    Otherwise, route to synthesizer.
    """
    if state["current_step_index"] < len(state["plan"]):
        console.print(f"[bold cyan][SUPERVISOR][/bold cyan] More steps remain ({state['current_step_index']}/{len(state['plan'])}). Continuing.")
        return "continue"
    console.print("[bold cyan][SUPERVISOR][/bold cyan] All steps complete. Moving to synthesis.")
    return "done"


# ──────────────────────────────────────────────
# 8. BUILD THE LANGGRAPH
# ──────────────────────────────────────────────

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner",     planner_node)
    graph.add_node("executor",    executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("planner")

    # Planner always goes to executor
    graph.add_edge("planner", "executor")

    # Executor goes to supervisor decision
    graph.add_conditional_edges(
        "executor",
        supervisor_decision,
        {
            "continue": "executor",      # loop back for next step
            "done":     "synthesizer",   # all steps done → synthesize
        }
    )

    # Synthesizer ends the graph
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# 9. MAIN — run some example goals
# ──────────────────────────────────────────────

def run_agent(
    goal: str,
    app: Any = None,
    thread_id: str = DEFAULT_THREAD_ID,
) -> str:
    if app is None:
        app = build_graph()
    console.print(Panel(f"[bold]GOAL:[/bold] {goal}", title="[bold yellow]Agent Started[/bold yellow]", expand=False))

    final_state = app.invoke(
        {"messages": [HumanMessage(content=goal)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    console.print()
    console.print(Panel(Markdown(final_state["final_answer"]), title="[bold green]FINAL ANSWER[/bold green]", expand=False))
    return final_state["final_answer"]


if __name__ == "__main__":
    with SqliteSaver.from_conn_string(str(DEFAULT_DATABASE_PATH)) as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        run_agent(
            "I'm planning a trip to Varanasi, India. "
            "Tell me about the city's history, top attractions, and current weather.",
            app=app,
        )
