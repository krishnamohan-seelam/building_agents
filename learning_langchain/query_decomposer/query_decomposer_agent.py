"""
Query Decomposer Agent with LangGraph
=======================================
Pattern: Self-Ask with Search
  Decomposer LLM → "Follow-up: <sub-question>" → Research Tool
                 → "Follow-up: <sub-question>" → Research Tool
                 → ...
                 → "Final Answer: <answer>"     → Synthesizer → END

The agent iteratively breaks a complex question into sub-questions,
calls search tools to answer each, accumulates a scratchpad, and
synthesizes a final answer once enough information is gathered.

Free APIs used:
  - Groq (free tier): https://console.groq.com  → set GROQ_API_KEY
  - Wikipedia (no key needed)
  - DuckDuckGo search (no key needed)
  - Tavily (free tier): https://tavily.com      → set TAVILY_API_KEY (optional)

Install:
  pip install langgraph langchain langchain-groq \\
              langchain-community langchain-tavily \\
              duckduckgo-search wikipedia requests python-dotenv rich
"""

import ast
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone



# Ensure UTF-8 output on Windows, except when running tests
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import operator
import re
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from pydantic import SecretStr
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from .utils.tools import web_search, wikipedia_lookup, calculator, pick_tool
from .utils.text import _coerce_text, format_scratchpad, WIKI_KEYWORDS
from .prompts import DECOMPOSER_SYSTEM, DECOMPOSER_HUMAN_TEMPLATE, SYNTHESIZER_PROMPT
load_dotenv()
console = Console()

# ──────────────────────────────────────────────
# STRUCTURED LOGGING CONFIGURATION
# ──────────────────────────────────────────────
logger = logging.getLogger("query_decomposer")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if imported multiple times
if not logger.handlers:
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "query_decomposer.log")
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

def log_event(event_type: str, data: dict) -> None:
    """Log a structured JSON event to the log file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "event_type": event_type,
        "data": data
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))

MAX_ITERATIONS = 5   # Safety limit to prevent runaway loops

# Matches both:
#   "Follow-up [tool_name]: <question>"   (preferred)
#   "Follow-up tool_name: <question>"     (LLM sometimes drops brackets)
#   "Follow-up: <question>"               (no tool hint)
_KNOWN_TOOLS = "wikipedia_lookup|web_search|calculator"
_FOLLOWUP_RE = re.compile(
    r"Follow-up\s*"
    r"(?:\[(?P<tool_bracket>[^\]]+)\]|(?P<tool_bare>" + _KNOWN_TOOLS + r"))?"
    r"\s*:\s*(?P<question>.+)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


# ──────────────────────────────────────────────
# 1. GRAPH STATE
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    question:     str                                    # Original user question
    scratchpad:   Annotated[List[str], operator.add]     # Accumulated Q→A pairs
    final_answer: str                                    # Set once the LLM signals done
    iterations:   int                                    # Safety counter


# ──────────────────────────────────────────────
# 2. TOOLS
# ──────────────────────────────────────────────
# Add tools from utils 



TOOLS = {t.name: t for t in [web_search, wikipedia_lookup, calculator]}


def get_llm():
    """Return the best available LLM: Groq → Google → OpenAI."""
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=SecretStr(groq_key),
        )
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=SecretStr(google_key),
        )
    if openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=SecretStr(openai_key),
        )
    else:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY, GOOGLE_API_KEY, or "
            "OPENAI_API_KEY in your .env file."
        )


# ──────────────────────────────────────────────
# 4. DECOMPOSER NODE
# ──────────────────────────────────────────────



def decomposer_node(state: AgentState) -> dict:
    """
    Calls the LLM to decide the next action:
      - "Follow-up [tool]: <question>" → research more
      - "Final Answer: <text>"          → done
    """
    iters = state.get("iterations", 0)

    log_event("node_start", {
        "node_name": "decomposer",
        "state": {
            "question": state.get("question"),
            "iterations": iters,
            "scratchpad": state.get("scratchpad", [])
        }
    })

    if iters >= MAX_ITERATIONS:
        console.print(
            f"[bold yellow][DECOMPOSER][/bold yellow] "
            f"Safety limit ({MAX_ITERATIONS}) reached. Forcing final answer."
        )
        # Force synthesize from whatever we have
        output = {"iterations": iters + 1, "final_answer": "__synthesize__"}
        log_event("node_end", {
            "node_name": "decomposer",
            "output": output
        })
        return output

    console.print(
        f"\n[bold magenta][DECOMPOSER][/bold magenta] "
        f"Iteration {iters + 1} — deciding next step..."
    )

    llm = get_llm()
    scratchpad_text = format_scratchpad(state.get("scratchpad", []))

    prompts = [
        {"role": "system", "content": DECOMPOSER_SYSTEM},
        {"role": "user", "content": DECOMPOSER_HUMAN_TEMPLATE.format(
            question=state["question"],
            scratchpad=scratchpad_text,
        )}
    ]
    log_event("llm_start", {
        "node_name": "decomposer",
        "prompts": prompts,
        "model": getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    })

    llm_start_time = time.perf_counter()
    try:
        response = llm.invoke([
            SystemMessage(content=DECOMPOSER_SYSTEM),
            HumanMessage(content=DECOMPOSER_HUMAN_TEMPLATE.format(
                question=state["question"],
                scratchpad=scratchpad_text,
            ))
        ])
        llm_latency = time.perf_counter() - llm_start_time
        raw = _coerce_text(response.content).strip()

        token_usage = getattr(response, "usage_metadata", None)
        if not token_usage:
            token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        log_event("llm_end", {
            "node_name": "decomposer",
            "response": raw,
            "latency_seconds": llm_latency,
            "token_usage": token_usage
        })
    except Exception as e:
        llm_latency = time.perf_counter() - llm_start_time
        log_event("llm_end", {
            "node_name": "decomposer",
            "error": str(e),
            "latency_seconds": llm_latency,
            "status": "error"
        })
        raise

    console.print(f"[bold magenta][DECOMPOSER][/bold magenta] → {raw[:200]}")

    output = {"iterations": iters + 1, "final_answer": raw}
    log_event("node_end", {
        "node_name": "decomposer",
        "output": output
    })
    return output


# ──────────────────────────────────────────────
# 5. ROUTER (conditional edge)
# ──────────────────────────────────────────────

def router(state: AgentState) -> str:
    """Route to 'research' or 'synthesize' based on decomposer output."""
    decision = state.get("final_answer", "")

    if decision == "__synthesize__":
        return "synthesize"
    if _FINAL_RE.search(decision):
        return "synthesize"
    if _FOLLOWUP_RE.search(decision):
        return "research"

    # Fallback: if no recognizable pattern, synthesize with what we have
    console.print("[bold yellow][ROUTER][/bold yellow] Unrecognized pattern — forcing synthesize.")
    return "synthesize"


# ──────────────────────────────────────────────
# 6. RESEARCH NODE
# ──────────────────────────────────────────────

# Tool routing heuristics — prefer Wikipedia for encyclopedic topics

def research_node(state: AgentState) -> dict:
    """
    Parse the decomposer's Follow-up, pick a tool, execute it,
    and append the Q→A pair to the scratchpad.
    """
    log_event("node_start", {
        "node_name": "research",
        "state": {
            "scratchpad": state.get("scratchpad", []),
            "final_answer": state.get("final_answer", "")
        }
    })

    decision = state.get("final_answer", "")
    match = _FOLLOWUP_RE.search(decision)

    if not match:
        console.print("[bold red][RESEARCH][/bold red] Could not parse Follow-up. Skipping.")
        output = {"scratchpad": ["Q: (parse error)\nA: Could not parse sub-question."]}
        log_event("node_end", {
            "node_name": "research",
            "output": output
        })
        return output

    tool_hint    = match.group("tool_bracket") or match.group("tool_bare")  # either bracket or bare form
    sub_question  = match.group("question").strip()

    tool_name = pick_tool(tool_hint, sub_question,WIKI_KEYWORDS,TOOLS)
    tool_fn   = TOOLS[tool_name]

    console.print(
        f"\n[bold blue][RESEARCH][/bold blue] "
        f"Sub-question: [italic]{sub_question}[/italic]"
    )
    console.print(f"[bold blue][RESEARCH][/bold blue] Using tool: [cyan]{tool_name}[/cyan]")

    log_event("tool_start", {
        "tool_name": tool_name,
        "arguments": {"input": sub_question}
    })
    tool_start_time = time.perf_counter()

    try:
        if tool_name == "wikipedia_lookup":
            answer = tool_fn.invoke({"topic": sub_question})
        elif tool_name == "calculator":
            answer = tool_fn.invoke({"expression": sub_question})
        else:  # web_search
            answer = tool_fn.invoke({"query": sub_question})
        
        tool_latency = time.perf_counter() - tool_start_time
        log_event("tool_end", {
            "tool_name": tool_name,
            "output": answer,
            "latency_seconds": tool_latency,
            "status": "success"
        })
    except Exception as e:
        tool_latency = time.perf_counter() - tool_start_time
        answer = f"Tool error: {e}"
        log_event("tool_end", {
            "tool_name": tool_name,
            "output": answer,
            "latency_seconds": tool_latency,
            "status": "error",
            "error": str(e)
        })

    # Truncate very long results to keep the scratchpad manageable
    if len(answer) > 800:
        answer = answer[:800] + "... [truncated]"

    console.print(f"[bold blue][RESEARCH][/bold blue] Answer: {answer[:150]}...")

    entry = f"Q: {sub_question}\nA: {answer}"
    output = {"scratchpad": [entry]}
    log_event("node_end", {
        "node_name": "research",
        "output": output
    })
    return output


# ──────────────────────────────────────────────
# 7. SYNTHESIZER NODE
# ──────────────────────────────────────────────



def synthesizer_node(state: AgentState) -> dict:
    """
    If the decomposer already gave "Final Answer:", extract it.
    Otherwise (safety-limit case), call the LLM to synthesize from the scratchpad.
    """
    log_event("node_start", {
        "node_name": "synthesizer",
        "state": {
            "question": state["question"],
            "scratchpad": state.get("scratchpad", []),
            "final_answer": state.get("final_answer", "")
        }
    })

    console.print("\n[bold green][SYNTHESIZER][/bold green] Composing final answer...")

    decision = state.get("final_answer", "")

    # Case 1: Decomposer already provided the answer inline
    final_match = _FINAL_RE.search(decision)
    if final_match and decision != "__synthesize__":
        raw_answer = final_match.group(1).strip()
        console.print("[bold green][SYNTHESIZER][/bold green] Using decomposer's inline answer.")
        output = {"final_answer": raw_answer}
        log_event("node_end", {
            "node_name": "synthesizer",
            "output": output
        })
        return output

    # Case 2: Safety limit hit or unclear — synthesize from scratchpad using LLM
    llm = get_llm()
    scratchpad_text = format_scratchpad(state.get("scratchpad", []))

    prompts = [
        {"role": "user", "content": SYNTHESIZER_PROMPT.format(
            question=state["question"],
            scratchpad=scratchpad_text,
        )}
    ]
    log_event("llm_start", {
        "node_name": "synthesizer",
        "prompts": prompts,
        "model": getattr(llm, "model_name", getattr(llm, "model", "unknown"))
    })

    llm_start_time = time.perf_counter()
    try:
        response = llm.invoke([
            HumanMessage(content=SYNTHESIZER_PROMPT.format(
                question=state["question"],
                scratchpad=scratchpad_text,
            ))
        ])
        llm_latency = time.perf_counter() - llm_start_time
        raw_answer = _coerce_text(response.content).strip()

        token_usage = getattr(response, "usage_metadata", None)
        if not token_usage:
            token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        log_event("llm_end", {
            "node_name": "synthesizer",
            "response": raw_answer,
            "latency_seconds": llm_latency,
            "token_usage": token_usage
        })
    except Exception as e:
        llm_latency = time.perf_counter() - llm_start_time
        log_event("llm_end", {
            "node_name": "synthesizer",
            "error": str(e),
            "latency_seconds": llm_latency,
            "status": "error"
        })
        raise

    console.print("[bold green][SYNTHESIZER][/bold green] Done.")
    output = {"final_answer": raw_answer}
    log_event("node_end", {
        "node_name": "synthesizer",
        "output": output
    })
    return output


# ──────────────────────────────────────────────
# 8. BUILD THE LANGGRAPH
# ──────────────────────────────────────────────

def build_graph() -> Any:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("decomposer",  decomposer_node)
    graph.add_node("research",    research_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry
    graph.set_entry_point("decomposer")

    # Decomposer → router → research OR synthesize
    graph.add_conditional_edges(
        "decomposer",
        router,
        {
            "research":   "research",
            "synthesize": "synthesizer",
        }
    )

    # After research, loop back to decomposer for the next sub-question
    graph.add_edge("research", "decomposer")

    # Synthesizer ends the graph
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ──────────────────────────────────────────────
# 9. PUBLIC API
# ──────────────────────────────────────────────

def run_agent(question: str) -> str:
    """Run the query decomposer agent on a complex question and return the final answer."""
    app = build_graph()

    console.print(Rule("[bold yellow]Query Decomposer Agent[/bold yellow]"))
    console.print(Panel(
        f"[bold]Question:[/bold] {question}",
        title="[bold yellow]Starting[/bold yellow]",
        expand=False,
        border_style="yellow",
    ))

    log_event("workflow_start", {
        "question": question
    })
    start_time = time.perf_counter()

    try:
        final_state = app.invoke({
            "question":     question,
            "scratchpad":   [],
            "final_answer": "",
            "iterations":   0,
        })
        latency = time.perf_counter() - start_time
        log_event("workflow_end", {
            "question": question,
            "final_answer": final_state.get("final_answer", ""),
            "iterations": final_state.get("iterations", 0),
            "scratchpad": final_state.get("scratchpad", []),
            "latency_seconds": latency
        })
    except Exception as e:
        latency = time.perf_counter() - start_time
        log_event("workflow_error", {
            "question": question,
            "error": str(e),
            "latency_seconds": latency
        })
        raise

    console.print()
    console.print(Rule("[bold green]Final Answer[/bold green]"))
    console.print(Panel(
        Markdown(final_state["final_answer"]),
        title="[bold green]Answer[/bold green]",
        expand=False,
        border_style="green",
    ))

    # Show reasoning summary
    scratchpad = final_state.get("scratchpad", [])
    if scratchpad:
        console.print(f"\n[dim]Research summary: {len(scratchpad)} sub-questions answered "
                      f"in {final_state.get('iterations', 0)} iterations[/dim]")

    return final_state["final_answer"]


# ──────────────────────────────────────────────
# 10. MAIN — example questions
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── Example 1: Multi-hop factual (biography + follow-up) ──────────────────
    # run_agent(
    #     "Who invented the Python programming language, "
    #     "and what university did they study at?"
    # )

    # print("\n\n")

    # ── Example 2: Comparative + calculation ─────────────────────────────────
    # run_agent(
    #     "What are the approximate GDPs of India and Germany? "
    #     "Calculate which is larger and by what percentage."
    # )

    # ── Example 3: Deep multi-hop research ───────────────────────────────────
    run_agent(
        "What is LangChain, who created it, and what are its main competitors "
        "in the LLM application framework space?"
    )

    # ── Example 4: Historical + current ──────────────────────────────────────
    # run_agent(
    #     "What is the capital of Australia, how did it get chosen as the capital, "
    #     "and what is its current population?"
    # )
