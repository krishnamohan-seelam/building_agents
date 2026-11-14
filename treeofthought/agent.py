"""
Tree of Thought (ToT) Agent with LangGraph
=============================================
Pattern: Generate → Evaluate → Prune (beam search) → Synthesize
  Generator LLM  → k candidate next-steps per active thought
  Evaluator LLM  → scores each candidate path 0-10
  Router         → "prune" (keep looping) or "finish" (solved/budget exhausted)
  Pruner         → keeps the best `beam_width` thoughts as the next frontier
  Synthesizer LLM → writes up the best thought seen across the whole search

Install:
  pip install langgraph langchain langchain-openai python-dotenv rich
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Annotated, Any, List, Optional, TypedDict

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from pydantic import SecretStr
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from treeofthought.memory import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_THREAD_ID,
    format_conversation_history,
)
from treeofthought.prompts import (
    build_generator_messages,
    build_evaluator_messages,
    build_synthesizer_messages,
)
from treeofthought.report import generate_html_report
from treeofthought.utils.thought import Thought, format_path, new_id
from treeofthought.utils.schemas import GeneratorOutput, EvaluatorOutput

load_dotenv()

console = Console()

# ──────────────────────────────────────────────
# STRUCTURED LOGGING CONFIGURATION
# ──────────────────────────────────────────────
logger = logging.getLogger("treeofthought")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "treeofthought.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


def log_event(event_type: str, data: dict) -> None:
    """Log a structured JSON event to the log file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "event_type": event_type,
        "data": data,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
NUM_CANDIDATES_PER_THOUGHT = 3
MAX_DEPTH = 3
BEAM_WIDTH = 2
MAX_TOTAL_THOUGHTS = 6
SOLVED_THRESHOLD = 8.0
MAX_INPUT_CHARS = 4000
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1

GENERATOR_MODEL = os.getenv("TOT_GENERATOR_MODEL", "gpt-5.4-nano")
EVALUATOR_MODEL = os.getenv("TOT_EVALUATOR_MODEL", "gpt-5.4")
SYNTHESIS_MODEL = os.getenv("TOT_SYNTHESIS_MODEL", "gemini-3.1-pro-preview")


def get_llm(model: str, temperature: float):
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=SecretStr(openai_key),
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    raise ValueError("No LLM API key found. Set OPENAI_API_KEY in your .env file.")


def _extract_text(content) -> str:
    """Pull plain text out of a chat response's content.

    Some models (e.g. Gemini with reasoning/signatures) return a list of content
    blocks instead of a plain string; only the "text" blocks matter to us.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)


def get_synthesis_llm(model: str, temperature: float):
    """Gemini for the synthesizer: the final write-up benefits from its larger,
    higher-quality model more than generator/evaluator need it for their short outputs."""
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=SecretStr(google_key),
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    raise ValueError("No LLM API key found. Set GOOGLE_API_KEY in your .env file.")


# ──────────────────────────────────────────────
# 1. GRAPH STATE
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    problem: str  # original question/task, expanded with conversation context.
    messages: Annotated[list, add_messages]  # persisted conversation history (checkpointed).
    frontier: List[Thought]  # current active branches to expand this round.
    all_thoughts: List[Thought]  # full history for *this turn's* search, for logging/synthesis.
    pending_evaluation: List[Thought]
    scored_thoughts: List[Thought]
    best_score_so_far: float
    depth: int  # current tree depth (safety counter).
    max_depth: int  # config, e.g. 3.
    beam_width: int  # how many branches survive pruning each round, e.g. 2-3.
    final_answer: str  # set once a thought clears the threshold or budget is exhausted.


# ──────────────────────────────────────────────
# 2. GENERATOR NODE
# ──────────────────────────────────────────────
def generator_node(agent_state: AgentState) -> dict:
    log_event("node_start", {
        "node_name": "generator",
        "state": {"depth": agent_state["depth"], "frontier_size": len(agent_state["frontier"])},
    })

    parents = agent_state["frontier"]
    problem = agent_state["problem"]
    new_thoughts: List[Thought] = []
    structured_llm = get_llm(GENERATOR_MODEL, temperature=0.7).with_structured_output(GeneratorOutput)

    console.print(
        f"\n[bold magenta][GENERATOR][/bold magenta] "
        f"Depth {agent_state['depth']} — expanding {len(parents)} thought(s)..."
    )

    for parent in parents:
        messages = build_generator_messages(
            problem=problem,
            path_so_far=format_path(parent),
            k=NUM_CANDIDATES_PER_THOUGHT,
        )
        log_event("llm_start", {"node_name": "generator", "parent_id": parent["id"], "messages": messages})
        response: GeneratorOutput = structured_llm.invoke(messages)  # type: ignore[assignment]
        log_event("llm_end", {"node_name": "generator", "response": response.model_dump()})

        for i, candidate in enumerate(response.candidates, start=1):
            thought = Thought(
                id=new_id(parent["id"], sibling_index=i),
                parent_id=parent["id"],
                path=parent["path"] + [candidate.thought],
                content=candidate.thought,
                score=None,
                status="active",
            )
            new_thoughts.append(thought)  # type: ignore[arg-type]

    console.print(f"[bold magenta][GENERATOR][/bold magenta] → {len(new_thoughts)} candidate thought(s)")
    output = {
        "all_thoughts": agent_state["all_thoughts"] + new_thoughts,
        "pending_evaluation": new_thoughts,
    }
    log_event("node_end", {"node_name": "generator", "output_count": len(new_thoughts)})
    return output


# ──────────────────────────────────────────────
# 3. EVALUATOR NODE
# ──────────────────────────────────────────────
def evaluator_node(agent_state: AgentState) -> dict:
    log_event("node_start", {
        "node_name": "evaluator",
        "state": {"pending_count": len(agent_state["pending_evaluation"])},
    })

    scored: List[Thought] = []
    structured_llm = get_llm(EVALUATOR_MODEL, temperature=0.2).with_structured_output(EvaluatorOutput)

    for thought in agent_state["pending_evaluation"]:
        messages = build_evaluator_messages(
            problem=agent_state["problem"],
            path=format_path(thought),
        )
        log_event("llm_start", {"node_name": "evaluator", "thought_id": thought["id"], "messages": messages})
        response: EvaluatorOutput = structured_llm.invoke(messages)  # type: ignore[assignment]
        log_event("llm_end", {"node_name": "evaluator", "response": response.model_dump()})

        thought["score"] = response.score
        if response.score >= SOLVED_THRESHOLD:
            thought["status"] = "solved"
        scored.append(thought)
        console.print(
            f"[bold blue][EVALUATOR][/bold blue] {thought['id']} → score {response.score:.1f}"
        )

    output = {"scored_thoughts": scored}
    log_event("node_end", {"node_name": "evaluator", "output_count": len(scored)})
    return output


# ──────────────────────────────────────────────
# 4. PRUNER NODE
# ──────────────────────────────────────────────
def best_pruned_sibling(all_thoughts: List[Thought]) -> Optional[Thought]:
    pruned = [t for t in all_thoughts if t["status"] == "pruned"]
    if not pruned:
        return None
    return max(pruned, key=lambda t: t["score"])


def pruner_node(agent_state: AgentState) -> dict:
    log_event("node_start", {"node_name": "pruner", "state": {"depth": agent_state["depth"]}})

    scored = agent_state["scored_thoughts"]
    beam_width = agent_state.get("beam_width") or BEAM_WIDTH
    survivors = sorted(scored, key=lambda t: t["score"], reverse=True)[:beam_width]
    survivor_ids = {t["id"] for t in survivors}
    losers = [t for t in scored if t["id"] not in survivor_ids]
    for t in losers:
        t["status"] = "pruned"

    best_this_round = max((t["score"] for t in survivors), default=agent_state["best_score_so_far"])
    if best_this_round < agent_state["best_score_so_far"]:
        reactivated = best_pruned_sibling(agent_state["all_thoughts"])
        if reactivated is not None:
            reactivated["status"] = "active"
            survivors = survivors + [reactivated]

    console.print(
        f"[bold yellow][PRUNER][/bold yellow] kept {len(survivors)} of {len(scored)} thought(s), "
        f"best score this round: {best_this_round:.1f}"
    )

    output = {
        "frontier": survivors,
        "depth": agent_state["depth"] + 1,
        "best_score_so_far": max(agent_state["best_score_so_far"], best_this_round),
    }
    log_event("node_end", {"node_name": "pruner", "output": {
        "frontier_ids": [t["id"] for t in survivors],
        "depth": output["depth"],
        "best_score_so_far": output["best_score_so_far"],
    }})
    return output


# ──────────────────────────────────────────────
# 5. SYNTHESIZER NODE
# ──────────────────────────────────────────────
def synthesizer_node(agent_state: AgentState) -> dict:
    log_event("node_start", {"node_name": "synthesizer", "state": {}})

    evaluated = [t for t in agent_state["all_thoughts"] if t["score"] is not None]
    winner = max(evaluated, key=lambda t: t["score"])

    console.print(
        f"\n[bold green][SYNTHESIZER][/bold green] Writing up winning thought "
        f"{winner['id']} (score {winner['score']:.1f})..."
    )

    messages = build_synthesizer_messages(
        problem=agent_state["problem"],
        winning_path=format_path(winner),
    )
    llm = get_synthesis_llm(SYNTHESIS_MODEL, temperature=0)
    log_event("llm_start", {"node_name": "synthesizer", "winner_id": winner["id"], "messages": messages})
    response = llm.invoke(messages)
    final_answer = _extract_text(response.content)
    log_event("llm_end", {"node_name": "synthesizer", "response": final_answer})

    output = {"final_answer": final_answer}
    log_event("node_end", {"node_name": "synthesizer", "output": output})
    return output


# ──────────────────────────────────────────────
# 6. ROUTER (conditional edge)
# ──────────────────────────────────────────────
def router(agent_state: AgentState) -> str:
    scored = agent_state["scored_thoughts"]
    total_thoughts_so_far = len(agent_state["all_thoughts"])

    max_depth = agent_state.get("max_depth") or MAX_DEPTH
    if any(t["score"] >= SOLVED_THRESHOLD for t in scored):
        return "finish"
    if agent_state["depth"] >= max_depth:
        return "finish"
    if total_thoughts_so_far >= MAX_TOTAL_THOUGHTS:
        return "finish"
    return "prune"


# ──────────────────────────────────────────────
# 7. BUILD THE LANGGRAPH
# ──────────────────────────────────────────────
def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("pruner", pruner_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("generator")
    graph.add_edge("generator", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        router,
        {
            "prune": "pruner",
            "finish": "synthesizer",
        },
    )

    graph.add_edge("pruner", "generator")
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# 8. PUBLIC API
# ──────────────────────────────────────────────
def run_agent(
    user_input: str,
    app: Any = None,
    thread_id: str = DEFAULT_THREAD_ID,
    report_path: Optional[str] = None,
    ask_to_save_report: bool = False,
) -> str:
    """Run one conversational turn of the tree-of-thought agent and return the final answer.

    Prior turns for `thread_id` are pulled from the checkpointer (when `app` was
    compiled with one) and folded into the problem statement so follow-up requests
    ("make it shorter", "try again but funnier") can refer back to earlier turns.
    """
    if len(user_input) > MAX_INPUT_CHARS:
        raise ValueError(f"user_input exceeds MAX_INPUT_CHARS ({MAX_INPUT_CHARS})")

    if app is None:
        app = build_graph()

    config = {"configurable": {"thread_id": thread_id}}
    if app.checkpointer is None:
        prior_messages = []
    else:
        prior_state = app.get_state(config)
        prior_messages = prior_state.values.get("messages", []) if prior_state and prior_state.values else []
    history = format_conversation_history(prior_messages)
    problem = (
        user_input if history == "None yet."
        else f"Conversation so far:\n{history}\n\nNew request: {user_input}"
    )

    console.print(Rule("[bold yellow]Tree of Thought Agent[/bold yellow]"))
    console.print(Panel(
        f"[bold]Request:[/bold] {user_input}",
        title="[bold yellow]Starting[/bold yellow]",
        expand=False,
        border_style="yellow",
    ))

    root = Thought(
        id="root",
        parent_id=None,
        path=[],
        content="",
        score=None,  # type: ignore[arg-type]
        status="active",
    )

    log_event("workflow_start", {"thread_id": thread_id, "problem": problem})
    start_time = time.perf_counter()

    try:
        final_state = app.invoke(
            {
                "problem": problem,
                "messages": [HumanMessage(content=user_input)],
                "frontier": [root],
                "all_thoughts": [],
                "pending_evaluation": [],
                "scored_thoughts": [],
                "best_score_so_far": 0.0,
                "depth": 0,
                "max_depth": MAX_DEPTH,
                "beam_width": BEAM_WIDTH,
                "final_answer": "",
            },
            config=config,
        )
        latency = time.perf_counter() - start_time
        log_event("workflow_end", {
            "thread_id": thread_id,
            "problem": problem,
            "final_answer": final_state.get("final_answer", ""),
            "depth": final_state.get("depth", 0),
            "total_thoughts": len(final_state.get("all_thoughts", [])),
            "latency_seconds": latency,
        })
    except Exception as e:
        latency = time.perf_counter() - start_time
        log_event("workflow_error", {
            "thread_id": thread_id,
            "problem": problem,
            "error": str(e),
            "latency_seconds": latency,
        })
        raise

    final_answer = final_state["final_answer"]
    if app.checkpointer is not None:
        app.update_state(config, {"messages": [AIMessage(content=final_answer)]})

    console.print()
    console.print(Rule("[bold green]Final Answer[/bold green]"))
    console.print(Panel(
        Markdown(final_answer),
        title="[bold green]Answer[/bold green]",
        expand=False,
        border_style="green",
    ))

    total_thoughts = len(final_state.get("all_thoughts", []))
    console.print(
        f"\n[dim]Search summary: {total_thoughts} thought(s) explored "
        f"across {final_state.get('depth', 0)} round(s)[/dim]"
    )

    if ask_to_save_report and not report_path:
        report_path = console.input(
            "[dim]Save this answer as an HTML report? "
            "Enter a file/folder path, or press Enter to skip:[/dim] "
        ).strip() or None

    if report_path:
        saved_to = generate_html_report(
            problem=user_input,
            final_answer=final_answer,
            thread_id=thread_id,
            all_thoughts=final_state.get("all_thoughts", []),
            depth=final_state.get("depth", 0),
            latency_seconds=latency,
            output_path=report_path,
        )
        console.print(f"[dim]HTML report saved to {saved_to}[/dim]")

    return final_answer


# ──────────────────────────────────────────────
# 9. MAIN — interactive conversational session
# ──────────────────────────────────────────────
def chat() -> None:
    """Run an interactive, session-persistent chat loop backed by SQLite memory."""
    thread_id = os.getenv("TOT_THREAD_ID", DEFAULT_THREAD_ID)

    with SqliteSaver.from_conn_string(str(DEFAULT_DATABASE_PATH)) as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        console.print(Panel(
            f"[bold]Session:[/bold] {thread_id}\n"
            f"[bold]Memory:[/bold] {DEFAULT_DATABASE_PATH}\n"
            "Type your request, or 'exit'/'quit' to leave.",
            title="[bold cyan]Tree of Thought — Chat[/bold cyan]",
            expand=False,
            border_style="cyan",
        ))

        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]Session ended.[/dim]")
                break

            run_agent(user_input, app=app, thread_id=thread_id, ask_to_save_report=True)


if __name__ == "__main__":
    chat()
