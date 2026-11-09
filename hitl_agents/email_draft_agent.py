"""
# Implementing Human-in-the-Loop in LangGraph: The Edit Pattern

## Learning Objectives

By the end of this tutorial, you will be able to:

1. Set up checkpointing in LangGraph (required for human-in-the-loop)
2. Use `interrupt()` to pause workflow execution at critical points
3. Resume workflows with `Command(resume=<value>)` to pass data back to the interrupt
4. Use `interrupt_before` as an alternative interrupt mechanism

## What You'll Build

We'll build an **Email Draft Agent** that:
1. Takes a user's request for an email
2. Generates a draft email using an LLM
3. **Pauses for human review** - allowing the human to edit the draft
4. Proceeds to "send" the email after human review

## Running

From the repo root (not from inside hitl_agents/), so the `hitl_agents` package is importable:

    python -m hitl_agents.email_draft_agent          # interactive: you approve/edit/reject
    python -m hitl_agents.email_draft_agent --auto   # scripted demos, no prompts
"""
import argparse
import logging
import os
from functools import lru_cache
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hitl_agents.prompts import build_email_draft_messages

log = logging.getLogger(__name__)

Status = Literal["drafting", "pending_review", "reviewed", "rejected", "sent"]

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7

EDITED_DRAFT = """Subject: Time Off Request - Friday, [Date]

Hi [Manager's Name],

I am writing to request time off next Friday for a medical appointment scheduled at 10:00 AM. I expect to be out for approximately half the day.

I will ensure all urgent tasks are completed before I leave and will be available by phone for any critical matters.

Please let me know if you need any additional information.

Thank you,
[Your Name]"""


class EmailAgentState(TypedDict):
    request: str
    email_draft: str
    status: Status


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def _banner(title: str, char: str = "=") -> None:
    log.info(char * 60)
    log.info(title)
    log.info(char * 60)


def _log_draft(title: str, draft: str) -> None:
    log.info(title)
    log.info("-" * 40)
    log.info(draft)
    log.info("-" * 40)


def generate_draft(state: EmailAgentState) -> dict:
    request = state["request"]
    log.info("Generating email draft for request: %s...", request[:50])
    response = get_llm().invoke(build_email_draft_messages(request))
    log.info("Draft generated successfully!")
    return {"email_draft": response.content, "status": "pending_review"}


def human_review(state: EmailAgentState) -> dict:
    """Pause with interrupt(); resume with an edited draft (str), True to accept, False to reject."""
    _banner("HUMAN REVIEW REQUIRED")
    _log_draft("Current email draft:", state["email_draft"])
    human_input = interrupt(
        {
            "message": "Please review the email draft. You can edit it before sending.",
            "current_draft": state["email_draft"],
        }
    )
    if human_input is False:
        log.info("Human rejected the draft. Email will not be sent.")
        return {"status": "rejected"}
    log.info("Human review complete. Proceeding with the email.")
    if isinstance(human_input, str):
        return {"email_draft": human_input, "status": "reviewed"}
    return {"status": "reviewed"}


def route_after_review(state: EmailAgentState) -> Literal["send_email", "__end__"]:
    return END if state["status"] == "rejected" else "send_email"


def send_email(state: EmailAgentState) -> dict:
    """Stub: logs the email instead of sending it."""
    _banner("SENDING EMAIL")
    _log_draft("Final email being sent:", state["email_draft"])
    log.info("Email sent successfully!")
    return {"status": "sent"}


def build_graph(use_interrupt_before: bool = False):
    """Build the email workflow.

    Default: pause inside `human_review` via interrupt().
    With use_interrupt_before=True: no review node; the graph is compiled with
    interrupt_before=["send_email"] and the human edits via update_state().
    """
    workflow = StateGraph(EmailAgentState)
    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("send_email", send_email)
    workflow.add_edge(START, "generate_draft")

    if use_interrupt_before:
        workflow.add_edge("generate_draft", "send_email")
        interrupt_before = ["send_email"]
    else:
        workflow.add_node("human_review", human_review)
        workflow.add_edge("generate_draft", "human_review")
        workflow.add_conditional_edges("human_review", route_after_review)
        interrupt_before = None

    workflow.add_edge("send_email", END)
    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=interrupt_before)


def run_interrupt_demo(request: str) -> None:
    """Edit pattern using interrupt() and Command(resume=...)."""
    app = build_graph()
    config = {"configurable": {"thread_id": "email-session-1"}}
    initial_state = {"request": request, "email_draft": "", "status": "drafting"}

    _banner("PHASE 1: Starting workflow (will pause for human review)")
    result = app.invoke(initial_state, config)

    _banner("INSPECTING PAUSED STATE")
    log.info("Request: %s...", result["request"][:50])
    log.info("Status: %s", result["status"])
    log.info("Draft length: %d characters", len(result.get("email_draft", "")))
    for interrupt_info in result.get("__interrupt__", []):
        log.info("Interrupt message: %s", interrupt_info.value.get("message"))

    _log_draft("CURRENT EMAIL DRAFT FOR REVIEW", result.get("email_draft", ""))
    log.info("Review options:")
    log.info("  1. Accept as-is: Command(resume=True)")
    log.info("  2. Edit the draft: Command(resume='your edited version')")
    _log_draft("HUMAN EDIT: Modified email draft", EDITED_DRAFT)

    _banner("PHASE 2: Resuming workflow with human edits")
    # Same thread_id resumes the paused session; the resume value is returned by interrupt().
    final_result = app.invoke(Command(resume=EDITED_DRAFT), config)

    _banner("WORKFLOW COMPLETE")
    log.info("Final Status: %s", final_result["status"])


def parse_review_reply(reply: str) -> bool | str:
    """Map a terminal reply to a resume value: '' / 'y' accept, 'n' reject, else the edited draft."""
    reply = reply.strip()
    if reply.lower() in ("", "y", "yes"):
        return True
    if reply.lower() in ("n", "no"):
        return False
    return reply


def run_interactive_demo(request: str) -> None:
    """Real human in the loop: the script blocks on input() until the reviewer answers."""
    app = build_graph()
    config = {"configurable": {"thread_id": "email-session-interactive"}}
    initial_state = {"request": request, "email_draft": "", "status": "drafting"}

    result = app.invoke(initial_state, config)
    _log_draft("DRAFT FOR REVIEW", result["email_draft"])
    reply = input("[Enter/y] accept, [n] reject (cancel send), or type a replacement draft: ")

    final_result = app.invoke(Command(resume=parse_review_reply(reply)), config)
    log.info("Final status: %s", final_result["status"])


def run_accept_as_is_demo(request: str) -> None:
    """Accept pattern: resume with Command(resume=True) and keep the generated draft."""
    app = build_graph()
    config = {"configurable": {"thread_id": "email-session-accept"}}
    initial_state = {"request": request, "email_draft": "", "status": "drafting"}

    _banner("ACCEPT AS-IS: running until human review")
    result = app.invoke(initial_state, config)
    generated_draft = result["email_draft"]

    final_result = app.invoke(Command(resume=True), config)
    if final_result["email_draft"] != generated_draft:
        raise RuntimeError("Draft changed although the reviewer accepted it as-is")
    log.info("Draft kept unchanged. Final Status: %s", final_result["status"])


def run_interrupt_before_demo(request: str) -> None:
    """Alternative: interrupt_before pauses ahead of send_email; edit state, then resume with None."""
    app = build_graph(use_interrupt_before=True)
    config = {"configurable": {"thread_id": "email-session-2"}}
    initial_state = {"request": request, "email_draft": "", "status": "drafting"}

    _banner("INTERRUPT_BEFORE: running until send_email")
    app.invoke(initial_state, config)
    snapshot = app.get_state(config)
    log.info("Paused before: %s", snapshot.next)
    _log_draft("Draft awaiting review:", snapshot.values["email_draft"])

    app.update_state(config, {"email_draft": EDITED_DRAFT, "status": "reviewed"})
    final_result = app.invoke(None, config)
    log.info("Final Status: %s", final_result["status"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--auto", action="store_true", help="run scripted demos without prompting")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    email_request = (
        "Write a professional email to my manager requesting time off "
        "next Friday for a medical appointment."
    )
    log.info("Email Request: %s", email_request)

    if args.auto:
        run_interrupt_demo(email_request)
        run_accept_as_is_demo(email_request)
        run_interrupt_before_demo(email_request)
    else:
        run_interactive_demo(email_request)


if __name__ == "__main__":
    main()
