"""Unit tests for recorder/trace.py (M2-T7). Canned/fake stream events only, no live graph run."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessageChunk

from recorder.trace import assemble_steps, build_header

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "trace.schema.json"

PRICE_TABLE = {
    "as_of": "2026-11-09",
    "models": {"fake-model": {"inr_per_1k_in": 0.04, "inr_per_1k_out": 0.08}},
}


def _chunk(run_id, content, usage=None):
    kwargs = {"content": content, "id": run_id}
    msg = AIMessageChunk(**kwargs)
    if usage:
        msg.usage_metadata = usage
    return msg


def test_assemble_steps_basic_node_cycle():
    events = [
        ("custom", {"summary": "Picked tool web_search"}),
        ("messages", (_chunk("run-1", "Final Answer: 42"), {"langgraph_node": "decomposer"})),
        ("updates", {"decomposer": {"iterations": 1, "final_answer": "Final Answer: 42"}}),
    ]

    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model")

    kinds = [s["kind"] for s in steps]
    assert kinds == ["node_start", "custom", "llm", "state", "node_end"]
    assert redactions == 0

    ids = [s["id"] for s in steps]
    assert ids == list(range(1, len(steps) + 1))

    llm_step = next(s for s in steps if s["kind"] == "llm")
    assert llm_step["detail"]["response"] == "Final Answer: 42"

    state_step = next(s for s in steps if s["kind"] == "state")
    assert state_step["detail"]["changed"] == {"iterations": 1, "final_answer": "Final Answer: 42"}


def test_assemble_steps_merges_streamed_chunks_into_one_llm_step():
    events = [
        ("messages", (_chunk("run-1", "Final "), {"langgraph_node": "decomposer"})),
        ("messages", (_chunk("run-1", "Answer: 42", usage={"input_tokens": 10, "output_tokens": 4}), {"langgraph_node": "decomposer"})),
        ("updates", {"decomposer": {"final_answer": "Final Answer: 42"}}),
    ]

    steps, _ = assemble_steps(events, PRICE_TABLE, model="fake-model")

    llm_steps = [s for s in steps if s["kind"] == "llm"]
    assert len(llm_steps) == 1
    assert llm_steps[0]["detail"]["response"] == "Final Answer: 42"
    assert llm_steps[0]["tokens"] == {"in": 10, "out": 4}
    assert llm_steps[0]["cost_inr"] > 0


def test_assemble_steps_redacts_custom_and_llm_text():
    events = [
        ("custom", {"summary": "contact a@example.com for help"}),
        ("messages", (_chunk("run-1", "email is b@example.com"), {"langgraph_node": "research"})),
        ("updates", {"research": {"scratchpad": ["entry"]}}),
    ]

    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model")

    custom_step = next(s for s in steps if s["kind"] == "custom")
    llm_step = next(s for s in steps if s["kind"] == "llm")
    assert "[REDACTED:email]" in custom_step["summary"]
    assert "[REDACTED:email]" in llm_step["detail"]["response"]
    assert redactions == 2


def test_assemble_steps_flags_truncation_on_long_llm_response():
    events = [
        ("messages", (_chunk("run-1", "x" * 5000), {"langgraph_node": "decomposer"})),
        ("updates", {"decomposer": {"final_answer": "x" * 5000}}),
    ]

    steps, _ = assemble_steps(events, PRICE_TABLE, model="fake-model", max_truncate_bytes=100)

    llm_step = next(s for s in steps if s["kind"] == "llm")
    assert llm_step["truncated"] is True
    assert llm_step["original_size"] == 5000


def test_header_and_steps_together_validate_against_schema():
    events = [
        ("custom", {"summary": "deciding"}),
        ("messages", (_chunk("run-1", "Final Answer: 42", usage={"input_tokens": 5, "output_tokens": 2}), {"langgraph_node": "decomposer"})),
        ("updates", {"decomposer": {"iterations": 1, "final_answer": "Final Answer: 42"}}),
    ]
    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model")

    header = build_header(
        example="query_decomposer",
        title="Test run",
        input_text="What is the answer?",
        recorded_at="2026-11-09T10:00:00+05:30",
        source_commit="abc1234",
        model="fake-model",
        pricing_as_of=PRICE_TABLE["as_of"],
        redactions=redactions,
        graph={"nodes": ["decomposer"], "edges": []},
    )
    trace = {**header, "steps": steps}

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(trace)


def test_assemble_steps_emits_tool_call_and_tool_result():
    events = [
        ("custom", {"summary": "web_search best matches this sub-question's domain, so using it"}),
        ("custom", {"kind": "tool_call", "tool": "web_search", "args": {"query": "who founded Apple"}}),
        ("custom", {"kind": "tool_result", "tool": "web_search", "output": "Steve Jobs and others"}),
        ("updates", {"research": {"scratchpad": ["Q: who founded Apple\nA: Steve Jobs and others"]}}),
    ]

    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model")

    kinds = [s["kind"] for s in steps]
    assert kinds == ["node_start", "custom", "tool_call", "tool_result", "state", "node_end"]
    assert redactions == 0

    tool_call_step = next(s for s in steps if s["kind"] == "tool_call")
    assert tool_call_step["detail"] == {"tool": "web_search", "args": {"query": "who founded Apple"}}

    tool_result_step = next(s for s in steps if s["kind"] == "tool_result")
    assert tool_result_step["detail"] == {"tool": "web_search", "output": "Steve Jobs and others"}


def test_tool_result_is_redacted_and_can_be_truncated():
    events = [
        ("custom", {"kind": "tool_call", "tool": "web_search", "args": {"query": "contact info"}}),
        ("custom", {"kind": "tool_result", "tool": "web_search", "output": "x" * 5000 + " a@example.com"}),
        ("updates", {"research": {"scratchpad": ["entry"]}}),
    ]

    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model", max_truncate_bytes=100)

    tool_result_step = next(s for s in steps if s["kind"] == "tool_result")
    assert tool_result_step["truncated"] is True
    assert tool_result_step["original_size"] > 100
    assert redactions == 1


def test_tool_call_args_are_redacted():
    events = [
        ("custom", {"kind": "tool_call", "tool": "web_search", "args": {"query": "email a@example.com"}}),
        ("updates", {"research": {"scratchpad": ["entry"]}}),
    ]

    steps, redactions = assemble_steps(events, PRICE_TABLE, model="fake-model")

    tool_call_step = next(s for s in steps if s["kind"] == "tool_call")
    assert "[REDACTED:email]" in tool_call_step["detail"]["args"]["query"]
    assert redactions == 1
