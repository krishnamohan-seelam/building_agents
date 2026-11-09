"""Builds agent-trace/1 header + steps from a LangGraph multi-mode stream (M2-T7).

Consumes the (mode, payload) tuples yielded by
`app.stream(..., stream_mode=["updates", "messages", "custom"])`. LangGraph only tags
`messages` events with their originating node (via `metadata["langgraph_node"]`); `custom`
events carry no node attribution at all. But a node's `messages`/`custom` events are always
flushed to the stream before that node's `updates` event (which fires once, on completion,
keyed by node name) — so we buffer messages/custom events and attribute the whole buffer to
the node named by the next `updates` event.
"""

import time
from typing import Any, Iterable

from recorder.pricing import estimate_cost_inr
from recorder.redact import redact_text
from recorder.truncate import truncate_value, DEFAULT_MAX_BYTES


def _redact_recursive(value: Any) -> tuple[Any, int]:
    """Applies redact_text to every string leaf in a (possibly nested) value."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        count = 0
        out = {}
        for k, v in value.items():
            out[k], c = _redact_recursive(v)
            count += c
        return out, count
    if isinstance(value, list):
        count = 0
        out = []
        for v in value:
            rv, c = _redact_recursive(v)
            out.append(rv)
            count += c
        return out, count
    return value, 0


class _TraceBuilder:
    def __init__(self, price_table: dict, model: str, max_truncate_bytes: int):
        self.price_table = price_table
        self.model = model
        self.max_truncate_bytes = max_truncate_bytes
        self.steps: list[dict] = []
        self.redactions = 0
        self._next_id = 1
        self._start = time.perf_counter()

    def _t_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    def _emit(self, kind: str, node: str, **fields) -> dict:
        step = {"id": self._next_id, "t_ms": self._t_ms(), "kind": kind, "node": node}
        step.update({k: v for k, v in fields.items() if v is not None})
        self._next_id += 1
        self.steps.append(step)
        return step

    def flush_node(self, node_name: str, delta: dict, buffered: list[tuple[str, Any]]) -> None:
        self._emit("node_start", node_name)

        llm_runs: dict[str, dict] = {}
        for mode, payload in buffered:
            if mode == "custom":
                summary, count = redact_text(str(payload.get("summary", "")))
                self.redactions += count
                self._emit("custom", node_name, summary=summary)
            elif mode == "messages":
                chunk, metadata = payload
                run_id = getattr(chunk, "id", None) or "default"
                run = llm_runs.setdefault(run_id, {"content": "", "tokens": None})
                run["content"] += getattr(chunk, "content", "") or ""
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    run["tokens"] = {
                        "in": usage.get("input_tokens", 0),
                        "out": usage.get("output_tokens", 0),
                    }

        for run in llm_runs.values():
            response, count = redact_text(run["content"])
            self.redactions += count
            response, truncated, original_size = truncate_value(response, self.max_truncate_bytes)

            cost_inr = None
            if run["tokens"] is not None:
                try:
                    cost_inr = estimate_cost_inr(
                        self.model, run["tokens"]["in"], run["tokens"]["out"], self.price_table
                    )
                except ValueError:
                    cost_inr = None

            self._emit(
                "llm",
                node_name,
                summary="LLM call",
                detail={"response": response},
                tokens=run["tokens"],
                cost_inr=cost_inr,
                truncated=truncated or None,
                original_size=original_size if truncated else None,
            )

        if delta:
            redacted_delta, count = _redact_recursive(delta)
            self.redactions += count
            self._emit("state", node_name, detail={"changed": redacted_delta})

        self._emit("node_end", node_name)


def assemble_steps(
    events: Iterable[tuple],
    price_table: dict,
    model: str,
    max_truncate_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[list[dict], int]:
    """Turns a LangGraph `updates`/`messages`/`custom` stream into agent-trace/1 steps."""
    builder = _TraceBuilder(price_table, model, max_truncate_bytes)
    buffered: list[tuple[str, Any]] = []

    for mode, payload in events:
        if mode in ("custom", "messages"):
            buffered.append((mode, payload))
        elif mode == "updates":
            node_name, delta = next(iter(payload.items()))
            builder.flush_node(node_name, delta, buffered)
            buffered = []

    return builder.steps, builder.redactions


def build_header(
    *,
    example: str,
    title: str,
    input_text: str,
    recorded_at: str,
    source_commit: str,
    model: str,
    pricing_as_of: str,
    redactions: int,
    graph: dict | None = None,
) -> dict:
    """Assembles the agent-trace/1 header (everything but `steps`)."""
    header = {
        "schema": "agent-trace/1",
        "example": example,
        "title": title,
        "input": input_text,
        "recorded_at": recorded_at,
        "source": {"commit": source_commit},
        "model": model,
        "pricing": {"currency": "INR", "as_of": pricing_as_of},
        "redactions": redactions,
    }
    if graph is not None:
        header["graph"] = graph
    return header
