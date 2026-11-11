"""Orchestrates one recorded run: record_run(example, input_index, cached) (M2-T8)."""

import importlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from recorder.cache import ToolResultCache, disable_llm_cache, enable_llm_cache
from recorder.pricing import load_price_table
from recorder.trace import assemble_steps, build_header

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "recorder" / "trace.schema.json"
LANGGRAPH_JSON_PATH = REPO_ROOT / "langgraph.json"

_MODEL_ID_MAP = {
    "llama-3.1-8b-instant": "groq/llama-3.1-8b-instant",
    "gemini-1.5-flash": "google/gemini-1.5-flash",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

_PSEUDO_NODES = {"__start__", "__end__"}


def _load_example_config(example: str) -> dict:
    example_yaml_path = REPO_ROOT / example / "example.yaml"
    with open(example_yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_build_graph(example: str):
    with open(LANGGRAPH_JSON_PATH, encoding="utf-8") as f:
        langgraph_config = json.load(f)

    entry = langgraph_config["graphs"][example]
    file_path, func_name = entry.split(":")
    module_path = file_path.removeprefix("./").removesuffix(".py").replace("/", ".")

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _graph_header(app) -> dict:
    g = app.get_graph()
    nodes = [n for n in g.nodes if n not in _PSEUDO_NODES]
    edges = [
        [e.source, e.target]
        for e in g.edges
        if e.source not in _PSEUDO_NODES and e.target not in _PSEUDO_NODES
    ]
    return {"nodes": nodes, "edges": edges}


def _resolve_model_id(get_llm_fn) -> str:
    llm = get_llm_fn()
    raw = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    return _MODEL_ID_MAP.get(raw, raw)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class _CachedTool:
    """Wraps a tool so `.invoke(args)` checks the recorder-side result cache first.

    LangChain's tool objects are pydantic models that reject arbitrary attribute
    assignment (e.g. `tool.invoke = ...`), so this wraps the whole object instead of
    monkeypatching it in place.
    """

    def __init__(self, name: str, tool_fn, cache: ToolResultCache):
        self._name = name
        self._tool_fn = tool_fn
        self._cache = cache

    def invoke(self, args, *a, **kw):
        hit = self._cache.get(self._name, args)
        if hit is not None:
            return hit
        result = self._tool_fn.invoke(args, *a, **kw)
        self._cache.set(self._name, args, result)
        return result


def preview_trace(out_path: Path, blog_path: Path) -> Path:
    """Copies a written trace file into <blog_path>/assets/traces/, for local review in the
    blog's own viewer before the author decides to commit (publish) it themselves.

    A plain file copy only -- never runs git in blog_path, per AGENTS.md's rule that this
    repo's tooling never commits on the author's behalf in the blog repo.
    """
    dest_dir = blog_path / "assets" / "traces"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / out_path.name
    shutil.copyfile(out_path, dest_path)
    return dest_path


def record_run(
    example: str, input_index: int, cached: bool = False, preview_path: str | None = None
) -> dict:
    """Runs `example` on its `input_index`-th sample input (1-based) and writes a validated
    agent-trace/1 trace file under traces/<slug>/. If `preview_path` is given (a path to a
    local my-blog checkout), also copies the trace into its assets/traces/ so it can be
    reviewed in the real viewer via `bundle exec jekyll serve` there -- no commit is made."""
    config = _load_example_config(example)
    sample_inputs = config["sample_inputs"]

    if not (1 <= input_index <= len(sample_inputs)):
        raise ValueError(
            f"--input {input_index} is out of range for {example} "
            f"(has {len(sample_inputs)} sample input(s))"
        )
    question = sample_inputs[input_index - 1]

    build_graph = _load_build_graph(example)
    example_module = importlib.import_module(build_graph.__module__)
    model = _resolve_model_id(example_module.get_llm)
    price_table = load_price_table()
    max_truncate_bytes = config.get("truncation", {}).get("max_bytes", 4096)

    original_tools: dict[str, Any] = {}
    tools_dict = getattr(example_module, "TOOLS", {})
    if cached:
        enable_llm_cache(example)
        tool_cache = ToolResultCache(example)
        for name, tool_fn in tools_dict.items():
            original_tools[name] = tool_fn
            tools_dict[name] = _CachedTool(name, tool_fn, tool_cache)
    else:
        disable_llm_cache()

    try:
        app = build_graph()
        events = app.stream(
            {"question": question, "scratchpad": [], "final_answer": "", "iterations": 0},
            stream_mode=["updates", "messages", "custom"],
        )
        steps, redactions = assemble_steps(events, price_table, model, max_truncate_bytes)
    finally:
        for name, original_tool_fn in original_tools.items():
            tools_dict[name] = original_tool_fn

    header = build_header(
        example=example,
        title=config["title"],
        input_text=question,
        recorded_at=datetime.now(timezone.utc).astimezone().isoformat(),
        source_commit=_git_commit(),
        model=model,
        pricing_as_of=price_table["as_of"],
        redactions=redactions,
        graph=_graph_header(app),
    )
    trace = {**header, "steps": steps}

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(trace)

    slug = config["slug"]
    out_dir = REPO_ROOT / "traces" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{example}-input{input_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    if preview_path is not None:
        preview_trace(out_path, Path(preview_path))

    return trace
