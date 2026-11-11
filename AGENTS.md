# AGENTS.md — building_agents

Instructions for AI coding agents (GitHub Copilot, Claude Code) working in this repo.
Keep this file short and current: if an agent gets something wrong twice, add a rule here.

## Repo at a glance

- Learning workspace for agentic AI with LangChain and LangGraph. Each top-level folder
  (`query_decomposer/`, `smart_query_router/`, `text_to_sql/`, `hitl_agents/`, …) is a
  self-contained example with its own `requirements.txt`, `tests/` and README.
- Root `conftest.py` puts the repo root on `sys.path`, so examples import as packages
  (`from query_decomposer.utils.tools import web_search`).
- Python 3.11+ for all new work (`langgraph dev` requires it), even where an example's
  `pyproject.toml` still says 3.10.
- Secrets live in `.env` (git-ignored). Never print, log or commit a key.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r query_decomposer/requirements.txt
pip install -U "langgraph-cli[inmem]" jsonschema pytest
pytest query_decomposer/tests -q                      # run one example's tests, not everything
langgraph dev                                         # Studio for every graph in langgraph.json
```

## Current project: Agent Trace (presentation layer for the blog)

**Goal.** Record agent runs as JSON traces and replay them on the blog
(`krishnamohan-seelam/my-blog`, Jekyll) with one reusable viewer. Readers see each step —
node, LLM call, tool call, state change — without running anything.

The full design (decisions D0–D8) is in the "Agent Presentation Layer — Design Doc".
The rules below are the parts an agent must never break.

### Milestones (work on one per session)

| ID | Milestone | Done when | Status |
| --- | --- | --- | --- |
| M0 | `langgraph.json` at root; `query_decomposer` opens in Studio | `langgraph dev` shows the 3-node graph | todo |
| M1 | `recorder/trace.schema.json` + hand-written sample trace | sample validates in pytest | done |
| M2 | `recorder/` package: `record query_decomposer --input <n> [--cached]` | output validates; redaction, truncation and pricing unit tests pass; cached rerun makes 0 model calls | done |
| M3 | Viewer in my-blog (separate repo, not here) | sample trace renders under `jekyll serve` | done (pending author's own visual spot-check) |
| M4 | `--publish ../my-blog` and `--preview` | trace copied; post previews locally | todo |

Pilot example: `query_decomposer`. Legacy-agent adapter and multi-input selector come after M4.

### Where things live

```
building_agents/
├── langgraph.json              # registers graph examples for Studio (M0)
├── recorder/                   # new package (M1–M2)
│   ├── trace.schema.json       # THE contract with the blog viewer
│   ├── samples/                # hand-written fixture traces for tests
│   └── tests/
├── traces/<slug>/              # recorded traces: the original copies, committed here
└── query_decomposer/
    └── example.yaml            # title, slug, post URL, 1–3 sample inputs
```

### Trace contract (`agent-trace/1`)

- One JSON file per run: header (`schema`, `example`, `title`, `input`, `recorded_at`,
  `source.commit`, `model`, `pricing.as_of`, `redactions`, optional `graph`) + flat `steps`.
- Step kinds: `node_start`, `node_end`, `llm`, `tool_call`, `tool_result`, `custom`, `state`, `error`.
- `state` steps hold only the fields that changed, never the full state.
- Step `id`s are stable integers from 1; blog posts link to them, so never renumber on re-record
  of the same input without saying so.
- Any change to the format bumps `schema` (`agent-trace/2`) and updates the schema file,
  the samples and the tests in the same change.

### Recorder rules

- Graph examples are recorded through LangGraph streaming (`stream_mode=["updates", "messages", "custom"]`).
  Do not parse the Rich console output or `query_decomposer.log`.
- Redact before writing to disk: API keys, email addresses, account and card numbers become
  `[REDACTED:<kind>]`; count them in `redactions`.
- Truncate long fields to 4 KB (configurable per example in `example.yaml`); set
  `truncated: true` and the original byte size.
- Cost: `cost_inr` per `llm` step from a price table in `recorder/`, with `pricing.as_of` in the header.
- `--cached`: model calls via LangChain's SQLite LLM cache; tool results cached by tool name + arguments.
- `--publish` only copies into `../my-blog/assets/traces/`. Never run git in my-blog; the author commits.

### Instrumenting examples

- Narration goes in custom events, emitted inside a node with
  `from langgraph.config import get_stream_writer` → `get_stream_writer()({"summary": "…"})`.
  Keep each summary to one plain sentence about *why*, not *what*.
- Do not change an example's public behaviour: `build_graph()` and `run_agent()` keep their
  signatures, and the Rich CLI output stays as it is.
- `query_decomposer` specifics: nodes `decomposer` → `research` (loops back) → `synthesizer`;
  state `question`, `scratchpad` (append-only), `final_answer`, `iterations`;
  `get_llm()` picks Groq → Google → OpenAI from whichever key is set; `MAX_ITERATIONS = 5`
  (the README says 8 — the code is the source of truth).

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in `krishnamohan-seelam/building_agents` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), unchanged. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` + `docs/adr/` for the whole repo. See `docs/agents/domain.md`.

## How to work here

1. **Plan before multi-file changes.** Read the relevant code, propose a plan, wait for approval.
   Skip the plan only when the change fits in one sentence.
2. **Tests first.** Write or update the test that defines "done", then the code.
3. **No live API calls in tests.** Mock the LLM and tools with `unittest.mock.patch`, as
   `query_decomposer/tests/test_agent.py` does. Use fixture traces in `recorder/samples/`.
4. **Show evidence, not claims.** Finish with the test command you ran and its output
   (or a screenshot for viewer work).
5. **Stay in scope.** One milestone per session; note anything else you find instead of fixing it.
6. **Never** commit `.env`, edit files in `../my-blog` other than `assets/traces/`, or weaken
   redaction to make a test pass.
