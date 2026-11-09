# Tree of Thought (ToT) Agent — Implementation Plan

Mirrors the structure of `query_decomposer/` (LangGraph `StateGraph`, `TypedDict` state,
one function per node, structured JSON logging, prompts in `prompts.py`, helpers in
`utils/`). This doc is the plan only — write the code yourself, then ask for review.

## 1. Core idea

Chain-of-thought = one linear reasoning path. Tree-of-thought = explore several
candidate "thoughts" per step, score them, keep the good ones, discard the rest,
and repeat until a thought is good enough to answer with (or a budget runs out).

State needs to track a **tree/frontier of branches**, not just one scratchpad.

## 2. State shape (`AgentState`, TypedDict)

- `problem: str` — original question/task.
- `frontier: list[Thought]` — current active branches to expand this round.
- `all_thoughts: Annotated[list[Thought], operator.add]` — full history, for logging/debugging.
- `pending_evaluation: list[Thought]` — this round's freshly generated, not-yet-scored
  thoughts; written by `generator_node`, read by `evaluator_node`. Plain overwrite
  each round (no reducer) — unlike `all_thoughts`, it's not meant to accumulate.
- `scored_thoughts: list[Thought]` — this round's thoughts after evaluation; written
  by `evaluator_node`, read by `router` and `pruner_node`. Also plain overwrite.
- `best_score_so_far: float` — highest score seen in any prior round; written by
  `pruner_node`, used to detect a round that regressed (feeds the v2 backtracking
  check in §4.4). Init to `0` (or `-inf`) in `run_agent`'s initial state.
- `depth: int` — current tree depth (safety counter, like `iterations` in query_decomposer).
- `max_depth: int` — config, e.g. 3.
- `beam_width: int` — how many branches survive pruning each round, e.g. 2–3.
- `final_answer: str` — set once a thought clears the threshold or budget is exhausted.

`Thought` (a small TypedDict or dataclass, not full LangChain messages):

- `id: str` — unique id (e.g. `f"{parent_id}.{n}"`) so you can reconstruct the path.
- `parent_id: str | None`
- `path: list[str]` — reasoning steps from root to this node (for building prompts/context).
- `content: str` — the thought text itself.
- `score: float` — heuristic score from the evaluator.
- `status: Literal["active", "pruned", "solved"]`

Keeping `Thought` separate from the LangGraph state keys (rather than raw strings, as
query_decomposer does with scratchpad) is the main structural difference — you need
structure to branch/merge/prune.

## 3. Structured outputs (replaces regex parsing)

query_decomposer parses free-text LLM output with regex (`_FOLLOWUP_RE`, `_FINAL_RE`)
because it only ever extracts one line. ToT needs a *list* of candidates plus a
*score*, so use LangChain's `with_structured_output()` with Pydantic models instead —
no regex, no `_coerce_text` munging, the LLM call itself returns a validated object.

```python
# utils/schemas.py
from pydantic import BaseModel, Field

class ThoughtCandidate(BaseModel):
    thought: str = Field(description="The next reasoning step, self-contained.")

class GeneratorOutput(BaseModel):
    candidates: list[ThoughtCandidate] = Field(
        description="k distinct next-step thoughts, no duplicates."
    )

class EvaluatorOutput(BaseModel):
    score: float = Field(ge=0, le=10, description="Heuristic quality score, 0-10.")
    reason: str = Field(description="One-sentence justification for the score.")

class SynthesisOutput(BaseModel):
    final_answer: str = Field(description="Complete, well-formatted answer to the problem.")
```

Bind once per node: `generator_llm = get_llm().with_structured_output(GeneratorOutput)`,
`evaluator_llm = get_llm().with_structured_output(EvaluatorOutput)`. Each `.invoke(...)`
then returns an already-parsed model instance — `result.candidates`, `result.score`,
etc. Keep the `log_event()` calls around each `.invoke()` exactly like query_decomposer
does (log the prompt in, the parsed object out); structured output doesn't remove the
need for the audit trail, it just removes the regex step in between.

Not every provider/model supports tool-calling-based structured output equally well —
if `with_structured_output` misbehaves with your chosen LLM, the fallback is a JSON
`response_format` + `model_validate_json`, but try `with_structured_output` first since
it's the idiomatic LangChain way and keeps parity with how you'd extend this later.

## 4. Nodes

### 4.1 Generator node (`generator_node`)

Prompts used: `GENERATOR_SYSTEM`, `GENERATOR_HUMAN_TEMPLATE` (see §6).
Structured output: `GeneratorOutput`.

```
function generator_node(state):
    log node_start
    parents = state.frontier                     # thoughts to expand this round
    new_thoughts = []

    for parent in parents:
        messages = build_generator_messages(
            problem = state.problem,
            path_so_far = format_path(parent),     # "" for the synthetic root
            k = NUM_CANDIDATES_PER_THOUGHT,
        )
        log llm_start(node="generator", messages)
        result: GeneratorOutput = generator_llm.invoke(messages)   # structured
        log llm_end(node="generator", result)

        for candidate in result.candidates:
            new_thoughts.append(Thought(
                id = new_id(parent.id),
                parent_id = parent.id,
                path = parent.path + [candidate.thought],
                content = candidate.thought,
                score = None,
                status = "active",
            ))

    log node_end(new_thoughts)
    return { "all_thoughts": new_thoughts, "pending_evaluation": new_thoughts }
```

Note: frontier size can grow to `len(parents) * k` — this is the branching step.
On the very first call, `parents` is a single synthetic root `Thought` with empty
`path`, so the graph always starts generator → evaluator → router, same entry
shape as query_decomposer.

### 4.2 Evaluator/Critic node (`evaluator_node`)

Prompts used: `EVALUATOR_SYSTEM`, `EVALUATOR_HUMAN_TEMPLATE` (see §6).
Structured output: `EvaluatorOutput`.

```
function evaluator_node(state):
    log node_start
    scored = []

    for thought in state.pending_evaluation:
        messages = build_evaluator_messages(
            problem = state.problem,
            path = format_path(thought),
        )
        log llm_start(node="evaluator", messages)
        result: EvaluatorOutput = evaluator_llm.invoke(messages)   # structured
        log llm_end(node="evaluator", result)

        thought.score = result.score
        if result.score >= SOLVED_THRESHOLD:
            thought.status = "solved"
        scored.append(thought)

    log node_end(scored)
    return { "scored_thoughts": scored }
```

Start with one LLM call per thought (mirrors query_decomposer's one-tool-call-per-step
simplicity). A cheap non-LLM heuristic scorer is a valid alternative/addition later —
note as v2, don't build both up front.

### 4.3 Router (conditional edge, like query_decomposer's `router`)

Not a node — a plain function called after `evaluator_node` to pick the next edge.

```
function router(state) -> "prune" | "finish":
    scored = state.scored_thoughts
    total_thoughts_so_far = len(state.all_thoughts)

    if any(t.score >= SOLVED_THRESHOLD for t in scored):
        return "finish"
    if state.depth >= MAX_DEPTH:
        return "finish"                      # force best-effort, like MAX_ITERATIONS
    if total_thoughts_so_far >= MAX_TOTAL_THOUGHTS:
        return "finish"                      # cost safety cap
    return "prune"
```

### 4.4 Backtracking/Pruning node (`pruner_node`)

No LLM call — pure Python over `state.scored_thoughts`.

```
function pruner_node(state):
    log node_start
    scored = state.scored_thoughts
    survivors = sorted(scored, key=lambda t: t.score, reverse=True)[:BEAM_WIDTH]
    losers = [t for t in scored if t not in survivors]

    for t in losers:
        t.status = "pruned"

    best_this_round = max(t.score for t in survivors)
    if best_this_round < state.best_score_so_far:
        # optional v2: pull back a previously pruned sibling instead of only
        # advancing the current frontier — dead-end recovery.
        survivors = survivors + best_pruned_sibling(state.all_thoughts)

    log node_end(survivors)
    return {
        "frontier": survivors,
        "depth": state.depth + 1,
        "best_score_so_far": max(state.best_score_so_far, best_this_round),
    }
```

Loop edge: `pruner → generator` (same loop shape as query_decomposer's
`research → decomposer`). Start without the v2 backtracking branch — plain beam
search first, add dead-end recovery once the basic loop works.

`best_pruned_sibling` (v2 only — not needed for the v1 build order in §10):

```
function best_pruned_sibling(all_thoughts) -> Thought | None:
    # Reaches back into history for a thought that lost the beam cut in a
    # PREVIOUS round, so we can re-activate it if this round's frontier regressed.
    pruned = [t for t in all_thoughts if t.status == "pruned"]
    if pruned is empty:
        return None
    return max(pruned, key=score)
```

Two design choices to make when you actually build this, both fine to defer:
- Scope: `all_thoughts` here means *every* pruned thought ever, so the pick could
  reach back several rounds. Restricting to `len(t.path) == state.depth` (only last
  round's losers — a thought's depth is just how many steps are in its `path`) is
  the more conservative, easier-to-reason-about version.
- Re-pruning: once re-activated into `frontier`, set its `status` back to `"active"`
  so it isn't picked again by a later `best_pruned_sibling` call.

### 4.5 Synthesizer/Finish node (`synthesizer_node`)

Prompt used: `SYNTHESIZER_PROMPT` (see §6). Structured output: `SynthesisOutput`.

```
function synthesizer_node(state):
    log node_start
    evaluated = [t for t in state.all_thoughts if t.score is not None]  # exclude root
    winner = max(evaluated, key=lambda t: t.score)     # best thought seen, any round

    messages = build_synthesizer_messages(
        problem = state.problem,
        winning_path = format_path(winner),
    )
    log llm_start(node="synthesizer", messages)
    result: SynthesisOutput = synthesizer_llm.invoke(messages)   # structured
    log llm_end(node="synthesizer", result)

    log node_end(result)
    return { "final_answer": result.final_answer }
```

Works whether we got here via `SOLVED_THRESHOLD`, `MAX_DEPTH`, or `MAX_TOTAL_THOUGHTS`
— always just "take the best thought seen and write it up," same graceful-degradation
spirit as query_decomposer's safety-limit synthesis path.

## 5. Graph wiring (`build_graph`)

```
generator → evaluator → [router] → prune  → generator   (loop)
                               \--> finish → synthesizer → END
```

- `set_entry_point("generator")`
- `add_conditional_edges("evaluator", router, {"prune": "pruner", "finish": "synthesizer"})`
- `add_edge("pruner", "generator")`
- `add_edge("synthesizer", END)`

This is structurally the same loop-plus-exit shape as query_decomposer
(`decomposer → router → research/synthesizer`, `research → decomposer`), just with
two extra stages (evaluate, prune) inside the loop body.

## 6. Prompts (`prompts.py`)

Follow query_decomposer's pattern: raw prompt strings + a `build_*_messages()` helper
using `PromptBuilder`. Since structured output (§3) handles the output *format*, these
prompts only need to describe the output *content* — no "respond in exactly this
format" scaffolding like query_decomposer's `Follow-up:`/`Final Answer:` needed.

```python
GENERATOR_SYSTEM = """You are a creative problem-solving agent exploring a tree of
reasoning paths to solve a problem. Given the problem and the reasoning path so far,
propose {k} distinct, plausible next steps. Each step should:
  - Follow logically from the path so far (or start fresh if the path is empty).
  - Be meaningfully different from the other candidates you propose (no near-duplicates).
  - Be a single concrete step, not a full solution.
"""

GENERATOR_HUMAN_TEMPLATE = """Problem: {problem}

Reasoning path so far:
{path_so_far}

Propose {k} distinct next steps."""


EVALUATOR_SYSTEM = """You are a critical evaluator scoring one reasoning path toward
solving a problem. Score how promising this path is on a scale of 0-10, where:
  - 0-3: flawed, contradicts the problem, or a dead end.
  - 4-6: plausible but incomplete or unremarkable.
  - 7-10: correct so far and clearly converging on a strong final answer.
Judge only what's written — do not reward length or confidence.
"""

EVALUATOR_HUMAN_TEMPLATE = """Problem: {problem}

Reasoning path to evaluate:
{path}

Give a score and a one-sentence reason."""


SYNTHESIZER_PROMPT = """You are a synthesis agent. A tree-of-thought search has found
its best reasoning path for the problem below. Turn it into a complete, well-formatted
final answer.

Problem: {problem}

Winning reasoning path:
{winning_path}

Write the final answer, directly addressing the problem, using this path's reasoning."""


def build_generator_messages(problem: str, path_so_far: str, k: int):
    return (
        PromptBuilder("generator")
        .system(GENERATOR_SYSTEM)
        .human(GENERATOR_HUMAN_TEMPLATE)
        .build(problem=problem, path_so_far=path_so_far or "(none yet — this is the root)", k=k)
    )


def build_evaluator_messages(problem: str, path: str):
    return (
        PromptBuilder("evaluator")
        .system(EVALUATOR_SYSTEM)
        .human(EVALUATOR_HUMAN_TEMPLATE)
        .build(problem=problem, path=path)
    )


def build_synthesizer_messages(problem: str, winning_path: str):
    return (
        PromptBuilder("synthesizer")
        .human(SYNTHESIZER_PROMPT)
        .build(problem=problem, winning_path=winning_path)
    )
```

## 7. Utils (`utils/`)

- `schemas.py` — the Pydantic models from §3 (`ThoughtCandidate`, `GeneratorOutput`,
  `EvaluatorOutput`, `SynthesisOutput`).
- `thought.py` — the `Thought` TypedDict/dataclass, an `id` generator (`new_id`), and a
  `format_path(thought)` helper (turns `path: list[str]` into readable numbered-step
  text for prompts — analogous to `format_scratchpad`).

  ```
  function new_id(parent_id, sibling_index) -> str:
      # Encodes the full ancestry path into the id itself, so the tree structure
      # is readable from ids alone (no separate lookup table needed for logging).
      if parent_id is None:
          return "root"
      return f"{parent_id}.{sibling_index}"
  ```

  Called once per candidate inside `generator_node`'s loop (§4.1), right before
  constructing each new `Thought`. Expanding `"root"` with k=3 candidates gives
  `"root.1"`, `"root.2"`, `"root.3"`; if `"root.2"` survives pruning and is expanded
  again, its children are `"root.2.1"`, `"root.2.2"`, `"root.2.3"` — so printing the
  tree for debugging is just sorting `all_thoughts` by id prefix.

  ```
  function format_path(thought) -> str:
      # thought.path is the list of reasoning-step strings from root to this thought,
      # e.g. ["Consider X", "Given X, try Y"] — same role as query_decomposer's
      # scratchpad list, but ordered steps of ONE path instead of Q/A pairs.
      if thought.path is empty:
          return "  (root — no steps taken yet)"

      lines = []
      for i, step in enumerate(thought.path, start=1):
          lines.append(f"  Step {i}: {step}")
      return join(lines, "\n")
  ```

  Called from `generator_node` (parent's path, to ask "what's next after this?") and
  `evaluator_node`/`synthesizer_node` (a thought's own path, to ask "how good/how do I
  write this up?"). Same signature shape as `format_scratchpad(scratchpad: list[str])`
  — pass it a list of strings, get back prompt-ready indented text — except the input
  list is ordered steps of a single branch rather than an accumulated Q&A history.
- `text.py` — reuse/adapt `_coerce_text` only if you fall back to non-structured calls
  anywhere; with `with_structured_output()` everywhere, this shrinks to near-nothing.

## 8. Logging & safety

- Reuse the exact `log_event()` JSON-line-to-file pattern from `agent.py` — one
  logger per module, `node_start` / `llm_start` / `llm_end` / `node_end` events.
- Safety counter: `depth >= max_depth` forces finish, same idea as
  `MAX_ITERATIONS` in query_decomposer.
- Also cap total LLM calls per run (generator fans out, so cost grows faster than
  query_decomposer's linear loop) — e.g. `MAX_TOTAL_THOUGHTS = 30`, checked in the
  router alongside depth.

## 9. Config knobs to expose (constants at top of `agent.py`, like `MAX_ITERATIONS`)

- `NUM_CANDIDATES_PER_THOUGHT` (k, e.g. 3)
- `BEAM_WIDTH` (e.g. 2)
- `MAX_DEPTH` (e.g. 3)
- `SOLVED_THRESHOLD` (e.g. 8.0 out of 10)
- `MAX_TOTAL_THOUGHTS` (safety cap)

## 10. Suggested build order

1. `Thought` type + `utils/thought.py` helpers.
2. `generator_node` alone — verify it fans out plausible candidate thoughts from a
   single root problem (no evaluation/pruning yet, just print the tree).
3. `evaluator_node` — verify scores look reasonable on the generator's output.
4. `router` + `pruner_node` — verify beam search actually narrows the frontier
   round over round.
5. `synthesizer_node` + full graph wiring — end-to-end run.
6. Add logging parity with query_decomposer once the happy path works.
7. (v2, optional) real backtracking: if a round's best score is worse than the
   previous round's, re-activate a previously pruned sibling instead of only the
   current frontier.

## 11. Example problem to test with

Pick something where branching visibly matters and is easy to hand-score, e.g.
"Choose the best 3-word slogan for a solar-powered backpack" or a small math/logic
puzzle with multiple candidate approaches — avoid open-ended research questions
(that's query_decomposer's job) so the ToT-specific mechanics (generate → score →
prune → backtrack) stay the focus.
