# Tree of Thought (ToT) Agent (LangGraph)

An implementation of the **Tree of Thought** pattern using LangGraph: instead of one linear
reasoning path, the agent explores several candidate "thoughts" per step, scores them, keeps
the best ones (beam search), and repeats until a thought is good enough to answer with, or a
budget runs out.

---

## How it works

```
Problem
   │
   ▼
┌─────────────────────────────────────────────┐
│  GENERATOR (LLM)                             │
│  k distinct next-step thoughts per active    │
│  branch in the frontier                      │
└────────────────────┬──────────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  EVALUATOR (LLM)                             │
│  Scores each candidate path 0-10             │
└────────────────────┬──────────────────────────┘
                      ▼
                 [ROUTER]
           solved / depth / cost cap?
             │                  │
      finish │                  │ prune
             ▼                  ▼
     ┌───────────────┐   ┌───────────────┐
     │ SYNTHESIZER   │   │ PRUNER        │
     │ Writes up the │   │ Keeps the best│
     │ best thought  │   │ beam_width    │
     │ seen overall  │   │ thoughts      │
     └───────┬───────┘   └───────┬───────┘
             │                   │
            [END]      loop back to GENERATOR
```

### Key concepts

- **Beam search over a thought tree**: state tracks a `frontier` (active branches to expand)
  and `all_thoughts` (full history), not a single scratchpad.
- **Structured outputs**: generator/evaluator/synthesizer bind `with_structured_output()` to
  Pydantic models (`utils/schemas.py`) — no regex parsing of LLM text.
- **Backtracking**: if a round's best score regresses versus the best seen so far, the pruner
  reactivates the best previously-pruned sibling (`best_pruned_sibling`) instead of only
  advancing the current frontier.
- **Safety limits**: `MAX_DEPTH`, `MAX_TOTAL_THOUGHTS`, and `SOLVED_THRESHOLD` bound both the
  search depth and total LLM calls per run.
- **Conversation memory**: `memory.py` persists turns via LangGraph's `SqliteSaver` so follow-up
  requests ("make it shorter") can refer to earlier turns in the same `thread_id`. This history is
  replayed into future prompts verbatim with no sanitization — a known prompt-injection surface if
  this agent is ever exposed to untrusted multi-turn input; treat `thread_id` content as trusted.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
OPENAI_API_KEY=your_openai_key_here

# Optional — override the default models per role:
TOT_GENERATOR_MODEL=gpt-5.4-nano
TOT_EVALUATOR_MODEL=gpt-5.4
TOT_SYNTHESIS_MODEL=gpt-5.4
```

### 4. Run the agent

```bash
python -m treeofthought.agent
```

---

## Example

```python
from treeofthought.agent import run_agent

run_agent("Choose the best 3-word slogan for a solar-powered backpack.")
```

Pick problems where branching visibly matters and is easy to hand-score (slogans, small
logic/math puzzles) — open-ended research questions are `query_decomposer`'s job, not this
agent's.

---

## Project Structure

```
treeofthought/
├── agent.py                    # Graph nodes, router, build_graph, run_agent, chat REPL
├── memory.py                   # Cross-turn conversation-history helpers (SqliteSaver-backed)
├── prompts.py                  # Generator/evaluator/synthesizer prompts
├── report.py                   # Renders a run into a self-contained HTML report
├── plan.md                     # Original implementation plan this module was built from
├── pyproject.toml               # Packaging setup
├── requirements.txt             # Python dependencies
├── templates/report_template.html
├── tests/                       # Test files (LLM mocked, no live API calls)
└── utils/
    ├── thought.py               # Thought TypedDict, new_id, format_path
    └── schemas.py                # Pydantic structured-output models
```

---

## Swapping the LLM

```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# Ollama (fully local)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1")
```

`get_llm()` in `agent.py` is the single seam to swap — every node calls it lazily per-invocation,
so changing providers there changes it everywhere.
