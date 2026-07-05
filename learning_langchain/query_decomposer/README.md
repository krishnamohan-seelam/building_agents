# Query Decomposer Agent (LangGraph)

An implementation of the **Self-Ask with Search** pattern using LangGraph and free APIs.

The agent tackles complex, multi-part questions by **iteratively decomposing** them into focused sub-questions, answering each with a search tool, and synthesizing a final answer — without needing to know the plan upfront.

---

## How it works

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│  DECOMPOSER (LLM)                           │
│  Prompt: "What follow-up question do        │
│           I need to answer first?"          │
│                                             │
│  Outputs EXACTLY ONE of:                    │
│    Follow-up [tool]: <sub-question>         │
│    Final Answer: <complete answer>          │
└─────────────────────────────────────────────┘
     │                         │
     │ Follow-up               │ Final Answer
     ▼                         ▼
┌──────────────┐         ┌──────────────────┐
│  RESEARCH    │         │  SYNTHESIZER     │
│  Runs tool:  │         │  Formats answer  │
│  - wikipedia │         │  from scratchpad │
│  - web_search│         └────────┬─────────┘
│  - calculator│                  │
│  Appends to  │                  ▼
│  scratchpad  │               [END]
└──────┬───────┘
       │
       └──────► loop back to DECOMPOSER
                (with updated scratchpad)
```

### Key concepts

- **Scratchpad**: A growing list of `Q: <sub-question> / A: <result>` pairs visible to the LLM at every iteration
- **Self-Ask prompting**: The LLM decides what to ask next based on what it already knows
- **Tool routing**: The LLM hints which tool to use via `Follow-up [tool_name]:` — with heuristic fallback
- **Safety limit**: Max 8 iterations to prevent runaway loops

---

## Comparison: Query Decomposer vs Planner-Executor

| | **Query Decomposer** (this) | **Planner-Executor** |
|---|---|---|
| Plan created | Dynamically, one step at a time | Upfront (all steps at once) |
| Next step decided by | Decomposer LLM (each iteration) | Executor LLM (follows fixed plan) |
| Best for | Deep research, multi-hop Q&A | Multi-task workflows with clear stages |
| Stopping condition | LLM signals "Final Answer:" | Fixed number of plan steps |
| Scratchpad | Grows with Q→A pairs | Accumulates step results |

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

### 3. Get your free API keys

| Service | URL | Tier |
|---|---|---|
| Groq (LLM) | <https://console.groq.com> | Free |
| Google Gemini | <https://aistudio.google.com> | Free |
| OpenAI | <https://platform.openai.com> | Paid |
| Tavily | <https://tavily.com> | Free tier |

### 4. Configure `.env`

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here   # optional fallback
OPENAI_API_KEY=your_openai_key_here   # optional fallback
TAVILY_API_KEY=your_tavily_key_here   # optional, for richer search
```

> LLM priority: **Groq → Google → OpenAI** (first key found wins)

### 5. Run the agent

```bash
python query_decomposer_agent.py
```

---

## Example Questions

The agent handles questions requiring multi-hop reasoning:

```python
# Multi-hop biography
run_agent("Who invented Python, and what university did they attend?")

# Comparative + calculation
run_agent(
    "What are the GDPs of India and Germany? "
    "Calculate which is larger and by what percentage."
)

# Deep research
run_agent(
    "What is LangChain, who created it, and what are its main "
    "competitors in the LLM framework space?"
)

# Historical + current
run_agent(
    "What is the capital of Australia, how was it chosen, "
    "and what is its current population?"
)
```

---

## Key LangGraph Concepts Used

| Concept | Where used |
|---|---|
| `TypedDict` state | `AgentState` — typed shared state |
| `Annotated[List, operator.add]` | `scratchpad` — append-only Q→A list |
| `add_conditional_edges` | Router: `decomposer` → `research` or `synthesizer` |
| Looping graph | `research` → `decomposer` creates the self-ask loop |
| `SystemMessage` prompting | Decomposer's self-ask instruction template |

---

## Project Structure

```
query_decomposer/
├── query_decomposer_agent.py   # Main agent
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed to git)
├── utils/
│   └── tools.py                # Underlying WebSearch & WikiSearch implementations
└── README.md                   # This file
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

# Mistral
from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(model="mistral-small", temperature=0)
```

---

## Adding Tools

Tools are plain Python functions decorated with `@tool`. Add them to the `TOOLS` dict:

```python
@tool
def currency_convert(expression: str) -> str:
    """Convert currency. E.g. '100 USD to EUR'."""
    ...

TOOLS = {t.name: t for t in [web_search, wikipedia_lookup, calculator, currency_convert]}
```

The decomposer will naturally route to it via `Follow-up [currency_convert]: ...`
