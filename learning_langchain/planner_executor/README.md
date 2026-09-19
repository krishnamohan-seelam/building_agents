# Planner-Executor Agent (LangGraph)

A real-world example of the Planner-Executor multi-agent pattern using LangGraph and free APIs.

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain langchain-groq \
            langchain-community duckduckgo-search \
            wikipedia requests python-dotenv \
            langgraph-checkpoint-sqlite
```

### 2. Get your free API keys

| Service | URL | Tier needed |
| --- | --- | --- |
| Groq (LLM) | <https://console.groq.com> | Free |
| OpenWeatherMap | <https://openweathermap.org/api> | Free (1000 calls/day) |

### 3. Create a `.env` file

```env
GROQ_API_KEY=your_groq_key_here
OPENWEATHER_API_KEY=your_openweathermap_key_here
```

> If you skip `OPENWEATHER_API_KEY`, the weather tool will return mock data.

### 4. Run the Agent

```bash
python planner_executor_agent.py
```

### 5. Run the Trip Advisor Chat Bot

A specialized travel planning agent built on the same architecture:

```bash
python trip_planner.py
```

## Conversation memory

Both agents use LangGraph's SQLite checkpointer. The default CLI session is
stored in `planner_memory.sqlite3` beside these scripts, so follow-up requests
remain available after restarting the process. The default thread ID is
`local_trip_planner`.

To keep separate conversations, call `run_trip_advisor` or `run_agent` with a
different `thread_id`. SQLite persistence is intended for local, single-process
use; use a server-backed checkpointer for multi-worker deployments.

---

## How it works

```
User Goal
   │
   ▼
┌─────────────────────────────────┐
│  PLANNER (LLM)                  │
│  → Outputs a JSON list of steps │
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  GRAPH STATE                    │
│  plan[], results[], step_index  │
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐   ←──────────────────────┐
│  EXECUTOR (LLM + tools)         │                          │
│  Runs current step, calls tools │                          │
└─────────────────────────────────┘                          │
   │                                                         │
   ▼                                                         │
SUPERVISOR (conditional edge)                                │
   ├── more steps? ─────────────────────────────────────────-┘
   └── all done?
         │
         ▼
   ┌──────────────────┐
   │  SYNTHESIZER     │
   │  Final answer    │
   └──────────────────┘
```

## Key LangGraph concepts used

- **`TypedDict` state** — typed shared state passed through every node
- **`Annotated[List, operator.add]`** — append-only list for accumulated results
- **`add_conditional_edges`** — the supervisor's routing logic
- **`ToolNode`** — prebuilt executor for tool calls
- **`bind_tools`** — attaches tools to an LLM so it can call them

## Swapping the LLM

Replace the Groq lines with any compatible provider:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# Ollama (fully local, no API key)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1")
```

## Adding more tools

Tools are plain Python functions decorated with `@tool`:

```python
@tool
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies using a free exchange rate API."""
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    data = requests.get(url).json()
    rate = data["rates"][to_currency]
    return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"
```

Then add it to the `TOOLS` list and LangGraph will route to it automatically.
