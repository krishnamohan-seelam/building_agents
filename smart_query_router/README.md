# Smart Query Router

An intent-aware agentic query routing module built with **LangGraph**, **OpenAI Structured Outputs**, **Tavily Web Search**, and **Rich CLI UI**.

## Overview

The **Smart Query Router** dynamically evaluates incoming user queries to determine whether they require up-to-date real-time external knowledge (web search) or can be answered directly using the model's parametric knowledge.

It uses a stateful **LangGraph** flow that classifies user intent with Pydantic structured output, routes execution through conditional edges, and presents answers via a Rich terminal UI.

---

## Architecture & Workflow

```
                        ┌──────────────────┐
                        │      START       │
                        └────────┬─────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   intent_detection    │
                     └───────────┬───────────┘
                                 │
                                 ▼
                         [ route_intent ]
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌──────────────────┐                           ┌──────────────────┐
│    web_search    │                           │ direct_response  │
└────────┬─────────┘                           └────────┬─────────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │       END        │
                        └──────────────────┘
```

### Graph Nodes

1. **`intent_detection`**: Invokes OpenAI (`gpt-4o-mini`) using `.with_structured_output(IntentClassification)` to classify query intent into `search` or `direct` alongside structured reasoning.
2. **`web_search`**: Triggered when intent is `search`. Executes a Tavily web search, formats the retrieved web content, and generates a search-augmented answer.
3. **`direct_response`**: Triggered when intent is `direct`. Generates a direct response using internal LLM parametric knowledge.

---

## Key Features

- **Structured Intent Classification**: Uses a strict Pydantic model (`IntentClassification`) to classify intent (`search` / `direct`) and capture decision reasoning.
- **Stateful Routing**: Built on LangGraph `StateGraph` using a typed state schema (`RouterState`).
- **Dependency Injection**: Node functions accept runtime `RunnableConfig` parameters (`configurable={"llm": ..., "search_tool": ...}`), enabling clean testing and component swapping.
- **Rich Interactive CLI**: Features animated status spinners, colored intent badges, structured search-context tables, and rendered Markdown output panels.
- **High Test Coverage**: Includes full unit and integration test suite with 98% code coverage.

---

## Directory Structure

```
smart_query_router/
├── __init__.py           # Package entry point and exports
├── cli.py                # Rich REPL terminal interface
├── graph.py              # StateGraph assembly & conditional routing edge
├── nodes.py              # Node functions (intent detection, search, direct response)
├── prompts.py            # System and human prompt message builders
├── state.py              # RouterState TypedDict & IntentClassification Pydantic schema
├── requirements.txt      # Dependency specifications
└── tests/
    └── test_router.py    # Pytest unit and integration test suite
```

---

## Requirements & Setup

### Environment Prerequisites

Set your API keys in `.env` (or pass them via environment variables):

```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

### Installation

Activate your virtual environment and install requirements:

```bash
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Install dependencies
pip install -r smart_query_router/requirements.txt
```

---

## Usage

### Interactive CLI

Launch the interactive REPL:

```bash
python -m smart_query_router.cli
```

Example interaction:
- **Query**: `"What were the latest findings from the James Webb Space Telescope in 2026?"`
  - **Intent**: `[SEARCH] ROUTED TO WEB SEARCH`
  - **Output**: Shows Tavily search context table followed by search-augmented response.
- **Query**: `"Explain the difference between a stack and a queue."`
  - **Intent**: `[DIRECT] ROUTED TO DIRECT KNOWLEDGE`
  - **Output**: Direct conceptual response.

### Python API Usage

```python
from smart_query_router.graph import build_router_graph

# Compile the router graph
graph = build_router_graph()

# Invoke graph with a query
result = graph.invoke({"query": "What is Quantum Computing?"})

print("Intent:", result["intent"])
print("Reasoning:", result["reasoning"])
print("Response:\n", result["response"])
```

---

## Running Tests

Run the test suite with coverage report using `pytest`:

```bash
pytest --cov=smart_query_router smart_query_router/tests
```
