# Smart Query Router Implementation Prompt

**Role / Persona:** Act as a Senior Agentic AI Python Engineer specializing in stateful agentic workflows using LangGraph, LangChain, Pydantic, and OpenAI.

---

## 1. Objective & Task

Build a modular, production-ready **Smart Query Router** application in Python using **LangGraph**. The application must:
1. Receive a user query.
2. Detect if the question requires real-time/current web information (`"search"`) or can be answered directly from internal model knowledge (`"direct"`).
3. Route the execution path dynamically via conditional edges in a LangGraph `StateGraph`.
4. Augment responses with Tavily web search results when routing to the web search node.
5. Provide an interactive REPL CLI loop for user interaction until exited with `exit`, `quit`, `bye`, or `stop`.

---

## 2. Architecture & Workflow

```
START ──> Intent Detection Node ──> [Conditional Edge: route_intent]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
            Web Search Node                                Direct Response Node
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                                           END
```

---

## 3. Tech Stack & Dependencies

- **Framework**: LangGraph (`StateGraph`, `START`, `END`)
- **LLM Provider**: OpenAI (`ChatOpenAI` via `langchain-openai`)
- **Structured Output**: Pydantic (`pydantic.BaseModel`) with `llm.with_structured_output(...)`
- **Web Search Integration**: Tavily Search (`TavilySearchResults` from `langchain_community.tools.tavily_search` or `tavily-python`)
- **Environment Management**: `python-dotenv` for loading `OPENAI_API_KEY` and `TAVILY_API_KEY`
- **Prompt Engineering System**: `prompt_helper` package (https://github.com/krishnamohan-seelam/prompt_helper) for managing prompts
- **Logging**: Python `logging` module for structured logging
- **Testing**: `pytest` for unit and integration testing
- **Documentation**: `mkdocs` for project documentation


---

## 4. Detailed Specification & Component Breakdown

### 4.1 State Schema (`RouterState`)
Define a state TypedDict or Pydantic model containing:
- `query`: `str` — The input user query.
- `intent`: `Optional[Literal["search", "direct"]]` — Classified query intent.
- `search_results`: `Optional[Union[List[dict], str]]` — Web search payload extracted from Tavily.
- `response`: `Optional[str]` — Final generated response to the user.

### 4.2 Pydantic Intent Classifier Model
Define `IntentClassification(BaseModel)`:
- `intent`: `Literal["search", "direct"]`
- `reasoning`: `str` (Explanation of why web search is or is not required)

### 4.3 Node Definitions
1. **`intent_detection_node(state: RouterState) -> dict`**:
   - Analyzes `state["query"]`.
   - Calls LLM with structured output (`IntentClassification`).
   - System Prompt managed via `prompt_helper`.
   - Returns updated state dictionary `{"intent": classification.intent}`.

2. **`web_search_node(state: RouterState) -> dict`**:
   - Executes Tavily web search for `state["query"]`.
   - Formats search results into context string.
   - Generates an augmented answer using `ChatOpenAI`.
   - Returns `{"search_results": raw_results, "response": generated_answer}`.

3. **`direct_response_node(state: RouterState) -> dict`**:
   - Generates an answer directly from LLM internal knowledge for `state["query"]`.
   - System Prompt managed via `prompt_helper`.
   - Returns `{"response": generated_answer}`.

### 4.4 Routing Logic & Graph Assembly
- **Router Function `route_intent(state: RouterState) -> str`**:
  - Returns `"web_search_node"` if `state["intent"] == "search"`.
  - Returns `"direct_response_node"` if `state["intent"] == "direct"`.
- **Graph Compilation**:
  - Add nodes: `intent_detection`, `web_search`, `direct_response`.
  - Add edge: `START` -> `intent_detection`.
  - Add conditional edges: `intent_detection` -> `route_intent` -> `{"web_search_node": "web_search", "direct_response_node": "direct_response"}`.
  - Add edges: `web_search` -> `END`, `direct_response` -> `END`.
  - Compile `StateGraph`.

### 4.5 Orchestration CLI Loop
Create an interactive terminal interface:
- Loads `.env` environment variables.
- Prompts user input: `User > `.
- Checks termination keywords (`bye`, `stop`, `exit`, `quit`).
- Invokes compiled graph with input query and prints output state response and intent.

---

## 5. Scope & Guardrails

```
Scope: smart_query_router/
Stop condition: All nodes, router functions, and state transitions verified with unit tests (pytest); interactive CLI loop verified with live API calls or mock tests.
Checkpoint: Pause for code review after graph structure compilation before final delivery.
```

---

## 6. Output Contract & Deliverables

1. Modular Python codebase adhering to PEP 8 standards.
2. Complete test suite using `pytest` covering all nodes and routing conditions.
3. Updated project documentation (`README.md`, `DOCS/SMART_QUERY_ROUTER_ARCHITECTURE.md`, `design_docs/SMART_QUERY_ROUTER_PHASE_COMPLETION.md`).
4. Clean virtual environment configuration with explicit dependency management.
