# Building Agentic AI - Project Workspace

This repository contains various modules and exercises related to building Agentic AI systems, specifically focusing on LangChain and LangGraph.

## Project Structure
- `smart_query_router/`: A production-ready LangGraph Smart Query Router using OpenAI structured outputs, Tavily web search, and `prompt_helper`.
- `langchain_agents/`: A library for LangChain agent development, featuring automated code review and generation.
- `text_to_sql/`: A specialized RAG system for translating natural language to SQL queries.
- `learning_langchain/`: Foundational LangChain scripts, experiments, and references.
- `Notebooks/`: Experimental and tutorial notebooks.
- `tests/`: Automated test suite for project modules.
- `prompt_helper/`: A reusable LangChain prompt helper module including `PromptBuilder` and dynamic strategies like PTCF and Few-Shot.
- `design_docs/`: Project design and phase completion tracking documents.

## Key Features
- **Smart Query Router**: Production-ready stateful LangGraph router featuring top-level node architecture, standard LangGraph runtime configuration (`RunnableConfig`), intent classification (`"search"` vs `"direct"`), conditional edge routing, Tavily web search enrichment, and interactive Rich CLI.
- **Prompt Helper Module**: Simplifies assembling, validating, and formatting system/human/AI messages from prompt templates.
- **Pluggable Prompt Strategies**: Includes built-in strategies like `SimpleStrategy`, `FewShotStrategy`, and `PTCFStrategy` (Persona-Task-Context-Format) for structured prompt designs.
- **Fluent Prompt Assembly**: `PromptBuilder` supports chained rendering of message lists, history splicing (handling system prompt offsets automatically), and exporting directly to native LangChain `ChatPromptTemplate`s.
- **Code Generation & Optimization**: Quickly bootstrap LangChain agents and automatically review them using the internal **Dev Agent**.
- **Agent-Ready Tools**: All skills are exposed as standard LangChain `@tool` objects, ready to be plugged into any agentic workflow.
- **Environment Management**: Robust loading and validation of environment variables.
- **Planner-Executor Workflow**: Stateful LangGraph-based sequential planning agent supporting dynamic LLM model selection (Groq, Google Gemini, OpenAI), tool execution protection (5-turn execution limits), and synthesized answers.
- **Query Decomposer (Self-Ask with Search)**: Iterative multi-hop research agent integrated with `prompt-helper` (`PromptBuilder`). Dynamically breaks complex questions into sub-questions, resolves each via tools, accumulates a scratchpad, and synthesizes a grounded final answer — no upfront plan required.
- **Meta-Prompt Skill (`.agents/skills/meta-prompt`)**: Engineered prompt generator supporting software lifecycle classification, pattern mapping (RTF, PTCF, Spec Block, etc.), guardrails, and saving output directly to a user-specified directory.
- **Safe Calculation & Weather Retrieval**: Calculator tool containing CPU-exhaustion checks (blocking `**`) and weather tool with parameter URL-encoding.

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install dependencies and the project in editable mode
pip install -e .
```

### Usage (Smart Query Router CLI & Configurable Graph)
```bash
# Launch interactive REPL CLI loop
.venv\Scripts\python -m smart_query_router.cli

# Programmatic usage with graph configuration:
from smart_query_router import build_router_graph

graph = build_router_graph()
# Pass runtime configuration dynamically via RunnableConfig
result = graph.invoke(
    {"query": "What is the stock price of Tesla today?"},
    config={"configurable": {"llm": custom_llm, "search_tool": custom_search}}
)
print(result["intent"], result["response"])
```


### Usage (Prompt Helper)
```python
from prompt_helper import PromptBuilder, PTCFStrategy

# 1. Using PromptBuilder for fluent assembly
messages = (
    PromptBuilder("query_decomposer")
    .system("You are an expert at decomposing {domain} queries into sub-tasks.")
    .human("Decompose this query: {query}")
    .build(domain="fintech", query="Reconcile Q3 transactions")
)

# 2. Using PTCF Strategy for structured templates
ptcf = PTCFStrategy(
    persona="Senior fintech reconciliation engineer",
    task="Identify discrepancies between two transaction ledgers",
    context="Ledger A and Ledger B are provided as JSON below: {ledger_a} {ledger_b}",
    format_instructions="Return a JSON list of discrepancy objects with fields: id, type, delta",
)
system_message = ptcf.render(ledger_a="[...]", ledger_b="[...]")[0]
```

### Usage (Planner-Executor Agent)
```python
from learning_langchain.planner_executor_agent import run_agent

# Run the agent to solve a complex research goal
final_answer = run_agent(
    "Compare Python and Rust: what are each language's main strengths "
    "and which should a backend developer learn in 2025?"
)
print(final_answer)
```

### Usage (Query Decomposer Agent)
```python
from learning_langchain.query_decomposer.query_decomposer_agent import run_agent

# Multi-hop research: agent breaks the question into sub-questions automatically
answer = run_agent(
    "Who invented Python, and what university did they study at?"
)
print(answer)
```

### Usage (LangChain Agents Skills)
```python
from langchain_agents.skills import review_code_tool, generate_code_tool

# Use as standalone tools or skills
code = generate_code_tool.invoke({"task": "Create a weather agent"})
review = review_code_tool.invoke({"code": code})
print(review)
```

### Usage (Developer Agent)
```python
from langchain_agents.agent import get_dev_agent

agent = get_dev_agent()
# This meta-agent can use its tools to design and review other agents!
```

## Testing
Run the automated test suite to verify project logic:
```bash
# Run Smart Query Router test suite with 99% coverage
.venv\Scripts\pytest --cov=smart_query_router smart_query_router/tests

# Run prompt_helper tests with coverage
.venv\Scripts\pytest --cov=prompt_helper tests/prompt_helper

# Run langchain_agents tests
.venv\Scripts\python -m pytest tests/langchain_agents

# Run learning_langchain tests (including Financial Analyst Workflow tests)
$env:PYTHONPATH=".;learning_langchain"
.venv\Scripts\python -m pytest tests/learning_langchain
```

## Documentation
- [Smart Query Router Architecture](DOCS/SMART_QUERY_ROUTER_ARCHITECTURE.md)
- [Architecture Details](DOCS/ARCHITECTURE.md)
- [Text-to-SQL README](text_to_sql/README.md)


