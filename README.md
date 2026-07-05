# Building Agentic AI - Project Workspace

This repository contains various modules and exercises related to building Agentic AI systems, specifically focusing on LangChain and LangGraph.

## Project Structure
- `langchain_agents/`: A library for LangChain agent development, featuring automated code review and generation.
- `text_to_sql/`: A specialized RAG system for translating natural language to SQL queries.
- `learning_langchain/`: Foundational LangChain scripts, experiments, and references.
- `Notebooks/`: Experimental and tutorial notebooks.
- `tests/`: Automated test suite for project modules.
- `design_docs/`: Project design and phase completion tracking documents.

## Key Features
- **Code Generation & Optimization**: Quickly bootstrap LangChain agents and automatically review them using the internal **Dev Agent**.
- **Agent-Ready Tools**: All skills are exposed as standard LangChain `@tool` objects, ready to be plugged into any agentic workflow.
- **Environment Management**: Robust loading and validation of environment variables.
- **Planner-Executor Workflow**: Stateful LangGraph-based sequential planning agent supporting dynamic LLM model selection (Groq, Google Gemini, OpenAI), tool execution protection (5-turn execution limits), and synthesized answers.
- **Query Decomposer (Self-Ask with Search)**: Iterative multi-hop research agent that dynamically breaks complex questions into sub-questions, resolves each via tools, accumulates a scratchpad, and synthesizes a grounded final answer — no upfront plan required.
- **Safe Calculation & Weather Retrieval**: Calculator tool containing CPU-exhaustion checks (blocking `**`) and weather tool with parameter URL-encoding.

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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

## Documentation
- [Architecture Details](DOCS/ARCHITECTURE.md)
- [Text-to-SQL README](text_to_sql/README.md)
