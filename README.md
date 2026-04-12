# Building Agentic AI - Project Workspace

This repository contains various modules and exercises related to building Agentic AI systems, specifically focusing on LangChain and LangGraph.

## Project Structure
- `langchain_agents/`: A library for LangChain agent development, featuring automated code review and generation.
- `text_to_sql/`: A specialized RAG system for translating natural language to SQL queries.
- `Notebooks/`: Experimental and tutorial notebooks.

## Key Features
- **Code Generation & Optimization**: Quickly bootstrap LangChain agents and automatically review them using the internal **Dev Agent**.
- **Agent-Ready Tools**: All skills are exposed as standard LangChain `@tool` objects, ready to be plugged into any agentic workflow.
- **Environment Management**: Robust loading and validation of environment variables.

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
- [Architecture](DOCS/ARCHITECTURE.md)
- [Text-to-SQL README](text_to_sql/README.md)
