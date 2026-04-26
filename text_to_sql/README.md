# Text-to-SQL RAG System - Setup & Instructions

A lightweight, fully typed, and modular Text-to-SQL project using LangGraph, Gemini, and ChromaDB. It uses a Retrieval-Augmented Generation (RAG) approach to pick relevant SQL examples and constructs optimized queries from natural language.

Most of the code is used from the chapter 2 of the book "Building Agentic AI: Workflows, Fine-Tuning, Optimization, and Deployment" - Sinan Ozdemir.  
Example from book uses Open AI , I have used gemini-embedding-001 and gemini-flash-latest. 
I have also included COLAB notebooks used for prototyping

**GEMINI API KEY is required to run this code**
---

## 1. Installation

### Production Dependencies
```bash
pip install -r requirements.txt
```

### Development/Testing Dependencies
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

---

## 2. Configuration & Environments

The system uses a robust configuration loader (`settings.py`) that supports multiple environments (**dev**, **prod**, **colab**) via profile selection.

### Configuration Hierarchy
Settings are loaded in the following order (lower items override higher ones):
1. **Defaults:** `text_to_sql/settings/defaults.yaml`
2. **Profile file:** `text_to_sql/settings/{ENV}.yaml` (e.g., `dev.yaml`)
3. **`.env` file:** Environment variables loaded via `python-dotenv`
4. **OS Environment Variables:** Highest priority

### Environment Selection
Set your environment by:
1. Setting the `TEXT_TO_SQL_ENV` environment variable (e.g., `export TEXT_TO_SQL_ENV=prod`).
2. **Auto-detection**: If running in Google Colab (detects `COLAB_GPU`), it automatically uses the `colab` profile.
3. Defaulting to `dev`.

### Key Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `TEXT_TO_SQL_ENV` | Profile to load (dev/prod/colab) | `prod` |
| `FILE_PATH` | Path to training JSON data | `data/dev.json` |
| `VECTOR_DB_PATH` | Path to ChromaDB persistent directory | `data/chromadb` |
| `DATABASES_DIR` | Base directory holding multiple SQLite DBs | `data/dbs/dev_databases` |
| `GOOGLE_API_KEY` | Your Gemini API Key | `AIzaSy...` |

---

## 3. Directory Structure

The system was refactored from a rigid DDD layout into a highly focused, modular application:

```text
text_to_sql/
├── text_to_sql/
│   ├── __init__.py                  # Exposes main functions & settings
│   ├── settings.py                  # Multi-source config loader
│   ├── settings/                    # YAML profile configs (dev, prod, colab, defaults)
│   ├── common.py                    # Shared models (RAGState), DB, & Vector helpers
│   ├── db_schema_manager.py         # SQLite schema extractor for LLM context
│   ├── generate_query_embeddings.py # Step 1: Embeds dev.json evidence -> ChromaDB
│   └── text_to_sql.py               # Step 2: LangGraph RAG workflow orchestrator
├── data/                            # Contains dev.json, SQLite DBs, and ChromaDB
├── requirements.txt                 
└── SETUP.md                         # This file
```

---

## 4. Usage Instructions

### Step 1: Generate & Store Embeddings 
This script reads your training data (e.g., `dev.json`), extracts SQL structural examples (evidence), generates Gemini embeddings, and stores them in your ChromaDB. 

*Only needs to be run once (or whenever your training data changes).*

```bash
# In the project root:
python -m text_to_sql.generate_query_embeddings
```

*(By default, this resolves all paths and API keys automatically using the `dev` profile combined with your `.env` file).*

### Step 2: Run the Text-to-SQL Workflow
Use the LangGraph workflow to translate a natural language question into an executable SQL query. The system will retrieve similar examples from ChromaDB and introspect your target SQLite database schema before generation.

```bash
python -m text_to_sql.text_to_sql \
    --query "How many drivers are from Germany?" \
    --db-id formula_1
```
*(The system automatically locates the `.sqlite` file using `<DATABASES_DIR>/<db-id>/<db-id>.sqlite` from settings).*
```bash
============================================================
📋 RESULTS
============================================================
Query:     How many drivers are from Germany?
SQL:       SELECT count(*) FROM drivers WHERE nationality = 'German'
Reasoning: The user wants to count the number of drivers whose nationality is German. I will use the 'drivers' table and filter by the 'nationality' column with the value 'German'.

Query executed successfully! Found 1 row(s):

count(*)
--------
49
```

### Using as a Python Library
You can directly import the components into your own scripts or notebooks:

```python
from text_to_sql import get_settings, generate_query_embeddings, create_and_run

# 1. Check settings
settings = get_settings()
print(f"Profile: {settings.ENV}, DB Dir: {settings.DATABASES_DIR}")

# 2. Rebuild embeddings vector store (reads paths from settings)
generate_query_embeddings()

# 3. Ask a question!
result = create_and_run(
    query="What is the highest eligible free rate for K-12 students?",
    db_id="california_schools",
    model_name="gemini-2.0-flash"
)

print(result.get("sql_query"))
print(result.get("final_answer"))
```

## 6. Testing

The project includes a comprehensive test suite with 90%+ coverage, including unit tests and integration tests using mocks.

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=text_to_sql text_to_sql/tests/
```

The tests cover:
- **Settings:** Profile loading, environment variable precedence, and path resolution.
- **Database Schema Manager:** SQLite schema extraction and foreign key detection.
- **Common Utilities:** Pydantic models and factory functions.
- **Embedding Generation:** Data preparation and ingestion pipeline.
- **Workflow:** LangGraph nodes and overall workflow orchestration (mocked LLM).
