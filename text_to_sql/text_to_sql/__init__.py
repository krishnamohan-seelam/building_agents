"""
Text-to-SQL RAG System — simplified.

Modules:
    settings                  : Multi-source config (YAML profiles + .env + env vars)
    generate_query_embeddings : Load training data → embed → store in ChromaDB
    text_to_sql               : Retrieve evidence → generate SQL via LLM → execute
    common                    : Shared utilities, state models, vector DB helpers
    db_schema_manager         : Extracts database schema for LLM context
"""

from .settings import get_settings, Settings
from .generate_query_embeddings import run as generate_query_embeddings
from .text_to_sql import create_and_run, build_workflow

__all__ = [
    "get_settings",
    "Settings",
    "generate_query_embeddings",
    "create_and_run",
    "build_workflow",
]
