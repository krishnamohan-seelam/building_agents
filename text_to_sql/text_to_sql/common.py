"""
Common utilities, models, and helpers for the Text-to-SQL system.

Contains all common functions and classes used in the project.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Annotated, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .settings import get_settings, Settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# State & Models
# ═══════════════════════════════════════════════════════════════════════════

class RetrievedDocument(BaseModel):
    """A single document retrieved from the vector store."""
    content: str = Field(description="Document text")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    score: float = Field(description="Similarity score (higher = better)")


class SQLOutput(BaseModel):
    """Structured output from the LLM for SQL generation."""
    reasoning: str = Field(description="Step-by-step reasoning for the SQL query.")
    sql_query: str = Field(description="The executable SQL query.")


class RAGState(BaseModel):
    """LangGraph state flowing through the workflow."""
    # --- inputs ---
    query: str = ""
    db_id: str = ""
    db_path: str = ""
    model_name: str = "gemini-flash-latest"
    k: int = 5
    # --- intermediate ---
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    context: str = ""
    # --- outputs ---
    sql_query: str = ""
    reasoning: str = ""
    sql_result: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# Shared Functions (Embeddings, Vector Store, Loading)
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data(file_path: str) -> Optional[list]:
    """Load training data JSON file (e.g., dev.json)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"✅ Loaded {len(data)} training examples from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {file_path}: {e}")
        return None

def create_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Create Gemini embedding function."""
    logger.info("✨ Initializing Gemini Embeddings (gemini-embedding-001)...")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )

def create_chroma_client(vector_db_path: str) -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    logger.info(f"📦 Opening ChromaDB at {vector_db_path}")
    return chromadb.PersistentClient(
        path=vector_db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

def create_vector_store(
    chroma_client: chromadb.ClientAPI,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = "documents",
    *,
    reset: bool = True,
) -> Chroma:
    """
    Create (or re-create) a Chroma vector store backed by the given client.
    """
    if reset:
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"🗑️  Deleted existing '{collection_name}' collection.")
        except Exception:
            pass  # collection didn't exist — that's fine

    logger.info(f"📚 Creating Chroma vector store '{collection_name}'...")
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

def open_vector_store(
    vector_db_path: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "documents",
    settings: Optional[Settings] = None,
) -> Tuple[chromadb.ClientAPI, Chroma]:
    """
    Open an existing ChromaDB vector store.
    """
    cfg = settings or get_settings()
    vector_db_path = vector_db_path or str(cfg.VECTOR_DB_PATH)
    api_key = api_key or cfg.GOOGLE_API_KEY

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    embeddings = create_embeddings(api_key)
    client = create_chroma_client(vector_db_path)
    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return client, store
