"""
Convert text to SQL using LangGraph workflow.

Retrieves relevant evidence from ChromaDB, reads the schema,
generates SQL via an LLM, and executes it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from functools import partial
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

from .settings import get_settings, Settings
from .common import RAGState, RetrievedDocument, SQLOutput, open_vector_store
from .db_schema_manager import describe_database

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Workflow nodes
# ═══════════════════════════════════════════════════════════════════════════

def begin_conversation(state: RAGState) -> Dict[str, Any]:
    """Set the system prompt."""
    return {
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant that generates SQL queries "
                    "to answer the user's question."
                )
            )
        ]
    }

def process_query(state: RAGState) -> Dict[str, Any]:
    """Add the user question to the message history."""
    logger.info(f"🔍 Processing query: {state.query}")
    messages = list(state.messages)
    if not any(isinstance(m, HumanMessage) and m.content == state.query for m in messages):
        messages.append(HumanMessage(content=state.query))
    return {"messages": messages}

def retrieve_documents(state: RAGState, vector_store: Chroma) -> Dict[str, Any]:
    """Retrieve relevant evidence from the ChromaDB vector store."""
    logger.info(
        f"📚 Retrieving top {state.k} documents for query: {state.query} "
        f"(db_id: {state.db_id})"
    )
    try:
        results = vector_store.similarity_search_with_score(
            query=state.query,
            k=state.k,
            filter={"db_id": state.db_id} if state.db_id else None,
        )

        docs = []
        context_parts: list[str] = []
        for i, (doc, score) in enumerate(results, 1):
            rd = RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=1 - score,  # distance → similarity
            )
            docs.append(rd)
            context_parts.append(f"Document {i}:\n{rd.content}")
            if rd.metadata:
                context_parts.append(f"Metadata: {rd.metadata}")
            context_parts.append("")

            logger.info(f"   {i}. Score: {rd.score:.4f} | {rd.content[:100]}...")

        logger.info(f"✅ Retrieved {len(docs)} documents")
        return {"retrieved_documents": docs, "context": "\n".join(context_parts)}

    except Exception as e:
        logger.error(f"❌ Error retrieving documents: {e}")
        return {
            "retrieved_documents": [],
            "context": "No documents could be retrieved due to an error.",
        }

def generate_sql_query(state: RAGState, api_key: str) -> Dict[str, Any]:
    """Use the LLM to generate a SQL query based on the retrieved context."""
    logger.info(f"🤖 Generating SQL with model: {state.model_name}")

    llm = ChatGoogleGenerativeAI(
        model=state.model_name,
        temperature=0,
        google_api_key=api_key,
    ).with_structured_output(SQLOutput)

    schema_desc = describe_database(state.db_path)
    new_msg = HumanMessage(
        content=(
            f"Question: {state.query}\n"
            f"Context: {state.context}\n"
            f"Database Description: {schema_desc}"
        )
    )
    messages = [*state.messages, new_msg]

    try:
        response: SQLOutput = llm.invoke(messages)
        return {
            "messages": [
                *messages,
                AIMessage(
                    content=f"Reasoning: {response.reasoning}\nSQL Query: {response.sql_query}"
                ),
            ],
            "sql_query": response.sql_query,
            "reasoning": response.reasoning,
        }
    except Exception as e:
        error_msg = f"Error generating SQL: {e}"
        logger.error(f"❌ {error_msg}")
        return {
            "messages": [*state.messages, AIMessage(content=error_msg)],
            "final_answer": error_msg,
        }

def run_sql_query(state: RAGState) -> Dict[str, Any]:
    """Execute the generated SQL against the target database."""
    logger.info(f"⚡ Executing SQL on '{state.db_id}': {state.sql_query}")

    if not os.path.exists(state.db_path):
        error = f"Database not found at {state.db_path}"
        logger.error(error)
        return {"sql_result": {"error": error}, "final_answer": f"Error: {error}"}

    try:
        conn = sqlite3.connect(state.db_path)
        cursor = conn.cursor()
        cursor.execute(state.sql_query)
        columns = (
            [d[0] for d in cursor.description] if cursor.description else []
        )
        rows = cursor.fetchall()
        conn.close()

        sql_result = {
            "success": True,
            "columns": columns,
            "data": rows,
            "row_count": len(rows),
            "sql_query": state.sql_query,
        }
        logger.info(f"✅ Query returned {len(rows)} row(s)")

        if rows:
            lines = [f"Query executed successfully! Found {len(rows)} row(s):", ""]
            if columns:
                lines.append(" | ".join(columns))
                lines.append("-" * len(lines[-1]))
            for row in rows[:10]:
                lines.append(
                    " | ".join(str(c) if c is not None else "NULL" for c in row)
                )
            if len(rows) > 10:
                lines.append("")
                lines.append(f"... and {len(rows) - 10} more row(s)")
            answer = "\n".join(lines)
        else:
            answer = "Query executed successfully but returned no results."

        return {"sql_result": sql_result, "final_answer": answer}

    except Exception as e:
        error_msg = f"Error executing SQL: {e}"
        logger.error(f"❌ {error_msg}")
        return {"sql_result": {"error": error_msg}, "final_answer": error_msg}

# ═══════════════════════════════════════════════════════════════════════════
# Workflow builder
# ═══════════════════════════════════════════════════════════════════════════

def build_workflow(vector_store: Chroma, api_key: str) -> StateGraph:
    """Build and return an uncompiled LangGraph StateGraph."""
    wf = StateGraph(RAGState)

    wf.add_node("begin_conversation", begin_conversation)
    wf.add_node("process_query", process_query)
    wf.add_node(
        "retrieve_documents",
        partial(retrieve_documents, vector_store=vector_store),
    )
    wf.add_node(
        "generate_sql_query",
        partial(generate_sql_query, api_key=api_key),
    )
    wf.add_node("run_sql_query", run_sql_query)

    wf.set_entry_point("begin_conversation")
    wf.add_edge("begin_conversation", "process_query")
    wf.add_edge("process_query", "retrieve_documents")
    wf.add_edge("retrieve_documents", "generate_sql_query")
    wf.add_edge("generate_sql_query", "run_sql_query")
    wf.add_edge("run_sql_query", END)

    return wf

def create_and_run(
    query: str,
    db_id: str,
    *,
    vector_db_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-flash-latest",
    k: int = 5,
    collection_name: str = "documents",
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """One-shot convenience function: open vector store → build workflow → run."""
    cfg = settings or get_settings()
    vector_db_path = vector_db_path or str(cfg.VECTOR_DB_PATH)
    api_key = api_key or cfg.GOOGLE_API_KEY

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    if not cfg.DATABASES_DIR:
        raise ValueError("DATABASES_DIR must be configured in settings.")
    
    db_path = str(cfg.DATABASES_DIR / db_id / f"{db_id}.sqlite")

    logger.info(f"🔧 Settings profile: {cfg.ENV}")

    client, store = open_vector_store(vector_db_path, api_key, collection_name)
    try:
        wf = build_workflow(store, api_key)
        app = wf.compile()
        result = app.invoke({
            "query": query,
            "db_id": db_id,
            "db_path": db_path,
            "model_name": model_name,
            "k": k,
        })
        return result
    finally:
        client.close()

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg = get_settings()

    parser = argparse.ArgumentParser(description="Text-to-SQL RAG Workflow")
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument("--db-id", required=True, help="Database ID (e.g., formula_1)")
    parser.add_argument(
        "--vector-db-path",
        default=None,
        help=f"Path to ChromaDB directory (default from settings: {cfg.VECTOR_DB_PATH})",
    )
    parser.add_argument(
        "--model", default="gemini-flash-latest", help="Gemini model name"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    args = parser.parse_args()

    if not cfg.DATABASES_DIR:
        raise ValueError("DATABASES_DIR must be configured in settings.")

    result = create_and_run(
        query=args.query,
        db_id=args.db_id,
        vector_db_path=args.vector_db_path,
        model_name=args.model,
        k=args.k,
        settings=cfg,
    )

    print("\n" + "=" * 60)
    print("📋 RESULTS")
    print("=" * 60)
    print(f"Query:     {args.query}")
    print(f"SQL:       {result.get('sql_query', 'N/A')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print()
    print(result.get("final_answer", "No answer"))
