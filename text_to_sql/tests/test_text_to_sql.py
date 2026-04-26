"""
Tests for the Text-to-SQL LangGraph workflow.
"""
import pytest
import sqlite3
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, partial
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from text_to_sql.text_to_sql import (
    begin_conversation,
    process_query,
    retrieve_documents,
    generate_sql_query,
    run_sql_query,
    build_workflow,
    create_and_run,
)
from text_to_sql.common import RAGState, RetrievedDocument, SQLOutput

def test_begin_conversation():
    """Test the begin_conversation node."""
    state = RAGState()
    result = begin_conversation(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)

def test_process_query():
    """Test the process_query node."""
    state = RAGState(query="What is the total number of users?")
    result = process_query(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "What is the total number of users?"

def test_retrieve_documents():
    """Test the retrieve_documents node."""
    mock_vector_store = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "test content"
    mock_doc.metadata = {"db_id": "db1"}
    mock_vector_store.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    
    state = RAGState(query="test query", db_id="db1", k=1)
    result = retrieve_documents(state, mock_vector_store)
    
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0].content == "test content"
    assert "Document 1:" in result["context"]

@patch("text_to_sql.text_to_sql.ChatGoogleGenerativeAI")
@patch("text_to_sql.text_to_sql.describe_database")
def test_generate_sql_query(mock_describe, mock_llm_class):
    """Test the generate_sql_query node."""
    mock_describe.return_value = "Schema description"
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = SQLOutput(reasoning="reasoning", sql_query="SELECT * FROM users")
    mock_llm_class.return_value.with_structured_output.return_value = mock_llm
    
    state = RAGState(query="How many users?", context="Some context", model_name="test-model")
    result = generate_sql_query(state, "test-api-key")
    
    assert result["sql_query"] == "SELECT * FROM users"
    assert result["reasoning"] == "reasoning"
    assert any("SELECT * FROM users" in m.content for m in result["messages"] if isinstance(m, AIMessage))

def test_run_sql_query(tmp_path):
    """Test the run_sql_query node."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER);")
    conn.execute("INSERT INTO users VALUES (1), (2);")
    conn.commit()
    conn.close()
    
    state = RAGState(sql_query="SELECT COUNT(*) FROM users", db_path=str(db_path), db_id="test_db")
    result = run_sql_query(state)
    
    assert result["sql_result"]["success"] is True
    assert result["sql_result"]["row_count"] == 1
    assert result["sql_result"]["data"][0][0] == 2
    assert "Found 1 row(s)" in result["final_answer"]

def test_run_sql_query_error(tmp_path):
    """Test the run_sql_query node with a faulty query."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER);")
    conn.commit()
    conn.close()
    
    state = RAGState(sql_query="SELECT * FROM nonexistent", db_path=str(db_path), db_id="test_db")
    result = run_sql_query(state)
    
    assert "error" in result["sql_result"]
    assert "no such table: nonexistent" in result["sql_result"]["error"]

def test_build_workflow():
    """Test that the workflow graph is built correctly."""
    mock_vector_store = MagicMock()
    wf = build_workflow(mock_vector_store, "test-api-key")
    assert wf is not None
    # We can't easily inspect the internal nodes of a StateGraph without compiling, 
    # but we can check if it compiles
    app = wf.compile()
    assert app is not None

@patch("text_to_sql.text_to_sql.open_vector_store")
@patch("text_to_sql.text_to_sql.build_workflow")
@patch("text_to_sql.text_to_sql.get_settings")
def test_create_and_run(mock_get_settings, mock_build_workflow, mock_open_store):
    """Test the create_and_run convenience function."""
    mock_settings = MagicMock()
    mock_settings.VECTOR_DB_PATH = "data/chromadb"
    mock_settings.GOOGLE_API_KEY = "test-key"
    mock_settings.DATABASES_DIR = Path("data/dbs")
    mock_get_settings.return_value = mock_settings
    
    mock_client = MagicMock()
    mock_store = MagicMock()
    mock_open_store.return_value = (mock_client, mock_store)
    
    mock_app = MagicMock()
    mock_app.invoke.return_value = {"result": "success"}
    mock_wf = MagicMock()
    mock_wf.compile.return_value = mock_app
    mock_build_workflow.return_value = mock_wf
    
    result = create_and_run(
        query="test query",
        db_id="test_db",
        api_key="test-key"
    )
    
    assert result == {"result": "success"}
    mock_open_store.assert_called_once()
    mock_build_workflow.assert_called_once_with(mock_store, "test-key")
    mock_app.invoke.assert_called_once()
    mock_client.close.assert_called_once()
