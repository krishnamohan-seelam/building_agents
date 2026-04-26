"""
Tests for common utilities and models.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from text_to_sql.common import (
    load_training_data,
    create_embeddings,
    create_chroma_client,
    create_vector_store,
    RetrievedDocument,
    SQLOutput,
    RAGState,
)

def test_load_training_data(tmp_path):
    """Test loading training data from JSON."""
    data = [{"db_id": "test", "evidence": "test evidence"}]
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    loaded = load_training_data(str(file_path))
    assert loaded == data

def test_load_training_data_not_found():
    """Test loading non-existent training data."""
    assert load_training_data("nonexistent.json") is None

def test_retrieved_document_model():
    """Test RetrievedDocument Pydantic model."""
    doc = RetrievedDocument(content="test", metadata={"db_id": "test"}, score=0.9)
    assert doc.content == "test"
    assert doc.metadata["db_id"] == "test"
    assert doc.score == 0.9

def test_sql_output_model():
    """Test SQLOutput Pydantic model."""
    output = SQLOutput(reasoning="reason", sql_query="SELECT *")
    assert output.reasoning == "reason"
    assert output.sql_query == "SELECT *"

@patch("text_to_sql.common.GoogleGenerativeAIEmbeddings")
def test_create_embeddings(mock_embeddings):
    """Test embedding creation."""
    create_embeddings("test-api-key")
    mock_embeddings.assert_called_once_with(
        model="gemini-embedding-001",
        google_api_key="test-api-key",
    )

@patch("text_to_sql.common.chromadb.PersistentClient")
def test_create_chroma_client(mock_client):
    """Test Chroma client creation."""
    create_chroma_client("/test/path")
    mock_client.assert_called_once()
    args, kwargs = mock_client.call_args
    assert kwargs["path"] == "/test/path"

@patch("text_to_sql.common.Chroma")
def test_create_vector_store(mock_chroma):
    """Test vector store creation."""
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    
    create_vector_store(mock_client, mock_embeddings, "test_collection", reset=True)
    
    mock_client.delete_collection.assert_called_once_with("test_collection")
    mock_chroma.assert_called_once()
    args, kwargs = mock_chroma.call_args
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["embedding_function"] == mock_embeddings
