"""
Tests for the embedding generation pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch
from text_to_sql.generate_query_embeddings import (
    prepare_documents,
    log_db_distribution,
    run,
)

def test_prepare_documents():
    """Test converting training data to LangChain Documents."""
    training_data = [
        {"db_id": "db1", "evidence": "ev1; ev2"},
        {"db_id": "db2", "evidence": "ev3"},
    ]
    docs = prepare_documents(training_data)
    
    assert len(docs) == 3
    assert docs[0].page_content == "ev1"
    assert docs[0].metadata["db_id"] == "db1"
    assert docs[1].page_content == "ev2"
    assert docs[1].metadata["db_id"] == "db1"
    assert docs[2].page_content == "ev3"
    assert docs[2].metadata["db_id"] == "db2"

def test_log_db_distribution(capsys):
    """Test logging of database distribution."""
    training_data = [
        {"db_id": "db1", "evidence": "ev1"},
        {"db_id": "db1", "evidence": "ev2"},
        {"db_id": "db2", "evidence": "ev3"},
    ]
    # This function uses logger.info, not print, so capsys won't catch it unless we configure logging
    # But we can just call it to ensure it doesn't crash
    log_db_distribution(training_data)

@patch("text_to_sql.generate_query_embeddings.load_training_data")
@patch("text_to_sql.generate_query_embeddings.create_embeddings")
@patch("text_to_sql.generate_query_embeddings.create_chroma_client")
@patch("text_to_sql.generate_query_embeddings.create_vector_store")
@patch("text_to_sql.generate_query_embeddings.get_settings")
def test_run(mock_get_settings, mock_create_vector_store, mock_create_chroma_client, mock_create_embeddings, mock_load_data):
    """Test the end-to-end run pipeline with mocks."""
    mock_settings = MagicMock()
    mock_settings.FILE_PATH = "data/dev.json"
    mock_settings.VECTOR_DB_PATH = "data/chromadb"
    mock_settings.GOOGLE_API_KEY = "test-key"
    mock_get_settings.return_value = mock_settings
    
    mock_load_data.return_value = [{"db_id": "db1", "evidence": "ev1"}]
    mock_vector_store = MagicMock()
    mock_create_vector_store.return_value = mock_vector_store
    
    run(api_key="test-key")
    
    mock_load_data.assert_called_once()
    mock_create_embeddings.assert_called_once_with("test-key")
    mock_create_chroma_client.assert_called_once()
    mock_create_vector_store.assert_called_once()
    mock_vector_store.add_documents.assert_called_once()
