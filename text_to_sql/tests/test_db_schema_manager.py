"""
Tests for the Database Schema Manager.
"""
import sqlite3
import pytest
from pathlib import Path
from text_to_sql.db_schema_manager import describe_database

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with a known schema."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
    cursor.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, user_id INTEGER, FOREIGN KEY(user_id) REFERENCES users(id));")
    
    conn.commit()
    conn.close()
    return str(db_path)

def test_describe_database(temp_db):
    """Test that describe_database returns the correct schema description."""
    description = describe_database(temp_db)
    
    assert "Table: users" in description
    assert "- id (INTEGER)" in description
    assert "- name (TEXT)" in description
    
    assert "Table: posts" in description
    assert "- id (INTEGER)" in description
    assert "- title (TEXT)" in description
    assert "- user_id (INTEGER)" in description
    
    assert "Foreign Keys:" in description
    assert "user_id → users.id" in description

def test_describe_database_no_tables(tmp_path):
    """Test describe_database with an empty database."""
    db_path = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(db_path)
    conn.close()
    
    description = describe_database(str(db_path))
    assert description == ""
