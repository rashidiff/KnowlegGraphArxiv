import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.local_sqlite import LocalSQLiteDB
from backend.main import app

@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a isolated temporary SQLite database path."""
    db_file = str(tmp_path / "test_navigator.db")
    return db_file

@pytest.fixture
def test_db(temp_db_path):
    """Provides an initialized LocalSQLiteDB instance on a temp database."""
    db = LocalSQLiteDB(db_path=temp_db_path)
    return db

@pytest.fixture
def api_client():
    """Provides a FastAPI TestClient instance."""
    return TestClient(app)
