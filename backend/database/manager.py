import os
import sys
from dotenv import load_dotenv
from backend.database.base import BaseDatabase
from backend.database.local_sqlite import LocalSQLiteDB
from backend.database.postgres_neo4j import PostgresNeo4jDB

# Load env
load_dotenv()

_db_instance = None

def get_db() -> BaseDatabase:
    """
    Database Factory. Attempts to connect to PostgreSQL + Neo4j first.
    If connection fails, falls back to SQLite + NetworkX. Caches the connection
    globally to prevent repeated timeout delays.
    """
    global _db_instance
    if _db_instance is not None:
        return _db_instance

    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_db = os.getenv("POSTGRES_DB", "agent_research_navigator")

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")

    print("[DB Connection Manager] Attempting to connect to PostgreSQL + Neo4j...")
    try:
        pg_config = {
            "host": pg_host,
            "port": pg_port,
            "user": pg_user,
            "password": pg_pass,
            "dbname": pg_db
        }
        neo4j_config = {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_pass
        }
        
        # Test connections by instantiating PostgresNeo4jDB
        _db_instance = PostgresNeo4jDB(postgres_config=pg_config, neo4j_config=neo4j_config)
        print("[DB Connection Manager] SUCCESS: Connected to PostgreSQL + Neo4j.")
        return _db_instance
        
    except Exception as e:
        print(f"[DB Connection Manager] WARNING: Failed to connect to Docker DBs (Postgres/Neo4j).", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        print("[DB Connection Manager] FALLBACK: Initializing Local SQLite + NetworkX DB engine...", file=sys.stderr)
        
        # Fallback to local SQLite
        db_path = "data/research_navigator.db"
        _db_instance = LocalSQLiteDB(db_path=db_path)
        return _db_instance
