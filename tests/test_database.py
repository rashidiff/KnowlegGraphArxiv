import pytest
import numpy as np
from backend.database.local_sqlite import LocalSQLiteDB

def test_db_initialization(temp_db_path):
    import os
    db = LocalSQLiteDB(db_path=temp_db_path)
    assert os.path.exists(temp_db_path)
    assert db._graph is not None

def test_insert_and_get_paper(test_db):
    embedding = [0.1] * 384
    paper_data = {
        "id": "paper_1",
        "title": "Autonomous LLM Agents: A Survey",
        "abstract": "This paper surveys LLM-based autonomous agent architectures.",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2024,
        "venue": "NeurIPS",
        "citation_count": 42,
        "venue_quality": 0.9,
        "embedding": embedding,
        "topics": ["LLM Agents", "Multi-Agent Systems"],
        "entities": [{"type": "concept", "value": "Agent Architectures"}]
    }
    
    test_db.insert_papers([paper_data])
    
    paper = test_db.get_paper("paper_1")
    assert paper is not None
    assert paper["title"] == "Autonomous LLM Agents: A Survey"
    assert paper["year"] == 2024
    assert paper["citation_count"] == 42
    assert "LLM Agents" in paper["topics"]

def test_vector_search_papers(test_db):
    emb1 = [1.0] + [0.0] * 383
    emb2 = [0.0, 1.0] + [0.0] * 382

    p1 = {
        "id": "p1",
        "title": "Deep Learning for Recommender Systems",
        "abstract": "Recommendation algorithms using deep networks.",
        "authors": ["Author A"],
        "year": 2023,
        "citation_count": 10,
        "embedding": emb1,
        "topics": ["Recommender Systems"]
    }
    p2 = {
        "id": "p2",
        "title": "LLM Agents in Web Environments",
        "abstract": "Autonomous web agents navigating the internet.",
        "authors": ["Author B"],
        "year": 2024,
        "citation_count": 50,
        "embedding": emb2,
        "topics": ["LLM Agents"]
    }

    test_db.insert_papers([p1, p2])

    query_emb = [0.0, 1.0] + [0.0] * 382
    results = test_db.search_papers(query_embedding=query_emb, keywords=["agents"], limit=10)
    
    assert len(results) >= 1
    assert results[0]["id"] == "p2"
    assert "semantic_similarity" in results[0]
    assert results[0]["semantic_similarity"] > 0.5

def test_graph_metrics_calculation(test_db):
    p1 = {"id": "node_a", "title": "Paper A", "year": 2022, "citation_count": 100, "references": ["node_b"]}
    p2 = {"id": "node_b", "title": "Paper B", "year": 2020, "citation_count": 200, "references": []}
    
    test_db.insert_papers([p1, p2])
    metrics = test_db.get_graph_metrics()
    
    assert metrics is not None
    assert "foundational_papers" in metrics
    assert "graph_stats" in metrics
    assert metrics["graph_stats"]["nodes"] >= 2
