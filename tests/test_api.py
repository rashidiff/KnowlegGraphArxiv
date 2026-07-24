import pytest

def test_read_root(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "project" in data

def test_health_check(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database_driver" in data
    assert "graph_stats" in data

def test_corpus_status(api_client):
    response = api_client.get("/api/corpus/status")
    assert response.status_code == 200
    data = response.json()
    assert "paper_count" in data
    assert "refresh_running" in data

def test_explore_graph(api_client):
    response = api_client.get("/api/graph/explore")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "links" in data

def test_search_papers_endpoint(api_client):
    response = api_client.get("/api/papers/search?q=agent&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total_results" in data
    assert "papers" in data
    assert len(data["papers"]) <= 5

def test_upload_pdf_invalid_extension(api_client):
    response = api_client.post(
        "/api/upload",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
        data={"topic": "LLM Agents"}
    )
    assert response.status_code == 400

def test_upload_pdf_invalid_header(api_client):
    response = api_client.post(
        "/api/upload",
        files={"file": ("test.pdf", b"not a pdf header", "application/pdf")},
        data={"topic": "LLM Agents"}
    )
    assert response.status_code == 400
    assert "Invalid PDF" in response.json()["detail"]
