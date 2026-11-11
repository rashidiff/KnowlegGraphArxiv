"""Exercise path lookup against a real isolated corpus, without LLM calls."""
import pytest
import backend.main as main


@pytest.fixture
def path_client(api_client, test_db, monkeypatch):
    test_db.insert_papers([
        {"id": "a", "title": "Paper A", "year": 2024, "references": ["b"]},
        {"id": "b", "title": "Bridge paper", "year": 2023, "references": ["c"]},
        {"id": "c", "title": "Paper C", "year": 2022},
        {"id": "isolated", "title": "Unconnected paper", "year": 2024},
    ])
    monkeypatch.setattr(main, "get_db", lambda: test_db)
    return api_client


def test_path_includes_intermediate_papers(path_client):
    response = path_client.get('/api/graph/path', params={"start_id": "a", "end_id": "c"})
    assert response.status_code == 200
    result = response.json()
    assert result['found'] is True
    assert result['hops'] == 2
    assert [p['id'] for p in result['path']] == ['a', 'b', 'c']
    assert 'embedding' not in result['path'][0]


def test_path_can_traverse_reverse_citations(path_client):
    result = path_client.get('/api/graph/path', params={"start_id": "c", "end_id": "a"}).json()
    assert [p['id'] for p in result['path']] == ['c', 'b', 'a']
    assert result['relationship'] == 'citation_connection'


def test_disconnected_is_distinct_from_missing(path_client):
    response = path_client.get('/api/graph/path', params={"start_id": "a", "end_id": "isolated"})
    assert response.status_code == 200
    assert response.json()['found'] is False
    assert response.json()['hops'] is None
    assert response.json()['path'] == []
    missing = path_client.get('/api/graph/path', params={"start_id": "a", "end_id": "missing"})
    assert missing.status_code == 404


def test_same_paper_has_zero_hops(path_client):
    result = path_client.get('/api/graph/path', params={"start_id": "isolated", "end_id": "isolated"}).json()
    assert result['found'] is True
    assert result['hops'] == 0
    assert [p['id'] for p in result['path']] == ['isolated']


def test_path_requires_nonempty_endpoints(path_client):
    assert path_client.get('/api/graph/path').status_code == 422
    assert path_client.get('/api/graph/path', params={"start_id": "", "end_id": "a"}).status_code == 422
