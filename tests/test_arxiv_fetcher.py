import pytest
from backend.agents.arxiv_fetcher import _bare_arxiv_id, search_arxiv

def test_bare_arxiv_id_parsing():
    url1 = "http://arxiv.org/abs/2303.11366v1"
    assert _bare_arxiv_id(url1) == "2303.11366"
    
    url2 = "2401.05678v3"
    assert _bare_arxiv_id(url2) == "2401.05678"

def test_search_arxiv_empty_keywords():
    results = search_arxiv([])
    assert results == []
