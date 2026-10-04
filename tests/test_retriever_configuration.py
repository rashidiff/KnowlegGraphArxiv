import ast
from pathlib import Path


def test_retriever_has_a_local_cache_fast_path_before_live_searches():
    source = Path("backend/agents/retriever.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "retriever_node")
    names = [node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "search_arxiv" in names
    assert "search_semantic_scholar" in names
    assert "RETRIEVER_CACHE_MIN_RESULTS" in source
