import pytest
from backend.agents.graph import build_workflow
from backend.agents.router import router_node

def test_build_workflow():
    app = build_workflow()
    assert app is not None

def test_router_node_clarification_flag():
    state = {
        "query": "compare papers",
        "clarification_answers": [{"question": "Which papers?", "answer": "ReAct and Toolformer"}]
    }
    res = router_node(state)
    assert res["clarification_needed"] is False
    assert res["clarification_question"] is None
