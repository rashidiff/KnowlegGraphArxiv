from backend.conversation import conversation_context


def test_conversation_context_keeps_valid_recent_turns_only():
    context = conversation_context([
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": "Compare these papers"},
        {"role": "assistant", "content": "Which papers?"},
        {"role": "user", "content": "The first two"},
    ])
    assert "System:" not in context
    assert "Compare these papers" in context
    assert "The first two" in context
