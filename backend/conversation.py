"""Bounded conversation context shared by research agents."""

from typing import Any


def conversation_context(history: list[dict[str, Any]], max_messages: int = 12, max_chars: int = 6_000) -> str:
    """Format only valid user/assistant turns, keeping prompts bounded."""
    lines: list[str] = []
    for item in history[-max_messages:]:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        text = content.strip()
        if text:
            lines.append(f"{role.title()}: {text[:1200]}")
    if not lines:
        return ""
    return "Previous conversation (use only to resolve references, never as evidence):\n" + "\n".join(lines)[-max_chars:]
