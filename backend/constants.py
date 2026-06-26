"""
Shared constants used across backend modules.
Single source of truth for taxonomy, topic aliases, and exclusion lists.
"""

TAXONOMY = [
    "Browser Agents",
    "Web Agents",
    "Agent Evaluation",
    "Tool Use for LLMs",
    "Autonomous Agents",
    "Multi-Agent Systems",
    "Human-AI Interaction",
]

# Maps LLM-generated or legacy topic strings to canonical taxonomy names.
TOPIC_ALIASES: dict[str, str] = {
    "Tool Use": "Tool Use for LLMs",
    "LLM Agents": "Autonomous Agents",
    "Browser Agents": "Browser Agents",
    "Web Agents": "Web Agents",
    "Agent Evaluation": "Agent Evaluation",
    "Tool Use for LLMs": "Tool Use for LLMs",
    "Autonomous Agents": "Autonomous Agents",
    "Multi-Agent Systems": "Multi-Agent Systems",
    "Human-AI Interaction": "Human-AI Interaction",
}

EXCLUSION_KEYWORDS = [
    "chimpanzee", "dolphin", "macaque", "monkey", "primate", "crow", "ape",
    "animal tool use", "infant development", "bottlenose", "neural basis of tool use",
    "ecological conditions", "pigeon", "baboon", "wild chimpanzees", "aquatic animals",
    "comparative psychology", "zoology", "marine biology", "neuroscience",
    "developmental psychology", "primatology", "body schema", "macaque neurones",
    "tool behavior in animals",
]
