import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm

def reading_path_agent_node(state: AgentState) -> dict:
    """
    Computes a logical, chronological, or dependency-based reading order 
    for the top retrieved papers, explaining the progression of research.
    """
    query = state["query"]
    retrieved = state.get("retrieved_papers", [])
    graph_ctx = state.get("graph_context", {})
    
    if not retrieved:
        return {"reading_path": "No papers retrieved to generate a reading path."}

    # Format retrieved papers for the LLM
    papers_context = []
    for i, p in enumerate(retrieved[:8]):
        papers_context.append(
            f"Paper {i+1} [ID: {p['id']}]:\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Year: {p['year']}\n"
            f"Venue: {p['venue']}\n"
            f"Citations: {p['citation_count']}\n"
            f"Abstract: {p['abstract'][:250]}...\n"
        )
    papers_str = "\n".join(papers_context)

    system_prompt = """You are a Reading Path Agent.
Your job is to arrange the provided research papers into a logical, step-by-step reading roadmap (Step 1, Step 2, Step 3, etc.) for a student or researcher wanting to understand the topic.
Guidelines:
1. Arrange them by dependency: Put foundational papers (high citations/earlier years) first, followed by method extensions, and finally benchmarks/evaluation.
2. For each step, provide:
   - Paper Title and Citation key (e.g. [Yao et al., 2023])
   - A 1-2 sentence explanation of *why* to read it in this sequence (e.g., "This paper introduces the core architecture", "This paper extends the core architecture to browser interfaces", "This paper evaluates the extensions").
3. Make the reading path clean, structured, and easy to follow.
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {query}\n\nPapers:\n{papers_str}\n\nGraph Context:\n{graph_ctx}")
    ]
    
    try:
        response = llm.invoke(messages)
        path = response.content
    except Exception as e:
        print(f"[Reading Path Agent] Error: {e}")
        path = "Failed to generate reading path due to system error."

    return {
        "reading_path": path
    }
