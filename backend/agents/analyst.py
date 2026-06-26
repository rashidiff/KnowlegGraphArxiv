import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm

def analyst_node(state: AgentState) -> dict:
    """
    Analyzes retrieved papers to synthesize contributions, methodologies,
    limitations, and state-of-the-art comparisons.
    """
    query = state["query"]
    retrieved = state.get("retrieved_papers", [])
    graph_ctx = state.get("graph_context", {})
    plan = state.get("research_plan", "")

    if not retrieved:
        return {"analysis": "No relevant papers were retrieved to conduct analysis."}

    # Format papers
    papers_context = []
    for i, p in enumerate(retrieved[:8]):
        papers_context.append(
            f"[{p['id']}] {p['title']} ({p['year']}) - Venue: {p['venue']}\n"
            f"Abstract: {p['abstract']}\n"
        )
    papers_str = "\n".join(papers_context)

    system_prompt = """You are a Senior Research Analyst Agent.
Your job is to read the abstracts and metadata of the retrieved papers, and perform a deep comparison.
Your output must contain:
1. **Key Contributions**: What are the main breakthroughs presented across these papers?
2. **Methodologies Used**: What datasets, model architectures (e.g. LLM routing, planning, tools), or techniques are utilized?
3. **Core Limitations**: What are the explicitly stated or apparent limitations in these works (e.g. cost, evaluation fragility, error propagation)?
4. **Research Trends**: What trends do you see (e.g., transition from text-only reasoning to interactive GUI-based environments)?

Be scientific, analytical, and objective. Ground your claims *only* in the provided paper information. Do not speculate or introduce unverified outer knowledge.
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {query}\nPlan: {plan}\n\nPapers:\n{papers_str}\n\nGraph Context:\n{graph_ctx}")
    ]
    
    try:
        response = llm.invoke(messages)
        analysis = response.content
    except Exception as e:
        print(f"[Analyst Agent] Error: {e}")
        analysis = "Failed to run synthesis due to an LLM error."

    return {
        "analysis": analysis
    }
