import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm

def research_planner_node(state: AgentState) -> dict:
    """
    Parses the user's query and creates a structured plan of what concepts, 
    paper types, and graph pathways need to be explored to answer the query.
    """
    query = state["query"]
    
    # If clarification history exists, append it to give the planner context
    clarification_context = ""
    if "clarification_answers" in state and state["clarification_answers"]:
        clarification_context = "\nUser Clarifications:\n" + "\n".join(
            [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in state["clarification_answers"]]
        )

    system_prompt = """You are a Research Planner Agent in a scientific navigator.
Your task is to analyze the user's research query (and any clarifications) and generate a structured research plan.
Your plan should outline:
1. Core terms and synonyms to search for.
2. The type of analysis needed (e.g., historical foundational retrieval, comparison, reading path generation, or gap analysis).
3. Specific graph entities to focus on (e.g. authors, concepts, datasets).
4. The key objectives of this search.

Keep the plan clear, structured, and focused. This plan will guide subsequent retrieval and analyst agents.
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {query}{clarification_context}")
    ]
    
    try:
        response = llm.invoke(messages)
        plan = response.content
    except Exception as e:
        print(f"[Research Planner Agent] Error: {e}")
        plan = f"Execute standard search and analysis for: {query}"

    return {
        "research_plan": plan
    }
