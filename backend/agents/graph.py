from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.router import router_node
from backend.agents.research_planner import research_planner_node
from backend.agents.retriever import retriever_node
from backend.agents.graph_agent import graph_agent_node
from backend.agents.reading_path import reading_path_agent_node
from backend.agents.analyst import analyst_node
from backend.agents.idea import idea_node
from backend.agents.verifier import verifier_node
from backend.agents.responder import responder_node

def route_clarification(state: AgentState):
    """Conditional router based on whether Orchestrator detected ambiguity."""
    if state.get("clarification_needed", False):
        return "clarify"
    return "continue"

def build_workflow():
    """Constructs and compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add all agent nodes
    workflow.add_node("router", router_node)
    workflow.add_node("planner", research_planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("graph_agent", graph_agent_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reading_path", reading_path_agent_node)
    workflow.add_node("idea", idea_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("responder", responder_node)

    # 2. Set entry point
    workflow.set_entry_point("router")

    # 3. Add conditional router edge
    workflow.add_conditional_edges(
        "router",
        route_clarification,
        {
            "clarify": END, # Stop and ask user clarification
            "continue": "planner"
        }
    )

    # 4. Add sequential execution flow edges
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "graph_agent")
    workflow.add_edge("graph_agent", "analyst")
    workflow.add_edge("analyst", "reading_path")
    workflow.add_edge("reading_path", "idea")
    workflow.add_edge("idea", "verifier")
    workflow.add_edge("verifier", "responder")
    workflow.add_edge("responder", END)

    # Compile the graph
    app = workflow.compile()
    return app
