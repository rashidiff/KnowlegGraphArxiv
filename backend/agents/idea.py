import os
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm

def idea_node(state: AgentState) -> dict:
    """
    Research Idea Generator Agent.
    Formulates novel research hypotheses *only* when backed by structural anomalies
    or gaps identified in the citation graph (e.g. missing links, disconnected clusters).
    """
    query = state["query"]
    analysis = state.get("analysis", "")
    graph_ctx = state.get("graph_context", {})
    
    # 1. Check path search results
    path_res = graph_ctx.get("path_search_result")
    has_evidence = False
    paths_str = ""
    
    if path_res:
        if path_res["status"] == "path_found":
            has_evidence = True
            steps = [f"'{step['from']}' [{step['from_id']}] {step['direction']} '{step['to']}' [{step['to_id']}]" for step in path_res["path_steps"]]
            paths_str = f"Shortest Citation Path between {path_res['endpoints'][0]} and {path_res['endpoints'][1]}:\n" + "\n".join([f"  - Step {i+1}: {step}" for i, step in enumerate(steps)])
        elif path_res["status"] == "no_path_found":
            has_evidence = True
            paths_str = f"Disconnected Subfields Gaps: No citation path exists between '{path_res['endpoints'][0]}' and '{path_res['endpoints'][1]}' in the current graph database."
            
    # 2. Extract other structural elements
    bridge_nodes = graph_ctx.get("bridge_papers", [])
    if bridge_nodes:
        has_evidence = True
        
    disconnected = graph_ctx.get("disconnected_subfields", [])
    if disconnected:
        has_evidence = True
        
    missing_links = graph_ctx.get("missing_links", [])
    if missing_links:
        has_evidence = True

    # 3. Strict Check: If no graph evidence exists, do not generate hypotheses
    if not has_evidence:
        return {
            "research_ideas": "No research hypotheses generated. Graph evidence indicates the citation network is uniform, and no structural anomalies, missing edges, or disconnected communities were detected in the current subgraph."
        }

    # Format lists for LLM prompt
    bridge_str = "\n".join([f"- Paper: '{p['title']}' [{p['id']}] (Centrality Score: {p['score']:.4f})" for p in bridge_nodes])
    disconnected_str = "\n".join([f"- Community '{item['topic_a']}' has no citation paths to Community '{item['topic_b']}'" for item in disconnected])
    missing_str = "\n".join([f"- Method '{item['method']}' has not been evaluated on Dataset '{item['dataset']}'" for item in missing_links])

    system_prompt = """You are a Graph-First Research Proposer Agent.
Your job is to look at the structural graph analysis (citation paths, bridging papers, disconnected components, missing Method-Dataset edges) and propose a research hypothesis.
CRITICAL RULES:
1. You MUST NOT brainstorm general ideas from scratch.
2. Every hypothesis you propose MUST be directly linked to one of the provided graph structural gaps:
   - A Bridge Node (how it acts as a gateway and what occurs on either side).
   - A Community Gap / Disconnected Cluster (why they are separated, and how bridging them can solve a problem).
   - A Missing Link (e.g., a Method that has never been evaluated on a specific Dataset).
   - An anomalous Citation Path.
3. If no clear graph anomaly is provided, state clearly: "No graph anomalies detected to justify a research proposal."

Your proposal must be structured as follows:
- **Graph Evidence Reference**: State the exact bridge node, missing link, or disconnected communities.
- **Proposed Research Hypothesis**: Formulate a testable claim.
- **Proposed Methodology**: Steps to bridge the gap.
- **Evaluation Plan**: Benchmarks to validate.
"""

    user_content = f"""User Query: {query}

--- GRAPH STRUCTURAL EVIDENCE ---
{paths_str}

Bridge Nodes:
{bridge_str}

Disconnected Communities:
{disconnected_str}

Missing Method-Dataset Evaluation Links:
{missing_str}

--- RETRIEVED PAPERS METADATA ---
{analysis}
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    try:
        response = llm.invoke(messages)
        ideas = response.content
    except Exception as e:
        print(f"[Idea Agent] Error: {e}")
        ideas = "Failed to generate research ideas due to LLM error."

    return {
        "research_ideas": ideas
    }
