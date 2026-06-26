import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm

def responder_node(state: AgentState) -> dict:
    """
    Responder Agent.
    Synthesizes the final output using the strict Graph-First reasoning sections:
    Graph Facts, Graph Metrics, Evidence, Research Opportunities, and Uncertainty.
    """
    query = state["query"]
    analysis = state.get("analysis", "")
    graph_ctx = state.get("graph_context", {})
    reading_path = state.get("reading_path", "")
    ideas = state.get("research_ideas", "")
    report = state.get("verification_report", {})
    retrieved = state.get("retrieved_papers", [])

    # Format verified references list
    ref_list = []
    for p in retrieved[:12]:
        ref_list.append(f"- **[{p['id']}]** *{p['title']}*. ({p['year']}) - {p['venue']}. Citations: {p['citation_count']}.")
    references_str = "\n".join(ref_list)

    system_prompt = """You are a Master Graph Reasoning Communicator.
Your job is to compile the verified research outputs into a final academic response.
You MUST strictly organize your final response into these five exact sections, using the headers exactly as specified:

## 1. Graph Facts
- Detail ONLY real data extracted from the graph. Do not make conceptual inferences or guess relationships here.
- If a path was searched:
  * If a citation path exists, output the exact path:
    Paper X
    → cites
    Paper Y
    → cites
    Paper Z
    Path Length: [length]
    Betweenness Centrality: [bridge centrality score of the nodes]
    Community: [Louvain community ID or topic name]
  * If NO path exists, output:
    "No citation path exists between these communities in the current graph."
    And then output the nearest papers in both topics/communities, along with their local statistics (citation count, PageRank).

## 2. Graph Metrics
- Show graph metrics computed directly from the database:
  * PageRank centralities of key papers.
  * Betweenness Centralities (Bridge Scores).
  * Degree Centrality.
  * Louvain Community / Cluster memberships.

## 3. Evidence
- Provide the structural proof:
  * Detailed citation paths.
  * Node and edge statistics of the subgraph.
  * Example database queries (SQL or Cypher) used to fetch these facts.

## 4. Research Opportunities
- Propose research gaps and hypotheses.
- EVERY OPPORTUNITY here MUST be connected directly to a graph structural anomaly (e.g. disconnected communities, missing Method-Dataset edges, or bridge nodes).
- If no graph anomaly is present, state "None". Do not formulate general ideas from scratch.

## 5. Uncertainty
- Explicitly list what is missing, disconnected, or unverified in the graph (e.g., disconnected topics, missing reference links, or elements marked as UNVERIFIED by the critic).

---
Always end with a **References** section listing the details of the papers.
Make the layout clean, readable, professional, and markdown-formatted.
"""

    user_content = f"""User Query: {query}

-- Multi-Agent Pipeline Data inputs --
Analysis: {analysis}
Graph Context: {json.dumps(graph_ctx, default=str)}
Reading Path: {reading_path}
Research Ideas: {ideas}
Verification Report: {json.dumps(report, indent=2)}

-- Available Citation References --
{references_str}
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    try:
        response = llm.invoke(messages)
        final_ans = response.content
    except Exception as e:
        print(f"[Responder Agent] Error: {e}")
        final_ans = "Failed to synthesize final response due to an LLM error."

    return {
        "final_response": final_ans
    }
