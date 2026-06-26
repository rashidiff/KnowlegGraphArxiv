import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm
from backend.database.manager import get_db

def verifier_node(state: AgentState) -> dict:
    """
    Verifier / Critic Agent.
    Strictly verifies that citation paths, centrality scores, communities, and citation counts
    mentioned in the responses correspond directly to the computed database metrics.
    Flags or marks as Unverified any unsupported assertions.
    """
    retrieved = state.get("retrieved_papers", [])
    analysis = state.get("analysis", "")
    ideas = state.get("research_ideas", "")
    reading_path = state.get("reading_path", "")
    graph_ctx = state.get("graph_context", {})
    
    db = get_db()
    metrics = db.get_graph_metrics() or {}

    # Extract verified graph values to cross-check
    verified_paths = []
    path_res = graph_ctx.get("path_search_result")
    if path_res and path_res.get("status") == "path_found":
        path_ids = [node["id"] for node in path_res.get("nodes", [])]
        verified_paths.append(" -> ".join(path_ids))

    # All centrality scores and citation counts in the DB
    verified_betweenness = metrics.get("betweenness_scores", {})
    verified_pagerank = metrics.get("pagerank_scores", {})
    verified_communities = metrics.get("communities", {})
    
    # Retrieved papers citation counts
    verified_citations = {p["id"]: p["citation_count"] for p in retrieved}
    
    # Format the verified metrics schema
    verified_metrics = {
        "verified_paths": verified_paths,
        "verified_betweenness_centrality": verified_betweenness,
        "verified_pagerank_centrality": verified_pagerank,
        "verified_citation_counts": verified_citations,
        "verified_communities": verified_communities
    }

    system_prompt = """You are a Graph Verifier and Critic Agent.
Your job is to read the generated Analysis, Reading Path, and Ideas, and cross-reference them with the actual computed VERIFIED GRAPH METRICS.
Verify the following rules:
1. If the text mentions a citation path between papers, it MUST exactly match one of the "verified_paths" lists, or be verifiable directly via the citation graph database. If the path does not exist, replace or prefix it with [UNVERIFIED CONCEPTUAL PATH].
2. If the text lists a Centrality Score (Betweenness or PageRank) for a paper, it MUST match the values in the verified lists. If it's a hallucination, replace it with [UNVERIFIED SCORE] or correct it.
3. If the text discusses a paper's citation count, it must match the citation count in "verified_citation_counts". If it doesn't, flag it.
4. If a community/cluster grouping is described, it must align with the Louvain divisions in "verified_communities".

You must respond with a JSON object in this format:
{
  "is_valid": true/false,
  "corrections": [
    {"target_claim": "hallucinated statement", "correction": "corrected statement or flag"}
  ],
  "unverified_elements": ["list of items marked as unverified"],
  "verification_report": "Summary of findings and corrections applied"
}
"""

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"--- VERIFIED GRAPH METRICS ---\n{json.dumps(verified_metrics, indent=2)}\n\n"
            f"--- GENERATED ANALYSIS ---\n{analysis}\n\n"
            f"--- GENERATED READING PATH ---\n{reading_path}\n\n"
            f"--- GENERATED IDEAS ---\n{ideas}"
        ))
    ]
    
    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        report = json.loads(response.content)
    except Exception as e:
        print(f"[Verifier Agent] Error: {e}")
        report = {
            "is_valid": False,
            "corrections": [],
            "unverified_elements": ["Verification could not run — LLM or JSON parse error."],
            "verification_report": f"Verification failed (LLM error: {e}). Treat all graph claims as unverified."
        }

    print(f"[Verifier Agent] Graph fact-checking complete. Valid: {report.get('is_valid')}")
    return {
        "verification_report": report
    }
