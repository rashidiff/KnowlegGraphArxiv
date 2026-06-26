import json
import networkx as nx
from typing import List, Dict, Any, Tuple
from backend.agents.state import AgentState
from backend.database.manager import get_db
from backend.agents.router import get_llm
from backend.agents.retriever import get_embedding_model
from langchain_core.messages import SystemMessage


def graph_agent_node(state: AgentState) -> dict:
    """
    Graph Insight Agent — dynamic mode.
    Builds a per-query citation graph from the retrieved papers,
    computes PageRank / betweenness / communities on that mini-graph,
    and (optionally) finds shortest paths between topics.
    """
    query = state["query"]
    retrieved: List[Dict] = state.get("retrieved_papers", [])
    db = get_db()

    # ── 1. Build per-query mini-graph ────────────────────────────────────
    mini_G = nx.DiGraph()
    paper_ids = {p["id"] for p in retrieved}

    for p in retrieved:
        mini_G.add_node(
            p["id"],
            title=p.get("title", ""),
            citation_count=p.get("citation_count", 0),
            year=p.get("year", 2024),
        )

    # Add citation edges between retrieved papers.
    # Primary source: in-memory graph (covers cached papers that may lack a 'references' field).
    # Secondary source: 'references' field on freshly-enriched papers (not yet in _graph).
    for u, v in db.get_citation_edges_between(list(paper_ids)):
        mini_G.add_edge(u, v)
    # Also pick up edges on newly-fetched papers before _graph was rebuilt
    for p in retrieved:
        for ref_id in p.get("references", []):
            if ref_id in paper_ids and not mini_G.has_edge(p["id"], ref_id):
                mini_G.add_edge(p["id"], ref_id)

    print(f"[Graph Agent] Mini-graph: {mini_G.number_of_nodes()} nodes, {mini_G.number_of_edges()} edges")

    # ── 2. PageRank ───────────────────────────────────────────────────────
    if mini_G.number_of_edges() > 0:
        pr = nx.pagerank(mini_G)
    else:
        # No citation edges between retrieved papers — use citation_count as proxy
        max_cit = max((p.get("citation_count", 0) for p in retrieved), default=1)
        pr = {p["id"]: (p.get("citation_count", 0) / max(max_cit, 1)) for p in retrieved}

    # ── 3. Betweenness centrality ─────────────────────────────────────────
    betweenness: Dict[str, float] = {}
    if mini_G.number_of_nodes() > 2 and mini_G.number_of_edges() > 1:
        betweenness = nx.betweenness_centrality(mini_G)

    # ── 4. Louvain communities ────────────────────────────────────────────
    community_map: Dict[str, int] = {}
    try:
        undirected = mini_G.to_undirected()
        if undirected.number_of_nodes() >= 3:
            comms = nx.community.louvain_communities(undirected, seed=42)
            for i, comm in enumerate(comms):
                for nid in comm:
                    community_map[nid] = i
    except Exception:
        pass

    # ── 5. Derive insights from mini-graph ───────────────────────────────
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    foundational_papers = [
        {
            "id": pid,
            "score": round(score, 6),
            "title": mini_G.nodes[pid].get("title", "") if mini_G.has_node(pid) else "",
        }
        for pid, score in sorted_pr[:8]
    ]

    sorted_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    bridge_papers = [
        {
            "id": pid,
            "score": round(score, 6),
            "title": mini_G.nodes[pid].get("title", ""),
        }
        for pid, score in sorted_bc[:5] if score > 0
    ]

    # Cluster groups
    cluster_groups: Dict[str, list] = {}
    for p in retrieved:
        pid = p["id"]
        comm_id = str(community_map.get(pid, "Unassigned"))
        cluster_groups.setdefault(comm_id, []).append({
            "id": pid,
            "title": p.get("title", ""),
            "citations": p.get("citation_count", 0),
        })

    # ── 6. Optional path search ───────────────────────────────────────────
    path_search_result = None
    llm = get_llm()

    parser_prompt = f"""You are a Graph Query parser.
Determine if the user asks to find a connection, path, or bridge between specific topics or papers.
Taxonomy topics: Browser Agents, Web Agents, Agent Evaluation, Tool Use for LLMs,
Autonomous Agents, Multi-Agent Systems, Human-AI Interaction

User Query: {query}

Respond ONLY with JSON:
{{
  "is_connection_query": true,
  "endpoints": [
    {{"type": "topic", "value": "Browser Agents"}},
    {{"type": "topic", "value": "Human-AI Interaction"}}
  ]
}}"""

    is_conn = False
    endpoints = []
    try:
        res = llm.invoke([SystemMessage(content=parser_prompt)], response_format={"type": "json_object"})
        parsed = json.loads(res.content)
        is_conn = parsed.get("is_connection_query", False)
        endpoints = parsed.get("endpoints", [])
    except Exception as e:
        print(f"[Graph Agent] Parser error: {e}")

    if is_conn and len(endpoints) >= 2:
        ep1, ep2 = endpoints[0], endpoints[1]
        nodes_a, papers_a = _resolve_endpoint(ep1, db)
        nodes_b, papers_b = _resolve_endpoint(ep2, db)

        shortest_path = None
        best_len = 999_999

        for na in nodes_a[:10]:
            for nb in nodes_b[:10]:
                path = db.get_citation_path(na, nb)
                if path and len(path) < best_len:
                    shortest_path = path
                    best_len = len(path)

        if shortest_path:
            path_steps = []
            for i in range(len(shortest_path) - 1):
                p_curr = shortest_path[i]
                p_next = shortest_path[i + 1]
                details = db.get_paper(p_curr["id"])
                direction = "cites" if details and p_next["id"] in details.get("references", []) else "is cited by"
                path_steps.append({
                    "from": p_curr["title"],
                    "from_id": p_curr["id"],
                    "direction": direction,
                    "to": p_next["title"],
                    "to_id": p_next["id"],
                })
            path_search_result = {
                "status": "path_found",
                "endpoints": [ep1["value"], ep2["value"]],
                "path_steps": path_steps,
                "path_length": len(shortest_path) - 1,
                "nodes": [
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "pagerank": pr.get(p["id"], 0.0),
                        "betweenness": betweenness.get(p["id"], 0.0),
                        "community": community_map.get(p["id"], "Unassigned"),
                    }
                    for p in shortest_path
                ],
            }
        else:
            path_search_result = {
                "status": "no_path_found",
                "endpoints": [ep1["value"], ep2["value"]],
                "nearest_papers_topic_a": [
                    {"id": p["id"], "title": p["title"], "citations": p.get("citation_count", 0)}
                    for p in papers_a[:5]
                ],
                "nearest_papers_topic_b": [
                    {"id": p["id"], "title": p["title"], "citations": p.get("citation_count", 0)}
                    for p in papers_b[:5]
                ],
            }

    # ── 7. Method–Dataset coverage gaps ──────────────────────────────────
    methods_seen = list({e["value"] for p in retrieved for e in p.get("entities", []) if e["type"] == "method"})
    datasets_seen = list({e["value"] for p in retrieved for e in p.get("entities", []) if e["type"] == "dataset"})
    missing_evaluations = [
        {"method": m, "dataset": d}
        for m in methods_seen[:4]
        for d in datasets_seen[:4]
        if not any(
            m in [e["value"] for e in p.get("entities", [])] and
            d in [e["value"] for e in p.get("entities", [])]
            for p in retrieved
        )
    ]

    # ── 8. Subgraph for graph view (use full DB graph focused on retrieved) ─
    retrieved_ids = [p["id"] for p in retrieved[:100]]
    try:
        subgraph = db.get_graph_data(focus_paper_ids=retrieved_ids, max_nodes=150)
    except Exception:
        subgraph = {"nodes": [], "links": []}

    citation_paths = []
    if path_search_result and path_search_result.get("status") == "path_found":
        citation_paths = [{"path": path_search_result["nodes"]}]

    graph_context = {
        "foundational_papers": foundational_papers,
        "bridge_papers": bridge_papers,
        "clusters": cluster_groups,
        "disconnected_subfields": [],
        "missing_links": missing_evaluations[:5],
        "communities": community_map,
        "path_search_result": path_search_result,
        "subgraph": subgraph,
        "citation_paths": citation_paths,
        "graph_stats": {
            "nodes": mini_G.number_of_nodes(),
            "edges": mini_G.number_of_edges(),
            "density": round(nx.density(mini_G), 6),
        },
    }

    print("[Graph Agent] Per-query graph analysis complete.")
    return {"graph_context": graph_context}


def _resolve_endpoint(endpoint: Dict[str, str], db) -> Tuple[List[str], List[Dict]]:
    """Resolve a query endpoint (topic or paper name) to paper IDs."""
    if endpoint["type"] == "topic":
        papers = db.get_papers_by_topic(endpoint["value"])
        return [p["id"] for p in papers], papers
    else:
        model = get_embedding_model()
        embedding = model.encode(endpoint["value"]).tolist()
        papers = db.search_papers(query_embedding=embedding, keywords=[endpoint["value"]], limit=5)
        return [p["id"] for p in papers], papers
