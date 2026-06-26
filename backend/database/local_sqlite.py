import os
import json
import time
import sqlite3
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from backend.database.base import BaseDatabase
from backend.constants import TAXONOMY

class LocalSQLiteDB(BaseDatabase):
    def __init__(self, db_path: str = "data/research_navigator.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enforce referential integrity so dangling citations are rejected
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._graph: Optional[nx.DiGraph] = None
        self._cached_metrics: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0.0
        self._metrics_ttl: float = 300.0  # 5 minutes
        self.init_db()
        self._load_graph()

    def init_db(self) -> None:
        cursor = self.conn.cursor()
        
        # Papers table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT, -- JSON List
            year INTEGER,
            venue TEXT,
            citation_count INTEGER DEFAULT 0,
            venue_quality REAL DEFAULT 0.0,
            embedding BLOB, -- Float32 bytes
            intro_summary TEXT,
            conclusion_summary TEXT,
            section_headers TEXT -- JSON List
        )
        """)

        # Citations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            source_id TEXT,
            target_id TEXT,
            PRIMARY KEY (source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES papers(id),
            FOREIGN KEY (target_id) REFERENCES papers(id)
        )
        """)

        # Paper Topics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_topics (
            paper_id TEXT,
            topic TEXT,
            PRIMARY KEY (paper_id, topic),
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
        """)

        # Paper Entities table (for concepts, methods, datasets, etc.)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            type TEXT, -- 'concept', 'method', 'dataset', 'task', etc.
            value TEXT,
            UNIQUE (paper_id, type, value),
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
        """)
        
        # Add arxiv_id column if it doesn't exist (migration for existing DBs)
        try:
            cursor.execute("ALTER TABLE papers ADD COLUMN arxiv_id TEXT")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers (arxiv_id)")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        self.conn.commit()

    def _load_graph(self) -> None:
        """Load citation graph from SQLite into a NetworkX DiGraph."""
        G = nx.DiGraph()
        cursor = self.conn.cursor()
        
        # Add nodes
        cursor.execute("SELECT id, title, year, venue, citation_count FROM papers")
        for row in cursor.fetchall():
            G.add_node(
                row["id"], 
                title=row["title"], 
                year=row["year"], 
                venue=row["venue"], 
                citation_count=row["citation_count"]
            )
            
        # Add edges
        cursor.execute("SELECT source_id, target_id FROM citations")
        for row in cursor.fetchall():
            # Only add edge if both nodes exist in papers database
            if G.has_node(row["source_id"]) and G.has_node(row["target_id"]):
                G.add_edge(row["source_id"], row["target_id"])
                
        self._graph = G

    def insert_papers(self, papers: List[Dict[str, Any]]) -> None:
        cursor = self.conn.cursor()
        for p in papers:
            # 1. Insert Paper Meta
            embedding_bytes = None
            if "embedding" in p and p["embedding"] is not None:
                embedding_bytes = np.array(p["embedding"], dtype=np.float32).tobytes()

            cursor.execute("""
            INSERT OR REPLACE INTO papers (
                id, title, abstract, authors, year, venue,
                citation_count, venue_quality, embedding,
                intro_summary, conclusion_summary, section_headers, arxiv_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p["id"],
                p["title"],
                p.get("abstract", ""),
                json.dumps(p.get("authors", [])),
                p.get("year", 2026),
                p.get("venue", "Unknown"),
                p.get("citation_count", 0),
                p.get("venue_quality", 0.0),
                embedding_bytes,
                p.get("intro_summary"),
                p.get("conclusion_summary"),
                json.dumps(p.get("section_headers", [])),
                p.get("arxiv_id"),
            ))

            # 2. Insert Topics
            topics = p.get("topics", [])
            for t in topics:
                cursor.execute("""
                INSERT OR IGNORE INTO paper_topics (paper_id, topic)
                VALUES (?, ?)
                """, (p["id"], t))

            # 3. Insert Entities (concepts, methods, datasets) — use IGNORE to avoid dups on re-seed
            entities = p.get("entities", []) # List of dicts: {"type": "concept", "value": "Browser Agent"}
            for ent in entities:
                cursor.execute("""
                INSERT OR IGNORE INTO paper_entities (paper_id, type, value)
                VALUES (?, ?, ?)
                """, (p["id"], ent["type"], ent["value"]))

        self.conn.commit()

        # 4. Insert Citations — only insert if BOTH endpoints exist in papers table.
        # PRAGMA foreign_keys=ON would reject invalid inserts, but we explicitly
        # guard here so missing referenced papers don't raise exceptions.
        # 4. Insert stub papers for references so the citations table can link them
        for p in papers:
            references = p.get("references", [])
            for ref_id in references:
                cursor.execute("""
                INSERT OR IGNORE INTO papers (
                    id, title, abstract, authors, year, venue, citation_count, venue_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ref_id, "Cited Reference", "", "[]", p.get("year", 2024) - 2, "Unknown", 0, 0.0))

        # 5. Insert Citations
        for p in papers:
            references = p.get("references", [])
            for ref_id in references:
                cursor.execute("""
                INSERT OR IGNORE INTO citations (source_id, target_id)
                VALUES (?, ?)
                """, (p["id"], ref_id))

        self.conn.commit()
        self._invalidate_cache()
        self._load_graph()

    def search_papers(self, 
                      query_embedding: List[float], 
                      keywords: List[str], 
                      topic: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        
        # 1. Base Query with optional topic filter (excluding stubs without embeddings)
        if topic:
            cursor.execute("""
            SELECT p.* FROM papers p
            JOIN paper_topics pt ON p.id = pt.paper_id
            WHERE pt.topic = ? AND p.embedding IS NOT NULL
            """, (topic,))
        else:
            cursor.execute("SELECT * FROM papers WHERE embedding IS NOT NULL")
            
        all_papers = [dict(row) for row in cursor.fetchall()]
        if not all_papers:
            return []

        # Ensure graph metrics are active
        pageranks = nx.pagerank(self._graph) if self._graph and len(self._graph) > 0 else {}
        
        # Compute min/max stats for normalization
        years = [p["year"] for p in all_papers if p["year"] is not None]
        min_year = min(years) if years else 2000
        max_year = max(years) if years else 2026
        year_range = max_year - min_year if max_year > min_year else 1

        citation_counts = [p["citation_count"] for p in all_papers if p["citation_count"] is not None]
        max_citations = max(citation_counts) if citation_counts else 1
        max_citations_log = np.log1p(max_citations)

        # Calculate scores for each paper
        scored_papers = []
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        for p in all_papers:
            # A. Semantic Similarity (45%)
            similarity = 0.0
            if p["embedding"] is not None and q_norm > 0:
                p_vec = np.frombuffer(p["embedding"], dtype=np.float32)
                p_norm = np.linalg.norm(p_vec)
                if p_norm > 0:
                    similarity = float(np.dot(p_vec, q_vec) / (p_norm * q_norm))
                    # Map cosine similarity [-1, 1] to [0, 1]
                    similarity = (similarity + 1.0) / 2.0
            
            # If keywords provided, perform simple keyword bonus
            keyword_score = 0.0
            if keywords:
                title_lower = p["title"].lower()
                abstract_lower = p["abstract"].lower()
                matches = sum(1 for kw in keywords if kw.lower() in title_lower or kw.lower() in abstract_lower)
                keyword_score = matches / len(keywords)
                # Blend keyword score with semantic similarity (70% semantic, 30% keyword)
                similarity = 0.7 * similarity + 0.3 * keyword_score

            # B. Citation Count Score (25%)
            citations = p["citation_count"] or 0
            citation_score = float(np.log1p(citations) / max_citations_log) if max_citations_log > 0 else 0.0

            # C. Graph Centrality (PageRank) (15%)
            centrality = pageranks.get(p["id"], 0.0)
            # Normalize centrality
            max_centrality = max(pageranks.values()) if pageranks else 1.0
            normalized_centrality = centrality / max_centrality if max_centrality > 0 else 0.0

            # D. Recency Score (10%)
            year = p["year"] or min_year
            recency = float((year - min_year) / year_range) if year_range > 0 else 1.0

            # E. Venue Quality (5%)
            venue_quality = p["venue_quality"] or 0.0

            # Weighted final score (gated by semantic similarity to filter out irrelevant highly-cited papers)
            if similarity < 0.65:
                final_score = 0.1 * similarity
            else:
                final_score = (
                    0.45 * similarity +
                    0.25 * citation_score +
                    0.15 * normalized_centrality +
                    0.10 * recency +
                    0.05 * venue_quality
                )

            p["authors"] = json.loads(p["authors"]) if p["authors"] else []
            p["section_headers"] = json.loads(p["section_headers"]) if p["section_headers"] else []
            p.pop("embedding", None) # Don't return bytes in results
            p["final_score"] = final_score
            p["graph_centrality"] = normalized_centrality
            p["semantic_similarity"] = similarity
            
            scored_papers.append(p)

        # Sort by final score descending
        scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        p = dict(row)
        p["authors"] = json.loads(p["authors"]) if p["authors"] else []
        p["section_headers"] = json.loads(p["section_headers"]) if p["section_headers"] else []
        p.pop("embedding", None)

        # References this paper cites — only include IDs that exist locally (so UI can navigate)
        cursor.execute("""
            SELECT c.target_id FROM citations c
            INNER JOIN papers p2 ON c.target_id = p2.id
            WHERE c.source_id = ?
        """, (paper_id,))
        p["references"] = [r["target_id"] for r in cursor.fetchall()]

        # Total stored reference count (including external ones not in our DB)
        cursor.execute("SELECT COUNT(*) as cnt FROM citations WHERE source_id = ?", (paper_id,))
        row2 = cursor.fetchone()
        p["total_references"] = row2["cnt"] if row2 else 0

        # Papers that cite this paper — only local ones
        cursor.execute("""
            SELECT c.source_id FROM citations c
            INNER JOIN papers p2 ON c.source_id = p2.id
            WHERE c.target_id = ?
        """, (paper_id,))
        p["citations"] = [r["source_id"] for r in cursor.fetchall()]

        # Add topics
        cursor.execute("SELECT topic FROM paper_topics WHERE paper_id = ?", (paper_id,))
        p["topics"] = [r["topic"] for r in cursor.fetchall()]

        # Add entities
        cursor.execute("SELECT type, value FROM paper_entities WHERE paper_id = ?", (paper_id,))
        p["entities"] = [{"type": r["type"], "value": r["value"]} for r in cursor.fetchall()]

        return p

    def get_citation_edges_between(self, paper_ids: List[str]) -> List[tuple]:
        """Return all (source, target) citation edges where both endpoints are in paper_ids."""
        id_set = set(paper_ids)
        if not self._graph:
            return []
        return [(u, v) for u, v in self._graph.edges() if u in id_set and v in id_set]

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Look up a cached paper by its arXiv ID (e.g. '2312.12345')."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self.get_paper(row["id"])

    def get_citation_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        if not self._graph:
            return []
        
        try:
            # shortest_path in NetworkX searches both directions if undirected, or directed path if directed G.
            # Standard citation path is a directed path. Since paper A cites paper B, path is A -> B.
            # Sometimes paths go through undirected links (A cites B, C cites B). Let's search directed first.
            # If no path, let's treat the graph as undirected to find any connection pathway.
            try:
                path = nx.shortest_path(self._graph, source=start_id, target=end_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                if not hasattr(self, "_undirected_graph") or self._undirected_graph is None:
                    self._undirected_graph = self._graph.to_undirected()
                path = nx.shortest_path(self._undirected_graph, source=start_id, target=end_id)
                
            path_papers = []
            for pid in path:
                p_meta = self.get_paper(pid)
                if p_meta:
                    path_papers.append({
                        "id": p_meta["id"],
                        "title": p_meta["title"],
                        "year": p_meta["year"],
                        "authors": p_meta["authors"]
                    })
            return path_papers
        except Exception:
            return []

    def get_graph_data(self, focus_paper_ids: Optional[List[str]] = None, max_nodes: int = 150) -> Dict[str, Any]:
        """Return nodes and edges. If focus_paper_ids is provided, filter nodes in neighborhood."""
        if not self._graph or len(self._graph) == 0:
            return {"nodes": [], "links": []}

        # Determine subset of nodes to return
        nodes_to_include = set()
        if focus_paper_ids:
            # First, add all focus papers that exist in our graph.
            valid_focus = [pid for pid in focus_paper_ids if self._graph.has_node(pid)]
            nodes_to_include.update(valid_focus)
            
            # Next, find neighbors. To avoid blowing up the graph, we count how many focus papers
            # each neighbor connects to (cited by or cites).
            neighbor_counts = {}
            for pid in valid_focus:
                for neighbor in list(self._graph.successors(pid)) + list(self._graph.predecessors(pid)):
                    if neighbor not in nodes_to_include:
                        neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1
            
            # 1. Prioritize shared neighbors (co-citations, count >= 2) to show strong community connections
            shared_neighbors = [n for n, count in neighbor_counts.items() if count >= 2]
            nodes_to_include.update(shared_neighbors)
            
            # 2. Fill the remaining space up to max_nodes with top cited direct neighbors
            if len(nodes_to_include) < max_nodes:
                remaining_slots = max_nodes - len(nodes_to_include)
                sorted_neighbors = sorted(
                    [(n, self._graph.nodes[n].get("citation_count", 0)) for n in neighbor_counts if n not in nodes_to_include],
                    key=lambda x: x[1],
                    reverse=True
                )
                nodes_to_include.update([n for n, _ in sorted_neighbors[:remaining_slots]])
            
        else:
            # Select top max_nodes by citation count
            sorted_nodes = sorted(self._graph.nodes(data=True), key=lambda x: x[1].get('citation_count', 0), reverse=True)
            nodes_to_include = {node[0] for node in sorted_nodes[:max_nodes]}

        # Build list of nodes
        nodes = []
        cursor = self.conn.cursor()
        for node_id in nodes_to_include:
            # Get main topics for styling in graph
            cursor.execute("SELECT topic FROM paper_topics WHERE paper_id = ? LIMIT 1", (node_id,))
            topic_row = cursor.fetchone()
            topic = topic_row["topic"] if topic_row else "Other"

            node_data = self._graph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "title": node_data.get("title", ""),
                "year": node_data.get("year", 2026),
                "citation_count": node_data.get("citation_count", 0),
                "topic": topic
            })

        # Build links
        links = []
        for u, v in self._graph.edges():
            if u in nodes_to_include and v in nodes_to_include:
                links.append({
                    "source": u,
                    "target": v,
                    "type": "cites"
                })

        return {"nodes": nodes, "links": links}

    def _invalidate_cache(self) -> None:
        self._cached_metrics = None
        self._cache_time = 0.0

    def get_graph_metrics(self) -> Dict[str, Any]:
        if not self._graph or len(self._graph) == 0:
            return {}

        # Return cached metrics if still fresh
        if self._cached_metrics and (time.time() - self._cache_time) < self._metrics_ttl:
            return self._cached_metrics

        # 1. PageRank
        pr = nx.pagerank(self._graph)

        # 2. Betweenness centrality
        betweenness = nx.betweenness_centrality(self._graph)

        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        # 3. Louvain communities on undirected graph
        undirected_G = self._graph.to_undirected()
        community_map: Dict[str, int] = {}
        try:
            communities = nx.community.louvain_communities(undirected_G)
            for i, comm in enumerate(communities):
                for node_id in comm:
                    community_map[node_id] = i
        except Exception:
            pass

        # 4. Disconnected topic pairs — computed here once, cached with rest of metrics
        disconnected_pairs = self._compute_disconnected_pairs()

        # 5. Graph-level stats
        n_nodes = self._graph.number_of_nodes()
        n_edges = self._graph.number_of_edges()
        density = nx.density(self._graph)
        degrees = [d for _, d in self._graph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        components = nx.connected_components(undirected_G)
        n_components = sum(1 for _ in components)

        metrics = {
            "foundational_papers": [
                {"id": pid, "score": score, "title": self._graph.nodes[pid].get("title", "")}
                for pid, score in sorted_pr[:10]
            ],
            "bridge_papers": [
                {"id": pid, "score": score, "title": self._graph.nodes[pid].get("title", "")}
                for pid, score in sorted_betweenness[:10] if score > 0
            ],
            "communities": community_map,
            "pagerank_scores": pr,
            "betweenness_scores": betweenness,
            "disconnected_pairs": disconnected_pairs,
            "graph_stats": {
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_degree": avg_degree,
                "connected_components": n_components,
            },
        }

        self._cached_metrics = metrics
        self._cache_time = time.time()
        return metrics

    def _compute_disconnected_pairs(self) -> List[Dict[str, str]]:
        """
        Find taxonomy topic pairs with no citation path between them.
        Runs once per cache cycle (not on every query).
        """
        disconnected: List[Dict[str, str]] = []
        for i, t1 in enumerate(TAXONOMY):
            for t2 in TAXONOMY[i + 1:]:
                papers_t1 = self.get_papers_by_topic(t1, limit=3)
                papers_t2 = self.get_papers_by_topic(t2, limit=3)
                if not papers_t1 or not papers_t2:
                    continue
                has_path = False
                for p1 in papers_t1:
                    for p2 in papers_t2:
                        if self.get_citation_path(p1["id"], p2["id"]):
                            has_path = True
                            break
                    if has_path:
                        break
                if not has_path:
                    disconnected.append({"topic_a": t1, "topic_b": t2})
        return disconnected

    def get_papers_by_topic(self, topic_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT p.* FROM papers p
        JOIN paper_topics pt ON p.id = pt.paper_id
        WHERE pt.topic = ?
        ORDER BY p.citation_count DESC
        LIMIT ?
        """, (topic_name, limit))
        
        papers = []
        for row in cursor.fetchall():
            p = dict(row)
            p["authors"] = json.loads(p["authors"]) if p["authors"] else []
            p["section_headers"] = json.loads(p["section_headers"]) if p["section_headers"] else []
            p.pop("embedding", None)
            papers.append(p)
        return papers

    def get_all_topics(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT topic FROM paper_topics ORDER BY topic")
        return [row["topic"] for row in cursor.fetchall()]
