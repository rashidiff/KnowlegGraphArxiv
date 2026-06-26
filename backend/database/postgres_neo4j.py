import json
import numpy as np
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from backend.database.base import BaseDatabase

class PostgresNeo4jDB(BaseDatabase):
    def __init__(self, postgres_config: dict, neo4j_config: dict):
        self.pg_config = postgres_config
        self.neo4j_config = neo4j_config
        self.pg_conn = None
        self.neo4j_driver = None
        self._connect()
        self.init_db()

    def _connect(self) -> None:
        # Postgres connection
        self.pg_conn = psycopg2.connect(
            user=self.pg_config["user"],
            password=self.pg_config["password"],
            host=self.pg_config["host"],
            port=self.pg_config["port"],
            database=self.pg_config["dbname"]
        )
        self.pg_conn.autocommit = True
        
        # Enable pgvector extension
        with self.pg_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_config["uri"],
            auth=(self.neo4j_config["user"], self.neo4j_config["password"])
        )

    def init_db(self) -> None:
        # 1. Initialize Postgres Tables
        with self.pg_conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id VARCHAR(100) PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                authors JSONB,
                year INTEGER,
                venue TEXT,
                citation_count INTEGER DEFAULT 0,
                venue_quality REAL DEFAULT 0.0,
                embedding VECTOR(384),
                intro_summary TEXT,
                conclusion_summary TEXT,
                section_headers JSONB
            );
            """)
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_topics (
                paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
                topic VARCHAR(100),
                PRIMARY KEY (paper_id, topic)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_entities (
                id SERIAL PRIMARY KEY,
                paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
                type VARCHAR(50),
                value TEXT
            );
            """)

        # 2. Initialize Neo4j constraints
        with self.neo4j_driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE")

    def insert_papers(self, papers: List[Dict[str, Any]]) -> None:
        # 1. Insert in Postgres
        with self.pg_conn.cursor() as cur:
            for p in papers:
                authors_json = json.dumps(p.get("authors", []))
                headers_json = json.dumps(p.get("section_headers", []))
                
                # Check embedding
                emb = p.get("embedding")
                if emb is not None:
                    # pgvector requires format [1.2, 3.4, ...]
                    emb_str = "[" + ",".join(map(str, emb)) + "]"
                else:
                    emb_str = None

                cur.execute("""
                INSERT INTO papers (
                    id, title, abstract, authors, year, venue, 
                    citation_count, venue_quality, embedding, 
                    intro_summary, conclusion_summary, section_headers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    authors = EXCLUDED.authors,
                    year = EXCLUDED.year,
                    venue = EXCLUDED.venue,
                    citation_count = EXCLUDED.citation_count,
                    venue_quality = EXCLUDED.venue_quality,
                    embedding = COALESCE(EXCLUDED.embedding, papers.embedding),
                    intro_summary = EXCLUDED.intro_summary,
                    conclusion_summary = EXCLUDED.conclusion_summary,
                    section_headers = EXCLUDED.section_headers;
                """, (
                    p["id"], p["title"], p.get("abstract", ""), authors_json,
                    p.get("year", 2026), p.get("venue", "Unknown"), p.get("citation_count", 0),
                    p.get("venue_quality", 0.0), emb_str, p.get("intro_summary"),
                    p.get("conclusion_summary"), headers_json
                ))

                # Topics
                topics = p.get("topics", [])
                for t in topics:
                    cur.execute("""
                    INSERT INTO paper_topics (paper_id, topic)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                    """, (p["id"], t))

                # Entities
                entities = p.get("entities", [])
                for ent in entities:
                    cur.execute("""
                    INSERT INTO paper_entities (paper_id, type, value)
                    VALUES (%s, %s, %s);
                    """, (p["id"], ent["type"], ent["value"]))

                # References stubs insertion in Postgres
                references = p.get("references", [])
                for ref_id in references:
                    cur.execute("""
                    INSERT INTO papers (id, title, abstract, authors, year, venue, citation_count, venue_quality)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """, (ref_id, "Cited Reference", "", "[]", p.get("year", 2024) - 2, "Unknown", 0, 0.0))

        # 2. Insert in Neo4j
        with self.neo4j_driver.session() as session:
            for p in papers:
                # Merge Paper
                session.run("""
                MERGE (paper:Paper {id: $id})
                SET paper.title = $title,
                    paper.year = $year,
                    paper.venue = $venue,
                    paper.citation_count = $citation_count
                """, id=p["id"], title=p["title"], year=p.get("year", 2026), 
                     venue=p.get("venue", "Unknown"), citation_count=p.get("citation_count", 0))

                # Merge Authors & Relationships
                authors = p.get("authors", [])
                for auth in authors:
                    session.run("""
                    MATCH (p:Paper {id: $pid})
                    MERGE (a:Author {name: $name})
                    MERGE (p)-[:WRITTEN_BY]->(a)
                    """, pid=p["id"], name=auth)

                # Merge Topics & Relationships
                topics = p.get("topics", [])
                for t in topics:
                    session.run("""
                    MATCH (p:Paper {id: $pid})
                    MERGE (topic:Topic {name: $name})
                    MERGE (p)-[:BELONGS_TO]->(topic)
                    """, pid=p["id"], name=t)

                # Merge Entities & Relationships
                entities = p.get("entities", [])
                for ent in entities:
                    ent_type = ent["type"].capitalize() # Concept, Method, Dataset
                    ent_val = ent["value"]
                    # Cypher parameter dynamic node label can be handled or generic entity relationship
                    if ent_type in ["Concept", "Method", "Dataset"]:
                        session.run(f"""
                        MATCH (p:Paper {{id: $pid}})
                        MERGE (e:{ent_type} {{name: $value}})
                        MERGE (p)-[:MENTIONS]->(e)
                        """, pid=p["id"], value=ent_val)

            # References link pass
            for p in papers:
                references = p.get("references", [])
                for ref_id in references:
                    session.run("""
                    MATCH (p1:Paper {id: $pid})
                    MERGE (p2:Paper {id: $ref_id})
                    ON CREATE SET p2.title = "Cited Reference", p2.year = $default_year, p2.venue = "Unknown", p2.citation_count = 0
                    MERGE (p1)-[:CITES]->(p2)
                    """, pid=p["id"], ref_id=ref_id, default_year=p.get("year", 2024) - 2)

    def search_papers(self, 
                      query_embedding: List[float], 
                      keywords: List[str], 
                      topic: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        
        # Load PageRank from Neo4j (using simple NetworkX bridge to avoid GDS extension requirement)
        pageranks = {}
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (p1:Paper)-[:CITES]->(p2:Paper) RETURN p1.id AS src, p2.id AS tgt")
                G = nx.DiGraph()
                for r in result:
                    G.add_edge(r["src"], r["tgt"])
                
                # Fetch all papers to populate orphan nodes
                result_nodes = session.run("MATCH (p:Paper) RETURN p.id AS id")
                for r in result_nodes:
                    if not G.has_node(r["id"]):
                        G.add_node(r["id"])
                        
                pageranks = nx.pagerank(G) if len(G) > 0 else {}
        except Exception:
            pass

        # Postgres hybrid search query
        emb_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        query_sql = """
        SELECT p.*,
               1 - (p.embedding <=> %s::vector) AS semantic_sim
        FROM papers p
        WHERE p.embedding IS NOT NULL
        """
        
        params = [emb_str]
        
        if topic:
            query_sql = """
            SELECT p.*,
                   1 - (p.embedding <=> %s::vector) AS semantic_sim
            FROM papers p
            JOIN paper_topics pt ON p.id = pt.paper_id
            WHERE pt.topic = %s AND p.embedding IS NOT NULL
            """
            params.append(topic)

        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()

        if not rows:
            return []

        all_papers = [dict(row) for row in rows]
        
        # Norm statistics
        years = [p["year"] for p in all_papers if p["year"] is not None]
        min_year = min(years) if years else 2000
        max_year = max(years) if years else 2026
        year_range = max_year - min_year if max_year > min_year else 1

        citation_counts = [p["citation_count"] for p in all_papers if p["citation_count"] is not None]
        max_citations = max(citation_counts) if citation_counts else 1
        max_citations_log = np.log1p(max_citations)

        scored_papers = []
        for p in all_papers:
            similarity = p["semantic_sim"] or 0.0
            # Normalize to [0,1]
            similarity = (similarity + 1.0) / 2.0

            # Keywords search bonus
            if keywords:
                title_lower = p["title"].lower()
                abstract_lower = (p["abstract"] or "").lower()
                matches = sum(1 for kw in keywords if kw.lower() in title_lower or kw.lower() in abstract_lower)
                keyword_score = matches / len(keywords)
                similarity = 0.7 * similarity + 0.3 * keyword_score

            # Citation score
            citations = p["citation_count"] or 0
            citation_score = float(np.log1p(citations) / max_citations_log) if max_citations_log > 0 else 0.0

            # Centrality
            centrality = pageranks.get(p["id"], 0.0)
            max_centrality = max(pageranks.values()) if pageranks else 1.0
            normalized_centrality = centrality / max_centrality if max_centrality > 0 else 0.0

            # Recency
            year = p["year"] or min_year
            recency = float((year - min_year) / year_range) if year_range > 0 else 1.0

            # Venue quality
            venue_quality = p["venue_quality"] or 0.0

            # Final rank (gated by semantic similarity to filter out irrelevant highly-cited papers)
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

            p["final_score"] = final_score
            p["graph_centrality"] = normalized_centrality
            p["semantic_similarity"] = similarity
            
            p.pop("embedding", None)
            scored_papers.append(p)

        scored_papers.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        # Fetch from Postgres
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            if not row:
                return None
            p = dict(row)

        # Get additional entities, topics, citations from postgres or neo4j
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT topic FROM paper_topics WHERE paper_id = %s", (paper_id,))
            p["topics"] = [r[0] for r in cur.fetchall()]

            cur.execute("SELECT type, value FROM paper_entities WHERE paper_id = %s", (paper_id,))
            p["entities"] = [{"type": r[0], "value": r[1]} for r in cur.fetchall()]

        # Query citations from Neo4j
        with self.neo4j_driver.session() as session:
            # References
            ref_result = session.run("""
            MATCH (p:Paper {id: $pid})-[:CITES]->(ref:Paper)
            RETURN ref.id AS id
            """, pid=paper_id)
            p["references"] = [r["id"] for r in ref_result]

            # Cited by
            cited_result = session.run("""
            MATCH (cited:Paper)-[:CITES]->(p:Paper {id: $pid})
            RETURN cited.id AS id
            """, pid=paper_id)
            p["citations"] = [r["id"] for r in cited_result]

        return p

    def get_citation_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        with self.neo4j_driver.session() as session:
            # Find shortest path ignoring direction of cites
            query = """
            MATCH p=shortestPath((p1:Paper {id: $start})-[r:CITES*..10]-(p2:Paper {id: $end}))
            RETURN [n in nodes(p) | {id: n.id, title: n.title, year: n.year}] AS path
            """
            result = session.run(query, start=start_id, end=end_id)
            record = result.single()
            if record and record["path"]:
                return record["path"]
            return []

    def get_graph_data(self, focus_paper_ids: Optional[List[str]] = None, max_nodes: int = 150) -> Dict[str, Any]:
        with self.neo4j_driver.session() as session:
            if focus_paper_ids:
                # Retrieve all citation paths directly connected to focus papers
                # This fetches nodes and relationships for focus papers and their neighbors.
                query = """
                MATCH (p1:Paper)-[:CITES]->(p2:Paper)
                WHERE p1.id IN $ids OR p2.id IN $ids
                RETURN p1.id AS src, p1.title AS src_title, p1.year AS src_year, p1.citation_count AS src_citations,
                       p2.id AS tgt, p2.title AS tgt_title, p2.year AS tgt_year, p2.citation_count AS tgt_citations
                """
                result = session.run(query, ids=focus_paper_ids)
                
                # Build local graph representation in Python
                local_G = nx.DiGraph()
                nodes_meta = {}
                
                # Initialize focus papers (even if they have no citations in DB yet)
                # To get their metadata, we query Neo4j for focus papers first
                focus_meta_res = session.run("""
                MATCH (p:Paper) WHERE p.id IN $ids
                RETURN p.id AS id, p.title AS title, p.year AS year, p.citation_count AS citation_count
                """, ids=focus_paper_ids)
                for r in focus_meta_res:
                    pid = r["id"]
                    nodes_meta[pid] = {
                        "id": pid,
                        "title": r["title"] or "",
                        "year": r["year"] or 2026,
                        "citation_count": r["citation_count"] or 0
                    }
                    local_G.add_node(pid)

                # Add nodes and edges from citations
                for r in result:
                    src, tgt = r["src"], r["tgt"]
                    local_G.add_edge(src, tgt)
                    
                    if src not in nodes_meta:
                        nodes_meta[src] = {
                            "id": src,
                            "title": r["src_title"] or "",
                            "year": r["src_year"] or 2026,
                            "citation_count": r["src_citations"] or 0
                        }
                    if tgt not in nodes_meta:
                        nodes_meta[tgt] = {
                            "id": tgt,
                            "title": r["tgt_title"] or "",
                            "year": r["tgt_year"] or 2026,
                            "citation_count": r["tgt_citations"] or 0
                        }

                # Node Selection Logic (same as SQLite):
                # 1. Start with valid focus papers
                nodes_to_include = set(pid for pid in focus_paper_ids if local_G.has_node(pid))
                
                # 2. Count neighbor connectivity
                neighbor_counts = {}
                for pid in nodes_to_include:
                    neighbors = list(local_G.successors(pid)) + list(local_G.predecessors(pid))
                    for neighbor in neighbors:
                        if neighbor not in nodes_to_include:
                            neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1
                            
                # 3. Add shared neighbors (co-citations, count >= 2)
                shared_neighbors = [n for n, count in neighbor_counts.items() if count >= 2]
                nodes_to_include.update(shared_neighbors)
                
                # 4. Fill remaining capacity with top cited neighbors
                if len(nodes_to_include) < max_nodes:
                    remaining_slots = max_nodes - len(nodes_to_include)
                    sorted_neighbors = sorted(
                        [(n, nodes_meta[n].get("citation_count", 0)) for n in neighbor_counts if n not in nodes_to_include],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    nodes_to_include.update([n for n, _ in sorted_neighbors[:remaining_slots]])

                # Build nodes output and fetch topics
                topics_res = session.run("""
                MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic)
                WHERE p.id IN $ids
                RETURN p.id AS pid, t.name AS topic
                """, ids=list(nodes_to_include))
                topic_map = {r["pid"]: r["topic"] for r in topics_res}
                
                nodes = []
                for nid in nodes_to_include:
                    meta = nodes_meta.get(nid, {"id": nid, "title": "Cited Reference", "year": 2024, "citation_count": 0})
                    meta["topic"] = topic_map.get(nid, "Other")
                    nodes.append(meta)
                    
                # Build links output
                links = []
                for u, v in local_G.edges():
                    if u in nodes_to_include and v in nodes_to_include:
                        links.append({"source": u, "target": v, "type": "cites"})
                        
                return {"nodes": nodes, "links": links}
            else:
                # Retrieve top max_nodes by citation count
                query = """
                MATCH (p:Paper)
                WITH p ORDER BY p.citation_count DESC LIMIT $limit
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(t:Topic)
                RETURN p.id AS id, p.title AS title, p.year AS year, p.citation_count AS citation_count, t.name AS topic
                """
                nodes_result = session.run(query, limit=max_nodes)
                nodes = []
                ids = []
                for r in nodes_result:
                    nodes.append({
                        "id": r["id"],
                        "title": r["title"] or "",
                        "year": r["year"] or 2026,
                        "citation_count": r["citation_count"] or 0,
                        "topic": r["topic"] or "Other"
                    })
                    ids.append(r["id"])
                    
                # Get links between these nodes
                links_query = """
                MATCH (p1:Paper)-[:CITES]->(p2:Paper)
                WHERE p1.id IN $ids AND p2.id IN $ids
                RETURN p1.id AS src, p2.id AS tgt
                """
                links_result = session.run(links_query, ids=ids)
                links = [{"source": r["src"], "target": r["tgt"], "type": "cites"} for r in links_result]
                
                return {"nodes": nodes, "links": links}

    def get_graph_metrics(self) -> Dict[str, Any]:
        # Compute metrics by loading Neo4j Graph into NetworkX
        # Louvain and PageRank are computed in Python for maximum compatibility.
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (p1:Paper)-[:CITES]->(p2:Paper) RETURN p1.id AS src, p2.id AS tgt")
            G = nx.DiGraph()
            for r in result:
                G.add_edge(r["src"], r["tgt"])
                
            nodes_res = session.run("MATCH (p:Paper) RETURN p.id AS id, p.title AS title")
            titles = {}
            for r in nodes_res:
                titles[r["id"]] = r["title"] or ""
                if not G.has_node(r["id"]):
                    G.add_node(r["id"])

        if len(G) == 0:
            return {}

        pr = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        # louvain communities
        undirected_G = G.to_undirected()
        try:
            communities = nx.community.louvain_communities(undirected_G)
            community_map = {}
            for i, comm in enumerate(communities):
                for node_id in comm:
                    community_map[node_id] = i
        except Exception:
            community_map = {}

        return {
            "foundational_papers": [
                {"id": pid, "score": score, "title": titles.get(pid, "")}
                for pid, score in sorted_pr[:10]
            ],
            "bridge_papers": [
                {"id": pid, "score": score, "title": titles.get(pid, "")}
                for pid, score in sorted_betweenness[:10] if score > 0
            ],
            "communities": community_map,
            "pagerank_scores": pr,
            "betweenness_scores": betweenness
        }

    def get_papers_by_topic(self, topic_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
            SELECT p.* FROM papers p
            JOIN paper_topics pt ON p.id = pt.paper_id
            WHERE pt.topic = %s
            ORDER BY p.citation_count DESC
            LIMIT %s
            """, (topic_name, limit))
            rows = cur.fetchall()
            
        papers = []
        for row in rows:
            p = dict(row)
            p["authors"] = json.loads(p["authors"]) if p["authors"] else []
            p["section_headers"] = json.loads(p["section_headers"]) if p["section_headers"] else []
            p.pop("embedding", None)
            papers.append(p)
        return papers

    def get_all_topics(self) -> List[str]:
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT DISTINCT topic FROM paper_topics ORDER BY topic")
            return [r[0] for r in cur.fetchall()]
