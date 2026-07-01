import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.state import AgentState
from backend.agents.router import get_llm
from backend.database.manager import get_db
from backend.constants import TOPIC_ALIASES

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[Retriever] Loading SentenceTransformer('all-MiniLM-L6-v2') on CPU...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _embedding_model


def retriever_node(state: AgentState) -> dict:
    """
    Dynamic retriever — two-pass arXiv search + Semantic Scholar enrichment.

    Pass A: primary keywords (specific to query)
    Pass B: broader fallback keywords to fill out the paper set
    Merges both passes, enriches via S2, caches, and returns up to 25 papers.
    """
    from backend.agents.arxiv_fetcher import search_arxiv, search_semantic_scholar, enrich_and_cache

    query = state["query"]
    plan  = state.get("research_plan", "")

    # ── Step 1: LLM keyword extraction ────────────────────────────────────
    system_prompt = """You are a Search Intent Extractor for an academic research assistant.
Analyze the user's research query and extract TWO sets of keywords for arXiv search:

1. "keywords": 3-5 specific 1-2 word terms directly related to the query topic.
   Good: ["web agents", "tool use", "LLM planning", "evaluation"]
   Bad:  ["web agents limitations and challenges", "new research directions"]

2. "broad_keywords": 2-3 general 1-word terms that broaden the search.
   Example: ["agents", "LLM", "autonomous"]

3. "topic": exactly one of:
   "Browser Agents", "Web Agents", "Agent Evaluation", "Tool Use for LLMs",
   "Autonomous Agents", "Multi-Agent Systems", "Human-AI Interaction"
   — or null.

Output JSON:
{
  "keywords": ["term1", "term2", ...],
  "broad_keywords": ["term1", "term2"],
  "topic": "topic name" or null
}"""

    llm      = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}\nResearch plan: {plan}"),
    ]

    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        extracts = json.loads(response.content)
    except Exception as e:
        print(f"[Retriever] Keyword extraction failed: {e}. Using raw query.")
        extracts = {"keywords": [query], "broad_keywords": [], "topic": None}

    keywords:       list = extracts.get("keywords", [query])
    broad_keywords: list = extracts.get("broad_keywords", [])
    raw_topic             = extracts.get("topic")
    topic                 = TOPIC_ALIASES.get(raw_topic, raw_topic) if raw_topic else None

    print(f"[Retriever] Keywords: {keywords} | Broad: {broad_keywords} | Topic: {topic}")

    model          = get_embedding_model()
    query_text     = f"{query} " + " ".join(keywords)
    query_embedding = model.encode(query_text).tolist()

    db = get_db()

    # ── Step 2A: arXiv search (good for very recent papers) ──────────────
    arxiv_a = search_arxiv(keywords, max_results=200)

    # ── Step 2B: arXiv second pass with broader keywords ─────────────────
    arxiv_b: list = []
    if broad_keywords:
        combined_broad = broad_keywords + ([keywords[0]] if keywords else [])
        arxiv_b = search_arxiv(combined_broad, max_results=150)

    # ── Step 2C: arXiv raw user query search (highly robust) ─────────────
    arxiv_raw = search_arxiv([query], max_results=200)

    # ── Step 2D: Semantic Scholar direct search (best for citation data) ──
    # Query with both the raw user query (for best recall) and keyword-based query
    s2_papers_raw = search_semantic_scholar(query, max_results=100)
    s2_query  = " ".join(keywords[:3])
    s2_papers_kw  = search_semantic_scholar(s2_query, max_results=100)
    
    # Deduplicate Semantic Scholar papers first
    s2_papers = []
    seen_s2_ids_pool = set()
    for p in s2_papers_raw + s2_papers_kw:
        if p["id"] not in seen_s2_ids_pool:
            seen_s2_ids_pool.add(p["id"])
            s2_papers.append(p)

    # Deduplicate: prefer S2-direct (has full data); mark arXiv-only ones
    seen_s2_ids:    set = {p["id"]       for p in s2_papers}
    seen_arxiv_ids: set = {p["arxiv_id"] for p in s2_papers if p.get("arxiv_id")}

    arXiv_only = [
        p for p in (arxiv_a + arxiv_b + arxiv_raw)
        if p["arxiv_id"] not in seen_arxiv_ids
    ]
    # Deduplicate within arXiv results
    seen_ax: set = set()
    arXiv_deduped = []
    for p in arXiv_only:
        if p["arxiv_id"] not in seen_ax:
            seen_ax.add(p["arxiv_id"])
            arXiv_deduped.append(p)

    combined_pool = s2_papers + arXiv_deduped
    print(f"[Retriever] Pool: {len(s2_papers)} S2-direct + {len(arXiv_deduped)} arXiv-only = {len(combined_pool)}")

    # ── Step 3: Enrich + cache ────────────────────────────────────────────
    live_papers = enrich_and_cache(combined_pool, db, model)
    print(f"[Retriever] After enrich/cache: {len(live_papers)} papers")

    # ── Step 4: Semantic DB search (surfaces relevant cached papers) ───────
    cached_results = db.search_papers(
        query_embedding=query_embedding,
        keywords=keywords,
        topic=topic,
        limit=100,
    )

    # ── Step 5: Merge & deduplicate ────────────────────────────────────────
    seen_ids: set = set()
    merged:   list = []

    for p in live_papers:
        pid = p.get("id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            merged.append(p)

    for p in cached_results:
        pid = p.get("id")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            merged.append(p)

    # ── Step 6: Score & rank ───────────────────────────────────────────────
    q_vec  = np.array(query_embedding, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_vec)) or 1.0

    def _score(p: dict) -> float:
        title_lower = p.get("title", "").lower()
        kw_bonus    = sum(0.12 for kw in keywords if kw.lower() in title_lower)
        cite_bonus  = min(0.3, (p.get("citation_count", 0) / 500))
        return p.get("final_score", 0.5) + kw_bonus + cite_bonus

    merged.sort(key=_score, reverse=True)
    final_results = merged[:500]

    print(f"[Retriever] Returning {len(final_results)} papers to pipeline")
    return {"retrieved_papers": final_results}
