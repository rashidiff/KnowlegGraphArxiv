import os
import sys
import json
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Adjust path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.manager import get_db
from backend.agents.graph import build_workflow
from backend.agents.retriever import get_embedding_model

# Load environment
load_dotenv()

app = FastAPI(title="Agentic Research Paper Knowledge Graph Navigator API")

# Setup CORS so the Next.js frontend can communicate with the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []

class ClarifyRequest(BaseModel):
    query: str
    question: str
    answer: str
    history: List[Dict[str, Any]] = []
    clarification_answers: List[Dict[str, str]] = []

# Instantiate the compiled LangGraph workflow
graph_workflow = build_workflow()

# Track background refresh state
_refresh_running = False
_refresh_result: Dict[str, Any] = {}

@app.get("/")
def read_root():
    return {"status": "running", "project": "Research Paper Navigator"}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Primary chat endpoint. Runs the LangGraph research navigator workflow.
    """
    initial_state = {
        "query": req.query,
        "messages": [],
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_answers": [],
        "research_plan": None,
        "retrieved_papers": [],
        "graph_context": {},
        "reading_path": None,
        "analysis": None,
        "research_ideas": None,
        "verification_report": {},
        "final_response": None
    }
    
    try:
        final_state = graph_workflow.invoke(initial_state)

        def _safe_paper(p: dict) -> dict:
            return {k: v for k, v in p.items() if k not in ("embedding",)}

        return {
            "clarification_needed": final_state.get("clarification_needed", False),
            "clarification_question": final_state.get("clarification_question"),
            "clarification_answers": final_state.get("clarification_answers", []),
            "final_response": final_state.get("final_response"),
            "retrieved_papers": [_safe_paper(p) for p in final_state.get("retrieved_papers", [])[:100]],
            "graph_context": final_state.get("graph_context", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")

@app.post("/api/chat/clarify")
async def chat_clarify_endpoint(req: ClarifyRequest):
    """
    Resumes the chat workflow with answers to clarifying questions.
    """
    # Append the new answer to history
    updated_clarifications = req.clarification_answers + [
        {"question": req.question, "answer": req.answer}
    ]
    
    # We bypass the router ambiguity checking by providing a flag
    initial_state = {
        "query": req.query,
        "messages": [],
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_answers": updated_clarifications,
        "research_plan": None,
        "retrieved_papers": [],
        "graph_context": {},
        "reading_path": None,
        "analysis": None,
        "research_ideas": None,
        "verification_report": {},
        "final_response": None
    }
    
    try:
        final_state = graph_workflow.invoke(initial_state)

        def _safe_paper(p: dict) -> dict:
            return {k: v for k, v in p.items() if k not in ("embedding",)}

        return {
            "clarification_needed": final_state.get("clarification_needed", False),
            "clarification_question": final_state.get("clarification_question"),
            "clarification_answers": final_state.get("clarification_answers", []),
            "final_response": final_state.get("final_response"),
            "retrieved_papers": [_safe_paper(p) for p in final_state.get("retrieved_papers", [])[:100]],
            "graph_context": final_state.get("graph_context", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")

@app.get("/api/papers/{paper_id}")
async def get_paper_endpoint(paper_id: str):
    """
    Get full metadata details of a specific paper.
    """
    db = get_db()
    paper = db.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@app.get("/api/graph/explore")
async def explore_graph(focus_id: Optional[str] = None):
    """
    Returns nodes and links representation of the citation network for the UI.
    """
    db = get_db()
    focus_ids = [focus_id] if focus_id else None
    return db.get_graph_data(focus_paper_ids=focus_ids, max_nodes=150)

@app.get("/api/corpus/audit")
async def corpus_audit():
    """Returns statistics and quality metrics about the ingested research corpus."""
    db = get_db()

    # 1. Topic distribution via active DB
    topic_distribution: Dict[str, int] = {}
    total_papers = 0
    try:
        topics = db.get_all_topics()
        for t in topics:
            papers = db.get_papers_by_topic(t, limit=10000)
            topic_distribution[t] = len(papers)
            total_papers += len(papers)
        # total_papers via distinct count (papers can have multiple topics)
        # Use a direct count from search instead
    except Exception:
        pass

    # Better total via search_papers with empty embedding
    try:
        from backend.agents.retriever import get_embedding_model
        model = get_embedding_model()
        zero_emb = [0.0] * 384
        all_p = db.search_papers(query_embedding=zero_emb, keywords=[], limit=10000)
        total_papers = len(all_p)
    except Exception:
        pass

    # 2. Graph metrics
    metrics = db.get_graph_metrics() or {}
    top_pr = metrics.get("foundational_papers", [])[:10]
    top_bridge = metrics.get("bridge_papers", [])[:10]
    graph_stats = metrics.get("graph_stats", {})

    # 3. Corpus metadata file
    excluded_count = 0
    quality_score = 1.0
    meta_path = "data/corpus_metadata.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                excluded_count = meta.get("excluded_papers_count", 0)
                quality_score = meta.get("quality_score", 1.0)
        except Exception:
            pass

    return {
        "total_papers": total_papers,
        "topic_distribution": topic_distribution,
        "top_pagerank_papers": top_pr,
        "top_bridge_papers": top_bridge,
        "excluded_papers_count": excluded_count,
        "quality_score": quality_score,
        "graph_stats": graph_stats,
    }


@app.get("/api/graph/health")
async def graph_health():
    """Detailed graph integrity report — nodes, edges, density, component breakdown."""
    db = get_db()
    metrics = db.get_graph_metrics() or {}
    stats = metrics.get("graph_stats", {})

    # Edge coverage: what fraction of stored citation rows have both endpoints in DB
    edge_coverage = None
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect("data/research_navigator.db")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM citations")
        total_rows = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM citations c
            JOIN papers p1 ON c.source_id = p1.id
            JOIN papers p2 ON c.target_id = p2.id
        """)
        valid_rows = cur.fetchone()[0]
        conn.close()
        edge_coverage = {
            "total_citation_rows": total_rows,
            "valid_edges": valid_rows,
            "dangling_references": total_rows - valid_rows,
            "coverage_pct": round(valid_rows / total_rows * 100, 2) if total_rows else 0,
        }
    except Exception:
        pass

    return {
        "graph_stats": stats,
        "edge_coverage": edge_coverage,
        "foundational_papers": metrics.get("foundational_papers", [])[:5],
        "bridge_papers": metrics.get("bridge_papers", [])[:5],
        "disconnected_topic_pairs": metrics.get("disconnected_pairs", []),
    }


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    topic: str = Form("LLM Agents"),
    force: bool = Form(False),
):
    """
    Uploads a PDF paper, extracts metadata via LLM, scores relevance,
    generates embeddings, and adds to the corpus.
    Returns a relevance_score and warning if the paper appears off-topic.
    Set force=true to index regardless of relevance score.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    from backend.scripts.seed_data_v2 import compute_relevance_and_classify, extract_entities

    temp_dir = "data/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        from pypdf import PdfReader
        reader = PdfReader(temp_path)

        extracted_text = ""
        for i in range(min(3, len(reader.pages))):
            extracted_text += reader.pages[i].extract_text() or ""

        from backend.agents.router import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """You are a Scientific Document Parser.
Analyze the provided text of a research paper's first pages and extract the following metadata:
1. "title" (exact title)
2. "abstract" (a concise 3-4 sentence summary of the paper's main contributions)
3. "authors" (list of author strings)
4. "year" (publication year as integer, default to 2026 if unknown)
5. "venue" (publication venue name, default to "Self Uploaded")
6. "concepts" (list of 2-3 key terms/concepts mentioned)

You must output a JSON object in this format:
{
  "title": "...",
  "abstract": "...",
  "authors": ["Author 1", "Author 2"],
  "year": 2026,
  "venue": "...",
  "concepts": ["Concept 1", "Concept 2"]
}
"""
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=extracted_text[:4000]),
        ]

        try:
            response = llm.invoke(messages, response_format={"type": "json_object"})
            meta = json.loads(response.content)
        except Exception as parse_err:
            # LLM couldn't parse the document (e.g. math-heavy PDF with no plain text)
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract metadata from PDF. The document may be image-based or contain non-standard formatting. Error: {parse_err}",
            )

        title = meta.get("title", "Unknown Title")
        abstract = meta.get("abstract", "")

        # ── Relevance check ──────────────────────────────────────────────
        relevance_score, detected_topics = compute_relevance_and_classify(title, abstract)
        is_off_topic = relevance_score < 0.3 or (len(detected_topics) == 1 and "Other" in detected_topics)

        if is_off_topic and not force:
            os.remove(temp_path)
            return {
                "warning": "off_topic",
                "message": (
                    f"Paper '{title}' appears unrelated to LLM agents / autonomous systems "
                    f"(relevance score: {relevance_score:.2f}). "
                    "Submit again with force=true to index it anyway."
                ),
                "relevance_score": round(relevance_score, 3),
                "detected_topics": detected_topics,
                "metadata": meta,
            }

        # Use detected topics if available, else fall back to the form value
        final_topics = detected_topics if detected_topics and "Other" not in detected_topics else [topic]

        # ── Section headers ───────────────────────────────────────────────
        section_headers = []
        for page in reader.pages[:10]:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if len(line) < 40 and line and line[0].isdigit() and ". " in line:
                    section_headers.append(line)

        paper_id = "upload_" + "".join(e for e in title if e.isalnum())[:20]

        model = get_embedding_model()
        emb_text = f"{title}. {abstract}"
        embedding = model.encode(emb_text).tolist()

        paper_data = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": meta.get("authors", []),
            "year": meta.get("year", 2026),
            "venue": meta.get("venue", "Self Uploaded"),
            "citation_count": 0,
            "venue_quality": 0.3,
            "embedding": embedding,
            "intro_summary": extracted_text[:1500],
            "conclusion_summary": extracted_text[-1500:] if len(reader.pages) > 1 else "",
            "section_headers": section_headers,
            "topics": final_topics,
            "entities": extract_entities(title, abstract) + [
                {"type": "concept", "value": c} for c in meta.get("concepts", [])
            ],
        }

        db = get_db()
        db.insert_papers([paper_data])
        os.remove(temp_path)

        return {
            "message": "Paper successfully uploaded and cataloged.",
            "paper_id": paper_id,
            "relevance_score": round(relevance_score, 3),
            "detected_topics": final_topics,
            "metadata": meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


# ── Corpus reset ─────────────────────────────────────────────────────────────

@app.delete("/api/corpus/reset")
async def reset_corpus():
    """
    Drop all cached papers and citations, then re-initialise the DB schema.
    Use this to clear the old static seed corpus so the system starts fresh
    and builds its knowledge graph dynamically from arXiv queries.
    """
    import sqlite3 as _sqlite3
    db_path = "data/research_navigator.db"
    try:
        conn = _sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM citations")
        conn.execute("DELETE FROM paper_entities")
        conn.execute("DELETE FROM paper_topics")
        conn.execute("DELETE FROM papers")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        conn.close()

        # Reload the in-memory graph so it reflects the empty DB
        db = get_db()
        db._load_graph()
        db._invalidate_cache()

        # Clear corpus metadata file
        meta_path = "data/corpus_metadata.json"
        if os.path.exists(meta_path):
            os.remove(meta_path)

        return {"status": "ok", "message": "Corpus cleared. The graph will now build dynamically from arXiv queries."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


# ── Dynamic arXiv refresh endpoints ─────────────────────────────────────────

def _run_refresh_task() -> None:
    global _refresh_running, _refresh_result
    try:
        from backend.scripts.dynamic_updater import run_update
        _refresh_result = run_update()
    except Exception as e:
        _refresh_result = {"status": "error", "error": str(e)}
    finally:
        _refresh_running = False


@app.post("/api/refresh")
async def refresh_corpus(background_tasks: BackgroundTasks):
    """Trigger an incremental arXiv refresh in the background."""
    global _refresh_running
    if _refresh_running:
        return {"status": "already_running", "message": "A refresh is already in progress."}
    _refresh_running = True
    background_tasks.add_task(_run_refresh_task)
    return {"status": "started", "message": "arXiv refresh started. Poll /api/corpus/status for progress."}


@app.get("/api/corpus/status")
async def corpus_status():
    """Returns corpus metadata: last update time, paper count, refresh state."""
    db = get_db()
    paper_count = 0
    try:
        zero_emb = [0.0] * 384
        papers = db.search_papers(query_embedding=zero_emb, keywords=[], limit=100000)
        paper_count = len(papers)
    except Exception:
        pass

    meta: Dict[str, Any] = {}
    if os.path.exists("data/corpus_metadata.json"):
        try:
            with open("data/corpus_metadata.json") as f:
                meta = json.load(f)
        except Exception:
            pass

    return {
        "paper_count": paper_count,
        "last_updated": meta.get("last_updated"),
        "last_update_added": meta.get("last_update_added"),
        "last_update_excluded": meta.get("last_update_excluded"),
        "refresh_running": _refresh_running,
        "last_refresh_result": _refresh_result if not _refresh_running else None,
    }
