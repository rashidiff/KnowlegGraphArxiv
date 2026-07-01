"""
Live arXiv search + Semantic Scholar citation enrichment + SQLite caching.
Semantic Scholar (S2) has far better coverage of recent CS/AI papers than OpenAlex.
"""
import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

ATOM_NS     = "http://www.w3.org/2005/Atom"
ARXIV_API   = "http://export.arxiv.org/api/query"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
OA_API      = "https://api.openalex.org/works"
OA_EMAIL    = "agent@research-navigator.dev"

ARXIV_CATS  = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.MA)"


# ── arXiv search ──────────────────────────────────────────────────────────────

def _bare_arxiv_id(raw: str) -> str:
    m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", raw)
    if m:
        return m.group(1)
    seg = raw.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", seg)


def search_arxiv(keywords: List[str], max_results: int = 30) -> List[Dict[str, Any]]:
    """
    Query arXiv API with user keywords.
    Cap phrase matching at 2 words to avoid over-filtering.
    Returns list of {arxiv_id, title, abstract, authors, year}.
    """
    if not keywords:
        return []

    def _fmt(kw: str) -> str:
        kw = kw.strip()
        words = kw.split()
        if len(words) >= 3:
            term = f'"{" ".join(words[:2])}"'
        elif len(words) == 2:
            term = f'"{kw}"'
        else:
            term = kw
        # Restrict matching to title/abstract only (not authors, comments, etc.)
        return f"(ti:{term} OR abs:{term})"

    primary = _fmt(keywords[0])
    secondary_parts = list(dict.fromkeys(_fmt(k) for k in keywords[1:5]))  # dedup
    secondary_parts = [p for p in secondary_parts if p != primary][:4]

    kw_part = f"({primary} OR {' OR '.join(secondary_parts)})" if secondary_parts else primary
    query = f"{ARXIV_CATS} AND {kw_part}"

    base_params = {"start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}
    print(f"[arXiv] Query: {query[:130]}")

    def _fetch(q: str) -> Optional[ET.Element]:
        try:
            r = requests.get(ARXIV_API, params={**base_params, "search_query": q}, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            return root if root.findall(f"{{{ATOM_NS}}}entry") else None
        except Exception as e:
            print(f"[arXiv] fetch error: {e}")
            return None

    root = _fetch(query)
    if root is None:
        # Fallback: just the first keyword, single word
        fw = keywords[0].split()[0]
        print(f"[arXiv] 0 results – fallback: ti/abs:{fw}")
        root = _fetch(f"{ARXIV_CATS} AND (ti:{fw} OR abs:{fw})")
    if root is None:
        return []

    papers = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        id_el = entry.find(f"{{{ATOM_NS}}}id")
        if id_el is None:
            continue
        arxiv_id = _bare_arxiv_id(id_el.text or "")
        if not arxiv_id:
            continue

        title_el   = entry.find(f"{{{ATOM_NS}}}title")
        summary_el = entry.find(f"{{{ATOM_NS}}}summary")
        pub_el     = entry.find(f"{{{ATOM_NS}}}published")

        title    = (title_el.text or "").strip().replace("\n", " ")
        abstract = (summary_el.text or "").strip().replace("\n", " ")
        year     = 2024
        if pub_el is not None:
            try:
                year = int(pub_el.text[:4])
            except Exception:
                pass

        authors = []
        for a_el in entry.findall(f"{{{ATOM_NS}}}author"):
            n_el = a_el.find(f"{{{ATOM_NS}}}name")
            if n_el is not None and n_el.text:
                authors.append(n_el.text.strip())

        papers.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract,
                        "authors": authors, "year": year})

    print(f"[arXiv] Returned {len(papers)} papers")
    return papers


# ── Semantic Scholar batch enrichment ─────────────────────────────────────────

def _batch_semantic_scholar(
    arxiv_ids: List[str],
    paper_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Enrich papers via the S2 batch endpoint.
    One HTTP request for up to 500 papers – much faster than individual calls.
    Returns {arxiv_id: enriched_dict}.
    """
    if not arxiv_ids:
        return {}

    fields = "paperId,title,year,authors,venue,citationCount,references"
    ids    = [f"arXiv:{aid}" for aid in arxiv_ids]

    print(f"[S2] Batch enriching {len(arxiv_ids)} papers...")
    try:
        resp = requests.post(
            f"{S2_API_BASE}/paper/batch",
            params={"fields": fields},
            json={"ids": ids},
            timeout=25,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            print(f"[S2] Batch failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return {}

        enriched: Dict[str, Dict[str, Any]] = {}
        results = resp.json()          # list aligned with `ids`

        for s2, arxiv_id in zip(results, arxiv_ids):
            if not s2 or not s2.get("paperId"):
                continue

            refs = [r["paperId"] for r in s2.get("references", []) if r and r.get("paperId")]
            venue = s2.get("venue") or "arXiv"

            enriched[arxiv_id] = {
                **paper_map[arxiv_id],
                "id":             s2["paperId"],
                "arxiv_id":       arxiv_id,
                "venue":          venue,
                "citation_count": s2.get("citationCount") or 0,
                "venue_quality":  0.6,
                "references":     refs[:100],
            }

        print(f"[S2] Found {len(enriched)}/{len(arxiv_ids)} papers")
        return enriched

    except Exception as e:
        print(f"[S2] Batch error: {e}")
        return {}


# ── OpenAlex fallback (individual calls in parallel) ─────────────────────────

def _openalex_single(arxiv_id: str, paper: Dict[str, Any]) -> tuple:
    try:
        resp = requests.get(
            OA_API,
            params={"filter": f"ids.arxiv:{arxiv_id}", "mailto": OA_EMAIL, "per_page": 1},
            timeout=8,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                work   = results[0]
                oa_id  = (work.get("id") or "").split("/")[-1]
                loc    = work.get("primary_location") or {}
                src    = (loc.get("source") or {})
                venue  = src.get("display_name") or "arXiv"
                refs   = [r.split("/")[-1] for r in work.get("referenced_works", [])]
                return arxiv_id, {
                    **paper,
                    "id":             oa_id,
                    "arxiv_id":       arxiv_id,
                    "venue":          venue,
                    "citation_count": work.get("cited_by_count") or 0,
                    "venue_quality":  0.5,
                    "references":     refs,
                }
    except Exception:
        pass
    return arxiv_id, None


def _batch_openalex_fallback(
    arxiv_ids: List[str],
    paper_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if not arxiv_ids:
        return {}
    print(f"[OpenAlex] Fallback enrichment for {len(arxiv_ids)} papers...")
    enriched: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(arxiv_ids))) as ex:
        futures = {ex.submit(_openalex_single, aid, paper_map[aid]): aid for aid in arxiv_ids}
        try:
            for fut in as_completed(futures, timeout=30):
                try:
                    aid, result = fut.result(timeout=10)
                    if result:
                        enriched[aid] = result
                except Exception:
                    pass
        except TimeoutError:
            print("[OpenAlex] Fallback enrichment timed out. Proceeding with partially enriched papers.")
    print(f"[OpenAlex] Fallback found {len(enriched)}/{len(arxiv_ids)}")
    return enriched


# ── Semantic Scholar direct search ────────────────────────────────────────────

def search_semantic_scholar(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar directly.
    Papers come back with FULL metadata: title, abstract, authors, year, venue,
    citationCount, references — no separate enrichment step needed.
    Falls back silently on error.
    """
    print(f"[S2 Search] Query: '{query[:80]}'  limit={max_results}")
    try:
        resp = requests.get(
            f"{S2_API_BASE}/paper/search",
            params={
                "query": query,
                "fields": "paperId,title,abstract,year,authors,venue,citationCount,references,externalIds",
                "limit": max_results,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[S2 Search] HTTP {resp.status_code}")
            return []

        results: List[Dict[str, Any]] = []
        for item in resp.json().get("data", []):
            if not item or not item.get("paperId") or not item.get("title"):
                continue

            arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
            refs      = [r["paperId"] for r in item.get("references", []) if r and r.get("paperId")]
            authors   = [a.get("name", "") for a in item.get("authors", [])]

            results.append({
                "id":             item["paperId"],
                "arxiv_id":       arxiv_id,
                "title":          (item.get("title") or "").strip(),
                "abstract":       (item.get("abstract") or "").strip(),
                "authors":        authors,
                "year":           item.get("year") or 2024,
                "venue":          item.get("venue") or "arXiv",
                "citation_count": item.get("citationCount") or 0,
                "venue_quality":  0.6,
                "references":     refs[:100],
                "_s2_direct":     True,   # flag: already fully enriched
            })

        print(f"[S2 Search] Returned {len(results)} papers")
        return results

    except Exception as e:
        print(f"[S2 Search] Error: {e}")
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

def enrich_and_cache(
    arxiv_papers: List[Dict[str, Any]],
    db,
    embedding_model,
) -> List[Dict[str, Any]]:
    """
    For each arXiv paper:
      1. Cache check by arxiv_id  (fast path — skip re-enrichment)
      2. Semantic Scholar batch   (primary — fast, good coverage of recent CS papers)
      3. OpenAlex parallel        (fallback for papers S2 doesn't have yet)
      4. arxiv_* stub             (last resort — no citation data but still indexed)
      5. Embed + classify + DB insert
    """
    from backend.scripts.seed_data_v2 import compute_relevance_and_classify, extract_entities

    results: List[Dict[str, Any]] = []
    to_enrich:     List[Dict[str, Any]] = []   # arXiv-only papers needing S2 enrichment
    to_enrich_ids: List[str] = []
    s2_direct:     List[Dict[str, Any]] = []   # S2 search papers (already fully enriched)

    # ── Pass 1: split S2-direct vs arXiv-only; cache check ───────────────
    for paper in arxiv_papers:
        if paper.get("_s2_direct"):
            # Already has full S2 data — just check cache by S2 ID
            cached = db.get_paper(paper["id"])
            if cached and cached.get("title") != "Cited Reference":
                results.append(cached)
            else:
                s2_direct.append(paper)
            continue

        # arXiv paper — check cache by arxiv_id
        aid    = paper["arxiv_id"]
        cached = db.get_paper_by_arxiv_id(aid)
        if cached and cached.get("title") != "Cited Reference":
            results.append(cached)
        else:
            to_enrich.append(paper)
            to_enrich_ids.append(aid)

    print(f"[Cache] {len(results)} hits | {len(s2_direct)} S2-direct | {len(to_enrich)} arXiv need S2 enrichment")

    # ── Pass 1b: embed + classify + insert S2-direct papers ──────────────
    if s2_direct:
        s2_to_insert: List[Dict[str, Any]] = []
        for paper in s2_direct:
            score, topics = compute_relevance_and_classify(paper["title"], paper.get("abstract", ""))
            if not topics or (len(topics) == 1 and "Other" in topics):
                topics = ["Autonomous Agents"]
            emb_text  = f"{paper['title']}. {paper.get('abstract', '')}"
            embedding = embedding_model.encode(emb_text).tolist()
            full = {
                **paper,
                "topics":             topics,
                "relevance":          score,
                "entities":           extract_entities(paper["title"], paper.get("abstract", "")),
                "embedding":          embedding,
                "intro_summary":      "",
                "conclusion_summary": "",
                "section_headers":    [],
            }
            full.pop("_s2_direct", None)
            s2_to_insert.append(full)
            results.append(full)
        try:
            db.insert_papers(s2_to_insert)
        except Exception as e:
            print(f"[Cache] S2-direct insert warning: {e}")

    if not to_enrich:
        return results

    paper_map = {p["arxiv_id"]: p for p in to_enrich}

    # ── Pass 2: Semantic Scholar batch (primary) ──────────────────────────
    s2_results = _batch_semantic_scholar(to_enrich_ids, paper_map)

    # ── Pass 3: OpenAlex fallback for papers S2 missed ────────────────────
    missing_ids = [aid for aid in to_enrich_ids if aid not in s2_results]
    oa_results: Dict[str, Dict[str, Any]] = {}
    if missing_ids:
        missing_map = {aid: paper_map[aid] for aid in missing_ids}
        oa_results  = _batch_openalex_fallback(missing_ids, missing_map)

    # Merge enrichment results (S2 takes priority)
    all_enriched = {**oa_results, **s2_results}

    to_insert: List[Dict[str, Any]] = []

    for paper in to_enrich:
        arxiv_id = paper["arxiv_id"]

        if arxiv_id in all_enriched:
            enriched = all_enriched[arxiv_id]
            # Already in DB under the enriched ID?
            if db.get_paper(enriched["id"]):
                results.append(db.get_paper(enriched["id"]))
                continue
        else:
            # Neither S2 nor OA found it — stub with arxiv_* ID
            safe_id  = "arxiv_" + arxiv_id.replace(".", "_")
            enriched = {
                **paper,
                "id":             safe_id,
                "arxiv_id":       arxiv_id,
                "venue":          "arXiv",
                "citation_count": 0,
                "venue_quality":  0.4,
                "references":     [],
            }

        # Classify + embed
        score, topics = compute_relevance_and_classify(
            enriched["title"], enriched.get("abstract", "")
        )
        if not topics or (len(topics) == 1 and "Other" in topics):
            topics = ["Autonomous Agents"]

        emb_text  = f"{enriched['title']}. {enriched.get('abstract', '')}"
        embedding = embedding_model.encode(emb_text).tolist()

        full_paper = {
            **enriched,
            "topics":              topics,
            "relevance":           score,
            "entities":            extract_entities(enriched["title"], enriched.get("abstract", "")),
            "embedding":           embedding,
            "intro_summary":       "",
            "conclusion_summary":  "",
            "section_headers":     [],
        }
        to_insert.append(full_paper)
        results.append(full_paper)

    # ── Pass 4: batch DB insert ───────────────────────────────────────────
    if to_insert:
        print(f"[Cache] Inserting {len(to_insert)} new papers...")
        try:
            db.insert_papers(to_insert)
        except Exception as e:
            print(f"[Cache] Insert warning: {e}")

    return results
