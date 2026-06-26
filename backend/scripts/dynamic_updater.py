"""
Dynamic arXiv Updater
Fetches recent papers (last N months) from arXiv, enriches with OpenAlex citation data,
filters by relevance, runs 2-hop expansion, and inserts into the corpus incrementally.

Usage:
    python backend/scripts/dynamic_updater.py
"""
import os
import sys
import json
import time
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.manager import get_db
from backend.constants import EXCLUSION_KEYWORDS
from backend.scripts.seed_data_v2 import (
    compute_relevance_and_classify,
    extract_entities,
    reconstruct_abstract,
    fetch_cited_papers,
)

# ── Config ─────────────────────────────────────────────────────────────────
MONTHS_BACK = 4
MAX_PER_QUERY = 200          # arXiv results per search term
RELEVANCE_THRESHOLD = 0.3
TWO_HOP_TARGET = 400
OPENALEX_BATCH = 50
EMAIL = "agent@research-navigator.dev"
META_PATH = "data/corpus_metadata.json"

ATOM_NS = "http://www.w3.org/2005/Atom"

# arXiv search queries (title-focused for precision)
ARXIV_QUERIES = [
    "cat:cs.AI AND (ti:agent OR ti:agents OR ti:agentic)",
    "cat:cs.CL AND (ti:agent OR ti:agents OR ti:agentic)",
    "cat:cs.LG AND ti:agent AND ti:\"language model\"",
    "cat:cs.MA AND (ti:agent OR ti:agents OR ti:\"multi-agent\")",
    "cat:cs.AI AND ti:\"tool use\"",
    "cat:cs.CL AND ti:\"tool use\"",
    "cat:cs.AI AND ti:\"autonomous agent\"",
    "cat:cs.CL AND ti:\"autonomous agent\"",
    "cat:cs.AI AND ti:\"web agent\"",
    "cat:cs.AI AND ti:\"browser agent\"",
]


# ── arXiv fetch ─────────────────────────────────────────────────────────────

def _arxiv_id_from_url(url: str) -> str:
    """Extract bare arXiv ID (e.g. '2506.12345') from full URL or urn."""
    # e.g. http://arxiv.org/abs/2506.12345v2
    m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", url)
    if m:
        return m.group(1)
    # fallback: take last path segment and strip version
    seg = url.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", seg)


def fetch_arxiv_papers(months_back: int = MONTHS_BACK) -> List[Dict[str, Any]]:
    """Query arXiv API for recent agent/LLM papers, return raw paper dicts."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    base_url = "http://export.arxiv.org/api/query"

    seen_ids: set = set()
    papers: List[Dict[str, Any]] = []

    for query in ARXIV_QUERIES:
        print(f"[arXiv] Querying: {query[:60]}...")
        params = {
            "search_query": query,
            "start": 0,
            "max_results": MAX_PER_QUERY,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"[arXiv] HTTP {resp.status_code} — skipping query")
                time.sleep(3)
                continue

            root = ET.fromstring(resp.content)
            entries = root.findall(f"{{{ATOM_NS}}}entry")

            added = 0
            for entry in entries:
                id_el = entry.find(f"{{{ATOM_NS}}}id")
                if id_el is None:
                    continue
                arxiv_id = _arxiv_id_from_url(id_el.text or "")
                if not arxiv_id or arxiv_id in seen_ids:
                    continue

                # Date filter
                pub_el = entry.find(f"{{{ATOM_NS}}}published")
                if pub_el is not None:
                    try:
                        pub_dt = datetime.fromisoformat(pub_el.text.replace("Z", "+00:00"))
                        if pub_dt < cutoff:
                            continue
                    except Exception:
                        pass

                title_el = entry.find(f"{{{ATOM_NS}}}title")
                summary_el = entry.find(f"{{{ATOM_NS}}}summary")
                title = (title_el.text or "").strip().replace("\n", " ")
                abstract = (summary_el.text or "").strip().replace("\n", " ")

                authors = []
                for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
                    name_el = author_el.find(f"{{{ATOM_NS}}}name")
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                pub_year = cutoff.year
                if pub_el is not None:
                    try:
                        pub_year = datetime.fromisoformat(pub_el.text.replace("Z", "+00:00")).year
                    except Exception:
                        pass

                seen_ids.add(arxiv_id)
                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": pub_year,
                })
                added += 1

            print(f"[arXiv]   -> {added} new entries (cutoff={cutoff.date()})")
            time.sleep(3)  # arXiv rate-limit courtesy

        except Exception as e:
            print(f"[arXiv] Error: {e}")
            time.sleep(5)

    print(f"[arXiv] Total candidate papers fetched: {len(papers)}")
    return papers


# ── OpenAlex enrichment ─────────────────────────────────────────────────────

def enrich_with_openalex(arxiv_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Look up each arXiv paper in OpenAlex by its arXiv ID.
    Adds: openalex_id, citation_count, references, venue.
    Papers not found in OpenAlex keep minimal metadata (no citation edges).
    """
    base_url = "https://api.openalex.org/works"
    enriched: List[Dict[str, Any]] = []

    # Batch lookup: OpenAlex supports ids.arxiv filter for single IDs
    # We loop individually for precision (arXiv IDs are unique)
    for i, paper in enumerate(arxiv_papers):
        arxiv_id = paper["arxiv_id"]
        try:
            resp = requests.get(
                base_url,
                params={
                    "filter": f"ids.arxiv:{arxiv_id}",
                    "mailto": EMAIL,
                    "per_page": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    work = results[0]
                    full_id = work.get("id", "")
                    oa_id = full_id.split("/")[-1] if "/" in full_id else full_id

                    venue = "arXiv"
                    loc = work.get("primary_location")
                    if loc and loc.get("source"):
                        venue = loc["source"].get("display_name") or "arXiv"

                    refs = [r.split("/")[-1] for r in work.get("referenced_works", [])]

                    # Use OpenAlex abstract if ours is empty
                    abstract = paper["abstract"]
                    if not abstract:
                        abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))

                    enriched.append({
                        **paper,
                        "id": oa_id,
                        "abstract": abstract,
                        "venue": venue,
                        "citation_count": work.get("cited_by_count") or 0,
                        "venue_quality": 0.5,
                        "references": refs,
                    })
                    if (i + 1) % 20 == 0:
                        print(f"[OpenAlex] Enriched {i+1}/{len(arxiv_papers)} papers...")
                    time.sleep(0.15)
                    continue

        except Exception as e:
            print(f"[OpenAlex] Error for {arxiv_id}: {e}")

        # Not found in OpenAlex — use arXiv-only data with generated ID
        safe_id = "arxiv_" + arxiv_id.replace(".", "_")
        enriched.append({
            **paper,
            "id": safe_id,
            "venue": "arXiv",
            "citation_count": 0,
            "venue_quality": 0.4,
            "references": [],
        })
        time.sleep(0.1)

    print(f"[OpenAlex] Enrichment done. {len(enriched)} papers processed.")
    return enriched


# ── Embedding generation ────────────────────────────────────────────────────

def generate_embeddings(papers: List[Dict[str, Any]]) -> None:
    """Adds 'embedding' key (list of floats) to each paper dict in-place."""
    from backend.agents.retriever import get_embedding_model
    model = get_embedding_model()
    print(f"[Embed] Generating embeddings for {len(papers)} papers...")
    for i, p in enumerate(papers):
        text = f"{p['title']}. {p.get('abstract', '')}"
        p["embedding"] = model.encode(text).tolist()
        if (i + 1) % 50 == 0:
            print(f"[Embed]   {i+1}/{len(papers)} done")


# ── Main update routine ─────────────────────────────────────────────────────

def run_update(months_back: int = MONTHS_BACK) -> Dict[str, Any]:
    """
    Full pipeline: arXiv fetch → OpenAlex enrich → filter → embed → 2-hop → insert.
    Returns a summary dict.
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  DYNAMIC ARXIV UPDATE  (last {months_back} months)")
    print(f"{'='*60}\n")

    db = get_db()

    # ── Phase 1: Fetch from arXiv ─────────────────────────────────────────
    arxiv_papers = fetch_arxiv_papers(months_back=months_back)
    if not arxiv_papers:
        print("[Updater] No papers fetched from arXiv.")
        return {"status": "no_papers", "added": 0}

    # ── Phase 2: Relevance filter (fast, before OpenAlex calls) ──────────
    print(f"\n[Filter] Scoring {len(arxiv_papers)} papers for relevance...")
    relevant = []
    excluded = 0
    for p in arxiv_papers:
        score, topics = compute_relevance_and_classify(p["title"], p["abstract"])
        if score >= RELEVANCE_THRESHOLD and "Other" not in topics:
            p["topics"] = topics
            p["relevance"] = score
            relevant.append(p)
        else:
            excluded += 1
    print(f"[Filter] {len(relevant)} relevant, {excluded} excluded.")

    if not relevant:
        print("[Updater] No relevant papers after filtering.")
        return {"status": "no_relevant", "added": 0}

    # ── Phase 3: OpenAlex enrichment for citation data ────────────────────
    print(f"\n[Enrich] Looking up {len(relevant)} papers in OpenAlex...")
    enriched = enrich_with_openalex(relevant)

    # ── Phase 4: Skip papers already in DB ───────────────────────────────
    print("\n[Dedup] Checking for already-indexed papers...")
    new_papers = []
    skipped = 0
    for p in enriched:
        if db.get_paper(p["id"]) is None:
            p["entities"] = extract_entities(p["title"], p.get("abstract", ""))
            new_papers.append(p)
        else:
            skipped += 1
    print(f"[Dedup] {len(new_papers)} new, {skipped} already in corpus.")

    if not new_papers:
        print("[Updater] Corpus already up-to-date.")
        _save_metadata(added=0, skipped=skipped, excluded=excluded)
        return {"status": "up_to_date", "added": 0, "skipped": skipped}

    # ── Phase 5: Generate embeddings ──────────────────────────────────────
    generate_embeddings(new_papers)

    # ── Phase 6: Insert new papers ────────────────────────────────────────
    print(f"\n[DB] Inserting {len(new_papers)} new papers...")
    db.insert_papers(new_papers)

    # ── Phase 7: 2-hop backward expansion ────────────────────────────────
    all_in_db_dict = {p["id"]: p for p in new_papers}
    two_hop = fetch_cited_papers(all_in_db_dict, target_extra=TWO_HOP_TARGET)

    truly_new_hop: List[Dict[str, Any]] = []
    if two_hop:
        for p in two_hop:
            if db.get_paper(p["id"]) is None:
                truly_new_hop.append(p)
        if truly_new_hop:
            generate_embeddings(truly_new_hop)
            db.insert_papers(truly_new_hop)
            print(f"[2-hop] Inserted {len(truly_new_hop)} additional cited papers.")

    total_added = len(new_papers) + len(truly_new_hop)
    elapsed = time.time() - start_time

    _save_metadata(added=total_added, skipped=skipped, excluded=excluded)

    print(f"\n{'='*60}")
    print(f"  UPDATE COMPLETE — {total_added} new papers added in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {
        "status": "success",
        "added": total_added,
        "skipped": skipped,
        "excluded": excluded,
        "elapsed_seconds": round(elapsed, 1),
    }


def _save_metadata(added: int, skipped: int, excluded: int) -> None:
    os.makedirs("data", exist_ok=True)
    existing: Dict[str, Any] = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH) as f:
                existing = json.load(f)
        except Exception:
            pass

    existing["last_updated"] = datetime.now(timezone.utc).isoformat()
    existing["last_update_added"] = added
    existing["last_update_skipped"] = skipped
    existing["last_update_excluded"] = excluded

    with open(META_PATH, "w") as f:
        json.dump(existing, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic arXiv corpus updater")
    parser.add_argument("--months", type=int, default=MONTHS_BACK, help="Months back to fetch")
    args = parser.parse_args()
    result = run_update(months_back=args.months)
    print(json.dumps(result, indent=2))
