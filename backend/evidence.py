"""Stable, client-safe evidence records for research answers."""

from typing import Any


def paper_source_url(paper: dict[str, Any]) -> str | None:
    """Return the best public canonical source available for a paper."""
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    openalex_id = str(paper.get("openalex_id") or "").strip()
    if openalex_id:
        return openalex_id
    return None


def build_evidence_cards(papers: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Make a bounded evidence payload without exposing vector embeddings."""
    cards: list[dict[str, Any]] = []
    for paper in papers[:limit]:
        abstract = str(paper.get("abstract") or "").strip()
        cards.append({
            "paper_id": paper.get("id"),
            "title": paper.get("title") or "Untitled paper",
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "citation_count": paper.get("citation_count", 0),
            "source_url": paper_source_url(paper),
            "locator": "Abstract",
            "excerpt": abstract[:600],
        })
    return cards
