from backend.evidence import build_evidence_cards, paper_source_url


def test_arxiv_paper_gets_a_canonical_source_url():
    assert paper_source_url({"arxiv_id": "2401.01234"}) == "https://arxiv.org/abs/2401.01234"


def test_evidence_cards_are_bounded_and_exclude_embeddings():
    cards = build_evidence_cards([
        {"id": "p1", "title": "Paper", "abstract": "a" * 700, "embedding": [0.1], "arxiv_id": "2401.01234"},
    ])
    assert cards[0]["paper_id"] == "p1"
    assert len(cards[0]["excerpt"]) == 600
    assert "embedding" not in cards[0]
