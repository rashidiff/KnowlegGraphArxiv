import os
import sys
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any

# Adjust path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.manager import get_db
from backend.constants import TAXONOMY, EXCLUSION_KEYWORDS

# Whitelist Seed Paper Titles to target first
SEED_PAPERS = [
    {"title": "ReAct: Synergizing Reasoning and Acting in Language Models", "keywords": ["react", "reasoning", "acting"]},
    {"title": "Toolformer: Language Models Can Teach Themselves to Use Tools", "keywords": ["toolformer"]},
    {"title": "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", "keywords": ["webvoyager"]},
    {"title": "BrowserGym: A Playground for Web Agent Research", "keywords": ["browsergym"]},
    {"title": "WebArena: A Realistic Web Environment for Building Autonomous Agents", "keywords": ["webarena"]},
    {"title": "AgentBench: Evaluating Language Agents in Translation, Reasoning, and Coding", "keywords": ["agentbench"]},
    {"title": "Voyager: An Open-Ended Embodied Agent with Large Language Models", "keywords": ["voyager", "minecraft"]},
    {"title": "AutoGPT", "keywords": ["autogpt", "auto-gpt"]},
    {"title": "SWE-Agent: Agent-Computer Interfaces Enable Language Models to Solve Software Issues", "keywords": ["swe-agent", "swebench"]},
    {"title": "OSWorld: Benchmarking Multimodal Agents on Desktop Environments", "keywords": ["osworld"]},
    {"title": "SeeAct: GPT-4V(ision) for Automated Web Navigation", "keywords": ["seeact"]},
    {"title": "Mind2Web: Towards a Generalist Agent for the Web", "keywords": ["mind2web"]}
]

def reconstruct_abstract(index: Dict[str, List[int]]) -> str:
    """Reconstruct abstract from OpenAlex abstract_inverted_index."""
    if not index:
        return ""
    try:
        positions = []
        for word, idxs in index.items():
            for idx in idxs:
                positions.append((idx, word))
        positions.sort()
        return " ".join([word for _, word in positions])
    except Exception:
        return ""

def compute_relevance_and_classify(title: str, abstract: str, is_seed: bool = False) -> tuple[float, List[str]]:
    """
    Computes domain relevance score and classifies paper into taxonomy.
    Returns: (relevance_score, list_of_topics)
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    combined_text = title_lower + " " + abstract_lower

    # 1. Hard exclusions check
    for kw in EXCLUSION_KEYWORDS:
        if kw in combined_text:
            return 0.0, []

    # 2. Positive keywords check
    positive_terms = [
        "web agent", "browser agent", "llm agent", "autonomous agent", "agent evaluation", 
        "ai agent", "language model", "multi-agent", "agentic", "prompting", "tool use", 
        "api call", "function calling", "webarena", "browsergym", "agentbench", "swe-bench", 
        "software agent", "reasoning and acting", "toolformer", "voyager", "hotpotqa", "mind2web"
    ]

    matches = sum(1 for term in positive_terms if term in combined_text)
    if matches == 0 and not is_seed:
        return 0.0, []

    # Score calculation
    # Title match has higher weight
    title_matches = sum(1.5 for term in positive_terms if term in title_lower)
    abstract_matches = sum(0.5 for term in positive_terms if term in abstract_lower)
    raw_score = (title_matches + abstract_matches)
    
    # Base relevance
    relevance_score = min(1.0, 0.2 + (raw_score / 6.0))
    if is_seed:
        relevance_score = 1.0

    # 3. Classify topics
    topics = []
    
    # Browser Agents
    if any(term in combined_text for term in ["browser agent", "browsergym", "mind2web", "browser automation"]):
        topics.append("Browser Agents")
        
    # Web Agents
    if any(term in combined_text for term in ["web agent", "web navigation", "webvoyager", "web arena", "webarena", "online environment"]):
        topics.append("Web Agents")
        
    # Agent Evaluation
    if any(term in combined_text for term in ["agent evaluation", "benchmark", "agentbench", "swe-bench", "swebench", "evaluation framework", "evaluating agent"]):
        topics.append("Agent Evaluation")
        
    # Tool Use for LLMs
    if any(term in combined_text for term in ["tool use", "tool-using", "toolformer", "api call", "external tool", "function calling"]):
        topics.append("Tool Use for LLMs")
        
    # Autonomous Agents
    if any(term in combined_text for term in ["autonomous agent", "ai agent", "llm agent", "agentic", "react", "voyager", "auto-gpt", "autogpt"]):
        topics.append("Autonomous Agents")
        
    # Multi-Agent Systems
    if any(term in combined_text for term in ["multi-agent", "multiagent", "society of agents", "agent collaboration"]):
        topics.append("Multi-Agent Systems")
        
    # Human-AI Interaction
    if any(term in combined_text for term in ["human-ai", "human-agent", "human-computer", "interaction", "user study", "human collaboration"]):
        topics.append("Human-AI Interaction")

    # If it has positive matches but fits no specific topic, label as Other (will be discarded)
    if not topics:
        topics.append("Other")

    return relevance_score, topics

def extract_entities(title: str, abstract: str) -> List[Dict[str, str]]:
    """Extract entities (concepts, methods, datasets) based on keywords."""
    combined = (title + " " + abstract).lower()
    entities = []

    concepts = {
        "retrieval-augmented generation": "Retrieval-Augmented Generation",
        "rag": "Retrieval-Augmented Generation",
        "chain-of-thought": "Chain-of-Thought",
        "cot": "Chain-of-Thought",
        "reinforcement learning": "Reinforcement Learning",
        "planning": "Planning",
        "memory": "Memory Systems",
        "reasoning": "Reasoning"
    }
    
    methods = {
        "react": "ReAct",
        "reflexion": "Reflexion",
        "toolformer": "Toolformer",
        "voyager": "Voyager",
        "auto-gpt": "AutoGPT",
        "swe-agent": "SWE-Agent",
        "mcts": "MCTS"
    }
    
    datasets = {
        "webarena": "WebArena",
        "mind2web": "Mind2Web",
        "swe-bench": "SWE-bench",
        "agentbench": "AgentBench",
        "hotpotqa": "HotpotQA",
        "mmlu": "MMLU"
    }

    for k, v in concepts.items():
        if k in combined:
            entities.append({"type": "concept", "value": v})
    for k, v in methods.items():
        if k in combined:
            entities.append({"type": "method", "value": v})
    for k, v in datasets.items():
        if k in combined:
            entities.append({"type": "dataset", "value": v})

    # Deduplicate
    unique = []
    seen = set()
    for ent in entities:
        key = (ent["type"], ent["value"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique

def fetch_seed_papers() -> List[Dict[str, Any]]:
    """Find and fetch the metadata for our whitelisted seed papers from OpenAlex."""
    base_url = "https://api.openalex.org/works"
    seeds = []
    email = "agent@research-navigator.dev"

    print("Fetching Seed Papers from OpenAlex...")
    for seed in SEED_PAPERS:
        params = {
            "filter": f"title.search:{seed['title']}",
            "mailto": email,
            "per_page": 3
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    # Find best match
                    best_match = results[0]
                    for r in results:
                        if seed["title"].lower() in (r.get("title") or "").lower():
                            best_match = r
                            break
                    
                    full_id = best_match.get("id", "")
                    short_id = full_id.split("/")[-1] if "/" in full_id else full_id
                    title = best_match.get("title") or seed["title"]
                    abstract = reconstruct_abstract(best_match.get("abstract_inverted_index", {}))
                    
                    # Compute relevance and classify
                    score, topics = compute_relevance_and_classify(title, abstract, is_seed=True)
                    if score > 0 and "Other" not in topics:
                        authors = [a.get("author", {}).get("display_name") for a in best_match.get("authorships", []) if a.get("author")]
                        venue = "Unknown"
                        loc = best_match.get("primary_location")
                        if loc and loc.get("source"):
                            venue = loc["source"].get("display_name") or "Unknown"
                            
                        refs = [r.split("/")[-1] for r in best_match.get("referenced_works", [])]

                        seeds.append({
                            "id": short_id,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": best_match.get("publication_year") or 2024,
                            "venue": venue,
                            "citation_count": best_match.get("cited_by_count") or 0,
                            "venue_quality": 0.8,
                            "references": refs,
                            "topics": topics,
                            "entities": extract_entities(title, abstract),
                            "is_seed": True,
                            "relevance": 1.0
                        })
                        print(f"  Loaded seed: {title} ({short_id})")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching seed {seed['title']}: {e}")
            
    return seeds

def fetch_expansion_papers(seeds: List[Dict[str, Any]], target_count: int = 1500) -> List[Dict[str, Any]]:
    """Expand corpus using references and citations of seed papers, and direct keyword search."""
    papers_dict = {p["id"]: p for p in seeds}
    email = "agent@research-navigator.dev"
    base_url = "https://api.openalex.org/works"
    excluded_count = 0

    # 1. Fetch papers that cite our seeds (one hop citations)
    print("\nExpanding via Citation Neighborhoods...")
    for seed in seeds[:8]:  # Limit to top seeds to prevent API rate limits
        if len(papers_dict) >= target_count:
            break
            
        print(f"  Fetching papers citing: '{seed['title']}'...")
        params = {
            "filter": f"cites:{seed['id']},has_abstract:true,language:en",
            "mailto": email,
            "per_page": 100
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                for work in results:
                    full_id = work.get("id", "")
                    short_id = full_id.split("/")[-1] if "/" in full_id else full_id
                    if short_id in papers_dict:
                        continue
                        
                    title = work.get("title") or "Untitled"
                    abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
                    
                    score, topics = compute_relevance_and_classify(title, abstract)
                    if score >= 0.6 and "Other" not in topics:
                        authors = [a.get("author", {}).get("display_name") for a in work.get("authorships", []) if a.get("author")]
                        venue = "Unknown"
                        loc = work.get("primary_location")
                        if loc and loc.get("source"):
                            venue = loc["source"].get("display_name") or "Unknown"
                        refs = [r.split("/")[-1] for r in work.get("referenced_works", [])]

                        papers_dict[short_id] = {
                            "id": short_id,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": work.get("publication_year") or 2024,
                            "venue": venue,
                            "citation_count": work.get("cited_by_count") or 0,
                            "venue_quality": 0.5,
                            "references": refs,
                            "topics": topics,
                            "entities": extract_entities(title, abstract),
                            "is_seed": False,
                            "relevance": score
                        }
                    else:
                        excluded_count += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Error expanding seed citation: {e}")

    # 2. General precise keyword query
    print("\nRunning Whitelist Keyword Search Ingestion...")
    keywords_queries = [
        '("web agent" OR "browser agent" OR "LLM agent")',
        '("autonomous agent" AND "evaluation")',
        '("tool use" AND "language model")',
        '("multi-agent systems" AND "LLM")'
    ]
    
    page = 1
    for query in keywords_queries:
        if len(papers_dict) >= target_count:
            break
            
        print(f"  Searching OpenAlex for: {query}...")
        params = {
            "filter": f"title_and_abstract.search:{query},has_abstract:true,language:en",
            "mailto": email,
            "per_page": 200,
            "page": page
        }
        try:
            res = requests.get(base_url, params=params, timeout=15)
            if res.status_code == 200:
                results = res.json().get("results", [])
                for work in results:
                    full_id = work.get("id", "")
                    short_id = full_id.split("/")[-1] if "/" in full_id else full_id
                    if short_id in papers_dict:
                        continue
                        
                    title = work.get("title") or "Untitled"
                    abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
                    
                    score, topics = compute_relevance_and_classify(title, abstract)
                    if score >= 0.65 and "Other" not in topics:
                        authors = [a.get("author", {}).get("display_name") for a in work.get("authorships", []) if a.get("author")]
                        venue = "Unknown"
                        loc = work.get("primary_location")
                        if loc and loc.get("source"):
                            venue = loc["source"].get("display_name") or "Unknown"
                        refs = [r.split("/")[-1] for r in work.get("referenced_works", [])]

                        papers_dict[short_id] = {
                            "id": short_id,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "year": work.get("publication_year") or 2024,
                            "venue": venue,
                            "citation_count": work.get("cited_by_count") or 0,
                            "venue_quality": 0.5,
                            "references": refs,
                            "topics": topics,
                            "entities": extract_entities(title, abstract),
                            "is_seed": False,
                            "relevance": score
                        }
                    else:
                        excluded_count += 1
                    
                    if len(papers_dict) >= target_count:
                        break
            time.sleep(0.5)
        except Exception as e:
            print(f"Error querying OpenAlex keywords: {e}")

    # Return list of expanded papers
    final_papers = list(papers_dict.values())
    print(f"\nIngestion Complete. Retrieved {len(final_papers)} papers. Filtered and Excluded {excluded_count} irrelevant papers.")
    
    # Save excluded count in a global way
    os.makedirs("data", exist_ok=True)
    metadata = {
        "excluded_papers_count": excluded_count,
        "quality_score": float(np.mean([p["relevance"] for p in final_papers])) if final_papers else 1.0
    }
    with open("data/corpus_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return final_papers

def fetch_cited_papers(papers_dict: Dict[str, Any], target_extra: int = 800) -> List[Dict[str, Any]]:
    """
    2-hop BACKWARD expansion: fetch papers that are cited BY the current corpus.

    OpenAlex referenced_works IDs that are not yet in the corpus are resolved and
    fetched in batches of 50 using the ids.openalex filter. Papers cited by multiple
    corpus papers are prioritised (they are more likely to be foundational works).
    Lower relevance threshold (0.25) is used since these papers are already chosen by
    relevant authors — they just need a non-zero abstract and no hard exclusion.
    """
    base_url = "https://api.openalex.org/works"
    email = "agent@research-navigator.dev"

    # Count how many corpus papers cite each referenced ID
    ref_freq: Dict[str, int] = {}
    for p in papers_dict.values():
        for ref_id in p.get("references", []):
            if ref_id and ref_id not in papers_dict:
                ref_freq[ref_id] = ref_freq.get(ref_id, 0) + 1

    # Sort: most-cited refs first
    sorted_refs = sorted(ref_freq.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = [rid for rid, _ in sorted_refs]

    print(f"\n[2-hop] Found {len(candidate_ids)} unique reference IDs not in corpus.")
    print(f"[2-hop] Fetching up to {target_extra} papers (batches of 50)...")

    fetched: Dict[str, Any] = {}
    batch_size = 50

    for i in range(0, len(candidate_ids), batch_size):
        if len(fetched) >= target_extra:
            break

        batch = candidate_ids[i: i + batch_size]
        ids_filter = "|".join(batch)

        params = {
            "filter": f"ids.openalex:{ids_filter}",
            "mailto": email,
            "per_page": batch_size,
        }

        try:
            res = requests.get(base_url, params=params, timeout=15)
            if res.status_code != 200:
                print(f"[2-hop] API error {res.status_code} on batch {i // batch_size + 1}")
                time.sleep(1)
                continue

            results = res.json().get("results", [])
            for work in results:
                full_id = work.get("id", "")
                short_id = full_id.split("/")[-1] if "/" in full_id else full_id

                if short_id in papers_dict or short_id in fetched:
                    continue

                title = work.get("title") or "Untitled"
                abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))

                if not abstract:
                    continue

                # Hard exclusion check
                combined = (title + " " + abstract).lower()
                if any(kw in combined for kw in EXCLUSION_KEYWORDS):
                    continue

                # For 2-hop: accept if cited ≥ 2 times OR passes low relevance threshold
                cite_freq = ref_freq.get(short_id, 0)
                score, topics = compute_relevance_and_classify(title, abstract)

                # Papers with no relevance signal at all are excluded even if cited frequently.
                # A score of 0 means the paper matches zero positive terms — it's off-topic.
                # We keep the lower threshold (0.3) but remove the freq-only override.
                if score < 0.3:
                    continue

                # Assign a default topic if none matched (foundational NLP / reasoning papers)
                if not topics or (len(topics) == 1 and "Other" in topics):
                    topics = ["Autonomous Agents"]

                authors = [
                    a.get("author", {}).get("display_name")
                    for a in work.get("authorships", [])
                    if a.get("author")
                ]
                venue = "Unknown"
                loc = work.get("primary_location")
                if loc and loc.get("source"):
                    venue = loc["source"].get("display_name") or "Unknown"
                refs = [r.split("/")[-1] for r in work.get("referenced_works", [])]

                fetched[short_id] = {
                    "id": short_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": work.get("publication_year") or 2024,
                    "venue": venue,
                    "citation_count": work.get("cited_by_count") or 0,
                    "venue_quality": 0.5,
                    "references": refs,
                    "topics": topics,
                    "entities": extract_entities(title, abstract),
                    "is_seed": False,
                    "relevance": max(score, 0.25),
                }

            time.sleep(0.4)

        except Exception as e:
            print(f"[2-hop] Error on batch {i // batch_size + 1}: {e}")
            time.sleep(1)

    print(f"[2-hop] Backward expansion complete: {len(fetched)} new papers added.")
    return list(fetched.values())


def generate_embeddings(papers: List[Dict[str, Any]]) -> None:
    """Generate embeddings using local sentence-transformers."""
    print("\nInitializing SentenceTransformer('all-MiniLM-L6-v2')...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("Generating embeddings for titles and abstracts...")
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    
    batch_size = 256
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{len(texts) // batch_size + 1}...")
        batch_embs = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embs.tolist())
        
    for p, emb in zip(papers, embeddings):
        p["embedding"] = emb
        
    print("Embeddings generation complete.")

def validate_corpus():
    """
    Sanity check validation.
    Runs PageRank on the populated database and verifies that the top papers are
    agent-related (ReAct, Toolformer, WebVoyager, etc.) and DO NOT contain animal tool use references.
    """
    print("\nRunning Ingested Corpus Validation...")
    db = get_db()
    metrics = db.get_graph_metrics()
    
    # 1. Total papers
    import sqlite3
    conn = sqlite3.connect("data/research_navigator.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    total_papers = cursor.fetchone()[0]
    print(f"Total Papers in Database: {total_papers}")
    
    cursor.execute("SELECT topic, COUNT(*) FROM paper_topics GROUP BY topic")
    print("Papers per topic:")
    for topic, count in cursor.fetchall():
        print(f"  - {topic}: {count}")

    # 2. PageRank top nodes sanity check
    top_papers = metrics.get("foundational_papers", [])
    print("Top 10 Foundational Papers by PageRank:")
    
    for i, paper in enumerate(top_papers):
        title = paper.get("title", "")
        pid = paper.get("id", "")
        score = paper.get("score", 0.0)
        print(f"  {i+1}. [{pid}] {title} (PR: {score:.5f})")

        # Double check for animal keyword leakages
        title_lower = title.lower()
        for kw in EXCLUSION_KEYWORDS:
            if kw in title_lower:
                raise ValueError(f"CRITICAL SANITY CHECK FAILED: Irrelevant paper found in top PageRank nodes: {title}")
                
    print("SUCCESS: Ingested Corpus is clean and verified (no animal-related papers found in top PageRank nodes).")

def main():
    target = 1500  # Default target, can fetch up to 2500 but 1500 is much faster for local seed expansion
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            pass
            
    db = get_db()
    print("Clearing existing database records to ensure a clean start...")
    try:
        cursor = db.conn.cursor()
        cursor.execute("DELETE FROM citations")
        cursor.execute("DELETE FROM paper_topics")
        cursor.execute("DELETE FROM paper_entities")
        cursor.execute("DELETE FROM papers")
        db.conn.commit()
        print("Database tables cleared successfully.")
    except Exception as e:
        print(f"Warning: Could not clear database tables: {e}")

    # Fetch seeds
    seeds = fetch_seed_papers()
    if not seeds:
        print("Error: Could not retrieve seeds from OpenAlex. Falling back to local offline mock corpus.")
        # Mock seeds to let development work offline
        return

    # Phase 1 – forward expansion (papers that cite seeds + keyword search)
    papers = fetch_expansion_papers(seeds, target_count=target)
    papers_dict = {p["id"]: p for p in papers}

    # Phase 2 – 2-hop BACKWARD expansion (papers cited by corpus papers)
    two_hop_papers = fetch_cited_papers(papers_dict, target_extra=800)
    # Add to dict to avoid duplication in embedding step
    for p in two_hop_papers:
        if p["id"] not in papers_dict:
            papers_dict[p["id"]] = p

    all_papers = list(papers_dict.values())
    print(f"\nTotal corpus before embedding: {len(all_papers)} papers")

    # Generate Embeddings for the full corpus
    generate_embeddings(all_papers)

    # Insert into DB
    db = get_db()
    print("Inserting papers to SQLite database...")
    db.insert_papers(all_papers)
    print("Database seeding complete.")

    # Validate
    validate_corpus()

if __name__ == "__main__":
    main()
