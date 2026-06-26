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

# Query keywords for OpenAlex
SEARCH_QUERY = '("web agent" OR "browser agent" OR "LLM agent" OR "AI agent" OR "agent evaluation" OR "human-AI interaction" OR "tool use")'

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

def classify_topics(title: str, abstract: str) -> List[str]:
    """Classify topics based on keywords in title and abstract."""
    topics = []
    text = (title + " " + abstract).lower()
    
    if "browser" in text:
        topics.append("Browser Agents")
    if "web" in text:
        topics.append("Web Agents")
    if "eval" in text or "benchmark" in text or "metrics" in text:
        topics.append("Agent Evaluation")
    if "human" in text or "interaction" in text or "user" in text:
        topics.append("Human-AI Interaction")
    if "tool" in text or "api" in text or "function calling" in text:
        topics.append("Tool Use")
        
    if not topics:
        topics.append("LLM Agents")
        
    return topics

def extract_entities(title: str, abstract: str) -> List[Dict[str, str]]:
    """Extract concepts, methods, and datasets using rule-based keyword matching."""
    text = (title + " " + abstract).lower()
    entities = []
    
    # Predefined entity list
    concepts = {
        "retrieval-augmented generation": "Retrieval-Augmented Generation",
        "rag": "Retrieval-Augmented Generation",
        "chain-of-thought": "Chain-of-Thought",
        "chain of thought": "Chain-of-Thought",
        "cot": "Chain-of-Thought",
        "reinforcement learning": "Reinforcement Learning",
        "planning": "Planning",
        "memory": "Memory Systems",
        "multi-agent": "Multi-Agent Systems",
        "human-in-the-loop": "Human-in-the-loop",
        "reasoning": "Reasoning"
    }
    
    methods = {
        "react": "ReAct",
        "reflexion": "Reflexion",
        "toolformer": "Toolformer",
        "voyager": "Voyager",
        "auto-gpt": "AutoGPT",
        "autogpt": "AutoGPT",
        "monte carlo tree search": "MCTS",
        "mcts": "MCTS",
        "tree of thoughts": "Tree of Thoughts",
        "tot": "Tree of Thoughts"
    }
    
    datasets = {
        "webarena": "WebArena",
        "mind2web": "Mind2Web",
        "swe-bench": "SWE-bench",
        "swebench": "SWE-bench",
        "hotpotqa": "HotpotQA",
        "mmlu": "MMLU",
        "humaneval": "HumanEval"
    }

    for kw, val in concepts.items():
        if kw in text:
            entities.append({"type": "concept", "value": val})
            
    for kw, val in methods.items():
        if kw in text:
            entities.append({"type": "method", "value": val})
            
    for kw, val in datasets.items():
        if kw in text:
            entities.append({"type": "dataset", "value": val})
            
    # Deduplicate
    unique_entities = []
    seen = set()
    for ent in entities:
        key = (ent["type"], ent["value"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
            
    return unique_entities

def fetch_openalex_papers(target_count: int = 5000) -> List[Dict[str, Any]]:
    """Fetch papers from OpenAlex with pagination."""
    papers = []
    base_url = "https://api.openalex.org/works"
    page = 1
    per_page = 200
    email = "agent@research-navigator.dev" # Polite pool

    print(f"Starting OpenAlex Ingestion. Target: {target_count} papers...")
    
    while len(papers) < target_count:
        params = {
            "filter": f"title_and_abstract.search:{SEARCH_QUERY},language:en,has_abstract:true",
            "per_page": per_page,
            "page": page,
            "mailto": email
        }
        
        try:
            print(f"Fetching page {page} (Fetched so far: {len(papers)})...")
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code != 200:
                print(f"Error calling OpenAlex: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            results = data.get("results", [])
            if not results:
                print("No more results returned from OpenAlex.")
                break
                
            for work in results:
                # Extract clean ID
                full_id = work.get("id", "")
                short_id = full_id.split("/")[-1] if "/" in full_id else full_id
                
                title = work.get("title") or "Untitled"
                abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
                
                # We skip works without title or abstract
                if not title or not abstract:
                    continue
                    
                # Authors list
                authors = []
                for auth in work.get("authorships", []):
                    author_name = auth.get("author", {}).get("display_name")
                    if author_name:
                        authors.append(author_name)
                        
                # Venue
                venue = "Unknown"
                loc = work.get("primary_location")
                if loc and loc.get("source"):
                    venue = loc["source"].get("display_name") or "Unknown"
                    
                # References (cites)
                references = []
                for ref in work.get("referenced_works", []):
                    ref_id = ref.split("/")[-1] if "/" in ref else ref
                    references.append(ref_id)
                    
                topics = classify_topics(title, abstract)
                entities = extract_entities(title, abstract)
                
                # Estimate venue quality based on citations or source type (simple mock quality score)
                citation_count = work.get("cited_by_count") or 0
                venue_quality = 0.5
                if "nature" in venue.lower() or "science" in venue.lower() or "neurips" in venue.lower() or "iclr" in venue.lower():
                    venue_quality = 0.9
                elif "arxiv" in venue.lower():
                    venue_quality = 0.4
                
                papers.append({
                    "id": short_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": work.get("publication_year") or 2026,
                    "venue": venue,
                    "citation_count": citation_count,
                    "venue_quality": venue_quality,
                    "references": references,
                    "topics": topics,
                    "entities": entities
                })
                
                if len(papers) >= target_count:
                    break
                    
            page += 1
            # Rate limiting politely
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Network error during OpenAlex fetch: {e}")
            break
            
    print(f"Completed fetch. Total papers retrieved: {len(papers)}")
    return papers

def generate_embeddings(papers: List[Dict[str, Any]]) -> None:
    """Generate embeddings using local sentence-transformers."""
    print("Initializing SentenceTransformer('all-MiniLM-L6-v2')...")
    from sentence_transformers import SentenceTransformer
    
    # Force CPU for local simplicity
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    print("Generating embeddings for titles and abstracts...")
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    
    # Run in batches
    batch_size = 256
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Embedding batch {i // batch_size + 1}/{len(texts) // batch_size + 1}...")
        batch_embs = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embs.tolist())
        
    for p, emb in zip(papers, embeddings):
        p["embedding"] = emb
        
    print("Embeddings generation complete.")

def main():
    target = 2500 # Default count for MVP, preferred 5000, let's fetch 2500 for speed first, or we can fetch 5000!
    # Let's check command line args for target count
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            pass
            
    # Fetch
    papers = fetch_openalex_papers(target_count=target)
    
    # Fallback to local sample papers if fetching completely failed (offline mode)
    if not papers:
        print("Ingestion failed. Seeding fallback mock papers to ensure database works...")
        papers = [
            {
                "id": "W1",
                "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
                "abstract": "We present ReAct, an approach where LLMs generate reasoning traces and task-specific actions. This synergizes reasoning and acting, allowing the model to perform dynamic planning and execute tool calls in interactive environments like WebArena and HotpotQA.",
                "authors": ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"],
                "year": 2023,
                "venue": "International Conference on Learning Representations (ICLR)",
                "citation_count": 850,
                "venue_quality": 0.9,
                "references": ["W2", "W3"],
                "topics": ["Browser Agents", "Web Agents", "Tool Use"],
                "entities": [{"type": "concept", "value": "Reasoning"}, {"type": "method", "value": "ReAct"}, {"type": "dataset", "value": "HotpotQA"}]
            },
            {
                "id": "W2",
                "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                "abstract": "We explore how generating a series of intermediate reasoning steps—a chain of thought—improves the ability of large language models to perform complex reasoning. We evaluate on arithmetic, commonsense, and symbolic reasoning tasks.",
                "authors": ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei-Fei Li", "Ed Chi", "Quoc V. Le"],
                "year": 2022,
                "venue": "NeurIPS",
                "citation_count": 4500,
                "venue_quality": 0.95,
                "references": [],
                "topics": ["LLM Agents"],
                "entities": [{"type": "concept", "value": "Chain-of-Thought"}, {"type": "concept", "value": "Reasoning"}]
            },
            {
                "id": "W3",
                "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
                "abstract": "Language models can use external tools (APIs) to access search engines, calculators, and translation systems. We train Toolformer, which decides which APIs to call, what arguments to pass, and how to merge the results into its text generation.",
                "authors": ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu", "Maria Lomeli", "Luke Zettlemoyer", "Nicola Cancedda", "Thomas Scialom"],
                "year": 2023,
                "venue": "arXiv",
                "citation_count": 620,
                "venue_quality": 0.4,
                "references": ["W2"],
                "topics": ["Tool Use"],
                "entities": [{"type": "concept", "value": "Retrieval-Augmented Generation"}, {"type": "method", "value": "Toolformer"}]
            }
        ]
        
    # Generate Embeddings
    generate_embeddings(papers)
    
    # Store in DB
    db = get_db()
    print("Writing papers to the database...")
    db.insert_papers(papers)
    print("Seeding complete! Database successfully populated.")

if __name__ == "__main__":
    main()
