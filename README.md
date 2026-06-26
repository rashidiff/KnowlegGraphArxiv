# Agentic Research Paper Knowledge Graph Navigator

An agentic, graph-reasoning web application designed to navigate, traverse, and discover research papers across a scientific citation network. Combining hybrid vector search, community-based topic clustering, topological graph analysis, and a multi-agent orchestration pipeline built with **LangGraph & LangChain**.

---

## 🚀 Key Features

*   **Real Scientific Data**: Ingests thousands of papers dynamically from the **OpenAlex API** with full metadata, author names, venues, citations, and reconstructed abstracts.
*   **Dual Database Layer & Fallback**: Exposes a Docker Compose setup for **Neo4j** (Graph) and **PostgreSQL + pgvector** (Relational & Semantic), with a seamless **SQLite + NetworkX + Numpy** in-memory/file-based fallback if Docker is inactive.
*   **Topological Graph Insights**: Calculates PageRank centrality (to find foundational papers), betweenness centrality (to detect bridge papers connecting fields), and Louvain community detection (for clustering).
*   **Multi-Agent Reasoning (LangGraph)**:
    *   `Router/Orchestrator Agent`: Handles intent classification and ambiguity checks (asks clarifying questions first).
    *   `Research Planner Agent`: Formulates execution objectives.
    *   `Retriever Agent`: Executes hybrid vector + keyword search ranked by a multi-factor score.
    *   `Graph Insight Agent`: Extracts network topologies and clusters.
    *   `Reading Path Agent`: Computes chronological/dependency-based reading roadmaps.
    *   `Idea Agent`: Proposes novel research gap hypotheses backed by graph metrics.
    *   `Verifier Agent (Critic)`: Fact-checks agent statements against raw paper abstracts.
    *   `Responder Agent`: Formulates final grounding outputs.
*   **Interactive D3 Force Graph UI**: A clean, research-oriented dashboard visualizing paper nodes (sized by citations, colored by topic) and animating links along citation paths.

---

## 🛠️ Tech Stack

*   **Frontend**: Next.js (React), HTML5, TypeScript, Tailwind CSS, Lucide React, `react-force-graph-2d` (D3 Canvas).
*   **Backend**: FastAPI, Uvicorn, Python 3.11.
*   **Orchestration**: LangGraph, LangChain, OpenAI / DeepSeek APIs.
*   **Database**: PostgreSQL + pgvector & Neo4j (Docker) / SQLite & NetworkX (Fallback).
*   **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2` - 384 dimensions, local).

---

## ⚙️ Installation & Setup

### 1. Configure Environment variables
Copy `.env.example` to `.env` and configure your API key (the system supports DeepSeek by default, compatible with OpenAI SDK):
```bash
cp .env.example .env
```
Ensure your `OPENAI_API_KEY` (e.g. DeepSeek API key) is defined.

### 2. Setup Python Backend Environment
```bash
pip install -r requirements.txt
```

### 3. Run Ingestion Pipeline (Seeding)
Fetch and embed 2,500 agent-related research papers from OpenAlex (computes embeddings locally on CPU):
```bash
python backend/scripts/seed_data.py 2500
```

---

## 🏃 Running the Application

### Option A: Local Dev Runs (SQLite & NetworkX Fallback)
This is the fastest, zero-config method.

1.  **Start Backend API Server**:
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ```
2.  **Start Frontend Web Client (Next.js)**:
    ```bash
    cd frontend
    npm run dev
    ```
3.  Open browser at: [http://localhost:3000](http://localhost:3000).

### Option B: Docker Stack (Postgres + Neo4j)
1.  **Spin up databases**:
    ```bash
    docker compose up -d
    ```
2.  Once databases boot, run the seeding script `python backend/scripts/seed_data.py 2500`. It will automatically detect Postgres and Neo4j, populate them, and start Uvicorn/Next.js.



## 💡 Example Queries

*   **Foundational search**: *"What are the most important papers on browser agents?"*
*   **Comparison**: *"Compare Yao et al., 2023 [ReAct] with Toolformer."*
*   **Roadmap**: *"I want to learn web agents. Give me a reading path."*
*   **Research gap**: *"Suggest a new research direction based on agent evaluation limitations in the corpus."*
