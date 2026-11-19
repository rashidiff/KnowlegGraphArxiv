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
*   **Citation Connection Explorer**: Select two papers in the graph's **Connect papers** controls to find a stored citation connection without an LLM call. Inspect intermediate papers, view the focused path, export its JSON, or clear it to restore the graph. Connections may traverse citations in either direction; they do not imply chronological reading order. A missing path only describes the indexed corpus and database traversal limits (Neo4j searches up to 10 hops).

---

## 🛠️ Tech Stack

*   **Frontend**: Next.js (React), HTML5, TypeScript, Tailwind CSS, Lucide React, `react-force-graph-2d` (D3 Canvas).
*   **Backend**: FastAPI, Uvicorn, Python 3.11.
*   **Orchestration**: LangGraph, LangChain, OpenAI / DeepSeek APIs.
*   **Database**: PostgreSQL + pgvector & Neo4j (Docker) / SQLite & NetworkX (Fallback).
*   **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2` - 384 dimensions, local).

---

## ⚙️ Installation & Setup

### One-command local startup (recommended)

Install **Python 3.11** and **Node.js 20.9 or newer** (including npm) once.
Then clone and start from a terminal:

```powershell
git clone https://github.com/rashidiff/KnowlegGraphArxiv.git
cd KnowlegGraphArxiv
.\start.cmd
```

The command creates `.venv`, installs backend and frontend dependencies, and
copies `.env.example` to `.env` if needed. On Windows it opens **Notepad** when
the API key is missing. Enter `OPENAI_API_KEY`, save, close the editor, and press
Enter in the terminal. DeepSeek is the default provider; for OpenAI also set
`OPENAI_BASE_URL=https://api.openai.com/v1` and `OPENAI_MODEL` to a model available
to your account. A valid provider key and internet access are required for chat.

An empty database is seeded with 100 papers (and the local embedding model is
downloaded on first use). This can take several minutes and requires internet
access and disk space for Python/ML packages. Existing papers are preserved.
Docker is optional: the existing SQLite fallback is used when databases are absent.
The browser opens at http://localhost:3000 after both services are ready.
Keep the terminal open; **Ctrl+C** stops both services.

Run `.\start.cmd` again for subsequent sessions. Dependency installs are reused
until their manifests change; `.env` is never overwritten. Use
`.\start.cmd --edit-env` to reopen configuration, `--seed-count 2500` to change
the initial corpus size, or `--no-browser` to suppress browser opening.
Ports 8000 and 3000 must be available. Installation errors are shown in the terminal;
fix the reported problem and rerun the same command.

On macOS/Linux, use `python3.11 scripts/start.py`; configuration opens in `$EDITOR`
(or `nano`). Manual setup remains available below.

### 1. Configure Environment variables
Copy `.env.example` to `.env` and configure your API key (the system supports DeepSeek by default, compatible with OpenAI SDK). Ensure your `OPENAI_API_KEY` is defined.

Keep `.env` local. Do not commit real API keys, database passwords, generated SQLite files, uploads, or corpus metadata.

For the frontend, set `NEXT_PUBLIC_API_BASE_URL` when the FastAPI server is not running at `http://localhost:8000`.

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
2.  Once databases boot, run `start.cmd` (or `python3.11 scripts/start.py`). It detects Postgres and Neo4j, seeds an empty corpus, and starts Uvicorn/Next.js. The manual seeding command only populates data; it does not start servers.

---

## Production Notes

The explorer is also available through `GET /api/graph/path?start_id=...&end_id=...`.
The response includes `found`, `hops`, and ordered `path` metadata. Disconnected
papers return an empty path with `hops: null`; unknown IDs return HTTP 404.
Use URL-encoded query parameters for IDs containing slashes, such as OpenAlex URLs.

* Set `CORS_ALLOWED_ORIGINS` to the exact frontend origins instead of using broad browser access.
* Set `NEXT_PUBLIC_API_BASE_URL` to the public backend URL used by the browser.
* Set `CORPUS_RESET_TOKEN` and send it as `X-Admin-Token` before exposing the corpus reset action.
* Store `OPENAI_API_KEY`, database passwords, and provider credentials in the deployment platform's secret manager.
* Keep upload and database storage on a persistent volume if the SQLite fallback is used.
* Run `python -m pytest` before deployment; the suite includes checks that real `.env` files and common API key patterns are not committed.
* Restrict the corpus reset endpoint before exposing the API publicly.


## 💡 Example Queries

*   **Foundational search**: *"What are the most important papers on browser agents?"*
*   **Comparison**: *"Compare Yao et al., 2023 [ReAct] with Toolformer."*
*   **Roadmap**: *"I want to learn web agents. Give me a reading path."*
*   **Research gap**: *"Suggest a new research direction based on agent evaluation limitations in the corpus."*
