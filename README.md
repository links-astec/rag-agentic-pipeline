# rag-agentic-pipeline

> **Iteration 2 of 6** — Agentic RAG with self-correcting retrieval, query rewriting, and relevance grading via LangGraph.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-red)
![PGVector](https://img.shields.io/badge/PGVector-PostgreSQL16-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## What this does

This project implements a **production-style Retrieval-Augmented Generation (RAG) pipeline** that goes beyond simple retrieve-then-generate. It uses a **LangGraph agent graph** to reason over multiple steps — rewriting queries for better recall, grading each retrieved chunk for relevance, and automatically retrying when retrieval quality is poor.

Instead of blindly passing whatever the vector store returns to the LLM, the agent decides whether the retrieved context is actually useful — and if not, it rewrites the question and tries again.

---

## Architecture

```
User question
      │
      ▼
┌─────────────────┐
│  rewrite_query  │  ← Groq rephrases the question for better retrieval recall
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    retrieve     │  ← PGVector cosine similarity search (top-k chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│grade_documents  │  ← Groq scores each chunk: {"relevant": true/false}
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
has docs   no docs + can retry
    │         │
    ▼         └──► rewrite_query (retry once)
┌─────────────────┐
│    generate     │  ← Groq produces a grounded answer
└─────────────────┘
         │
         ▼
      Answer
```

The graph compiles to a `StateGraph` where each node reads from and writes to a shared `AgentState` dict. The conditional edge after `grade_documents` enables the self-correction loop.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Agent orchestration | LangGraph | Stateful multi-step graphs with conditional edges and retry loops |
| LLM | Groq (`llama-3.3-70b-versatile`) | Fast inference, free tier, function calling support |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local, no API key, fast on CPU |
| Vector store | PGVector (PostgreSQL 16) | Production-grade, ACID transactions, SQL metadata filtering |
| Infrastructure | Docker Compose | Single command to spin up the full DB stack |
| Framework | LangChain LCEL | Composable chains: `prompt \| llm \| parser` |

---

## Project structure

```
rag-agentic-pipeline/
├── main.py                  # CLI — ingest / query / chat / steps / clear
├── requirements.txt
├── .env.example
├── docker-compose.yml       # PGVector on PostgreSQL 16
├── init.sql                 # Schema + IVFFlat index
├── data/
│   └── raw/                 # Drop source documents here
└── src/
    ├── state.py             # AgentState TypedDict (shared graph state)
    ├── nodes.py             # rewrite_query, retrieve, grade_documents, generate
    ├── graph.py             # LangGraph StateGraph + conditional edges
    ├── pipeline.py          # Public API: ingest() + query() + stream_steps()
    ├── loader.py            # PDF, TXT, URL, directory loaders
    ├── chunker.py           # RecursiveCharacterTextSplitter
    ├── vector_store.py      # PGVector wrapper with relevance scores
    └── generator.py         # Groq LCEL chain with streaming support
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop running
- Groq API key — free at [console.groq.com](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-agentic-pipeline.git
cd rag-agentic-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your values:

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=all-MiniLM-L6-v2
PGVECTOR_CONNECTION_STRING=postgresql+psycopg://rag:rag_secret@localhost:5433/rag_db
POSTGRES_USER=rag
POSTGRES_PASSWORD=rag_secret
POSTGRES_DB=rag_db
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

> **Note:** Port `5433` avoids conflicts with any local PostgreSQL installation.

### 3. Start the database

```bash
docker compose up -d

# Confirm healthy (wait ~15 seconds)
docker compose ps
```

### 4. Ingest documents

```bash
# From a URL
python main.py ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# From a local PDF
python main.py ingest ./data/raw/report.pdf

# From a directory
python main.py ingest ./data/raw/
```

### 5. Query

```bash
# Interactive chat
python main.py chat

# One-shot query with full agent trace
python main.py query "What is retrieval-augmented generation?"

# Watch each LangGraph node execute in real time
python main.py steps "What are the limitations of RAG?"

# Reset the vector store
python main.py clear
```

---

## CLI reference

| Command | Description |
|---|---|
| `python main.py ingest <source>` | Load and index a file, folder, or URL |
| `python main.py query "<question>"` | One-shot Q&A with verbose agent trace |
| `python main.py chat` | Interactive chat loop |
| `python main.py steps "<question>"` | Stream each graph node as it executes |
| `python main.py clear` | Drop and recreate the vector collection |

---

## How the agent works

### Query rewriting
Before retrieving, Groq rephrases the question to maximise recall — expanding acronyms, resolving pronouns, and adding synonyms. For example:

```
Original:  "What did they say about it?"
Rewritten: "What statements were made about retrieval-augmented generation?"
```

### Relevance grading
After retrieval, each chunk is individually scored by Groq:

```json
{"relevant": true}
{"relevant": false}
```

Irrelevant chunks are dropped before generation, reducing noise in the LLM context.

### Self-correcting retry
If grading leaves zero relevant chunks, the agent rewrites the query again and retries retrieval once. If the second pass also returns nothing, it generates a graceful fallback response instead of hallucinating.

### Watching it happen — `steps` command

```
Running agent graph for: 'What are the key limitations?'

  ✓ [rewrite_query]
    Rewritten: What are the primary limitations and challenges of RAG systems?
  ✓ [retrieve]
    Retrieved: 5 chunks
  ✓ [grade_documents]
    Kept: 3 relevant chunks
  ✓ [generate]
    Answer: The key limitations include...
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** From [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `PGVECTOR_CONNECTION_STRING` | — | PostgreSQL connection string |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

---

## What changed from Iteration 1

| | Iteration 1 | Iteration 2 |
|---|---|---|
| Querying | Single retrieve → generate | Full LangGraph agent graph |
| Query rewriting | ✗ | ✓ Groq rewrites for better recall |
| Relevance grading | ✗ | ✓ Groq filters irrelevant chunks |
| Retry logic | ✗ | ✓ Auto-retries with rewritten query |
| Step visibility | ✗ | ✓ `steps` command exposes each node |

---

## Part of the RAG Pipeline series

| Iteration | Repo | Focus |
|---|---|---|
| 1 | `rag-core-pipeline` | LangChain, Groq, PGVector, HuggingFace |
| **2** | **`rag-agentic-pipeline`** | **LangGraph agents, query rewriting, grading** |
| 3 | `rag-hybrid-retrieval` | BGE embeddings, BM25, RRF, re-ranking |
| 4 | `rag-api-service` | FastAPI, Docker, GitHub Actions CI/CD |
| 5 | `rag-gcp-deployment` | GKE, Cloud SQL, Artifact Registry |
| 6 | `rag-production-monitoring` | LangSmith, Prometheus, Grafana, RAGAS |