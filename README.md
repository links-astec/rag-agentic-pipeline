# RAG — Iteration 2: Agentic RAG

LangGraph · Groq function calling · Query rewriting · Relevance grading · Self-correcting retrieval

## What's new vs Iteration 1

| | Iteration 1 | Iteration 2 |
|---|---|---|
| Querying | Single retrieve → generate call | Full LangGraph agent graph |
| Query rewriting | ✗ | ✓ Groq rewrites for better retrieval |
| Relevance grading | ✗ | ✓ Groq grades each chunk, filters noise |
| Retry logic | ✗ | ✓ Auto-retries with new query if no relevant docs |
| Step visibility | ✗ | ✓ `stream_steps()` exposes each node |
| Ingestion | LangChain loaders + PGVector | Unchanged |

## Architecture

```
START
  │
  ▼
rewrite_query ──────────────────────────────────────────┐
  │  Groq rephrases for retrieval recall                 │ retry
  ▼                                                      │ (once)
retrieve                                                 │
  │  PGVector cosine similarity search                   │
  ▼                                                      │
grade_documents                                          │
  │  Groq: {relevant: true/false} per chunk              │
  │                                                      │
  ├── has relevant docs ──► generate ──► END             │
  │                                                      │
  └── no relevant docs + can retry ──────────────────────┘
  │
  └── no relevant docs + retried ──► generate (fallback) ──► END
```

## Quick Start

### 1. Start PGVector
```bash
docker compose up -d
```

### 2. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Set GROQ_API_KEY in .env
```

### 4. Ingest
```bash
python main.py ingest ./data/raw/
```

### 5. Query

```bash
# One-shot with full agent trace
python main.py query "What are the main findings?"

# Watch each graph node execute in real time
python main.py steps "What are the main findings?"

# Interactive chat
python main.py chat

# Reset vector store
python main.py clear
```

### `steps` output example

```
Running agent graph for: 'What are the key risks?'

  ✓ [rewrite_query]
    Rewritten: What are the primary risks and potential threats discussed?
  ✓ [retrieve]
    Retrieved: 5 chunks
  ✓ [grade_documents]
    Kept:      3 relevant chunks
  ✓ [generate]
    Answer:    The document identifies three key risks: ...
```

## Using as a library

```python
from src.pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest("./data/raw/")

# Standard query
answer = rag.query("What is X?", verbose=True)

# Step-by-step streaming (for UIs / debugging)
for node_name, state in rag.stream_steps("What is X?"):
    print(f"Node '{node_name}' completed")
    if node_name == "generate":
        print(state["generation"])
```

## Files changed from Iteration 1

```
rag_iter2/
├── main.py               + steps command
├── requirements.txt      + langgraph
└── src/
    ├── state.py          NEW — AgentState TypedDict
    ├── nodes.py          NEW — rewrite / retrieve / grade / generate nodes
    ├── graph.py          NEW — LangGraph StateGraph + conditional edges
    ├── pipeline.py       UPDATED — drives graph instead of direct LLM call
    ├── loader.py         unchanged
    ├── chunker.py        unchanged
    └── vector_store.py   unchanged
```

## What's next — Iteration 3

- HuggingFace sentence-transformers (swap embedding model)
- Hybrid BM25 + vector search
- Cross-encoder re-ranking
- PGVector HNSW index tuning
