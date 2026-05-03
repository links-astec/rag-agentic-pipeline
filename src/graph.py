"""
graph.py — builds the LangGraph StateGraph for agentic RAG.

Graph topology
--------------

    START
      │
      ▼
  rewrite_query          rephrase for better retrieval
      │
      ▼
   retrieve              PGVector similarity search
      │
      ▼
  grade_documents        Groq filters irrelevant chunks
      │
      ├─── has_docs ──► generate ──► END
      │
      └─── no_docs  ──► rewrite_query  (retry once)
                             │
                             ▼
                          retrieve
                             │
                             ▼
                        grade_documents
                             │
                             ▼  (always generate on second pass)
                           generate ──► END

The retry path fires only when grading leaves zero documents AND we
haven't already retried (rewrite_count < 1). On the second pass the
decide_next edge always routes to generate, even if docs are empty,
so we never loop indefinitely.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes import generate, grade_documents, retrieve, rewrite_query
from state import AgentState


# ── Conditional edge ──────────────────────────────────────────────────────────

def decide_next(state: AgentState) -> str:
    """
    After grade_documents:
      - If relevant docs exist      → generate
      - If no docs and can retry    → rewrite_query (will retrieve again)
      - If no docs and retry used   → generate (will return fallback)
    """
    has_docs     = len(state["documents"]) > 0
    can_retry    = state.get("rewrite_count", 0) < 1

    if has_docs:
        return "generate"
    elif can_retry:
        print("[Graph] No relevant docs — retrying with rewritten query.")
        return "rewrite_query"
    else:
        print("[Graph] No relevant docs after retry — generating fallback.")
        return "generate"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph RAG agent."""

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("rewrite_query",   rewrite_query)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate",        generate)

    # Edges
    builder.add_edge(START,             "rewrite_query")
    builder.add_edge("rewrite_query",   "retrieve")
    builder.add_edge("retrieve",        "grade_documents")

    # Conditional branch after grading
    builder.add_conditional_edges(
        "grade_documents",
        decide_next,
        {
            "generate":     "generate",
            "rewrite_query": "rewrite_query",   # retry path
        },
    )

    builder.add_edge("generate", END)

    return builder.compile()


# Singleton — compile once and reuse
rag_graph = build_graph()
