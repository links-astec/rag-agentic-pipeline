"""
pipeline.py — public API over the LangGraph agentic RAG graph.

Ingestion is unchanged from Iteration 1.
Querying now runs the full agent graph instead of a single LLM call.

    from src.pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.ingest("./data/raw/")
    answer = rag.query("What are the main risks?")
    print(answer)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage

from loader import load
from chunker import chunk_documents
from vector_store import VectorStore
from graph import rag_graph


class RAGPipeline:
    """
    Iteration 2: agentic RAG via LangGraph.

    Graph: rewrite_query → retrieve → grade_documents → generate
           (with one automatic retry if grading finds no relevant chunks)
    """

    def __init__(self):
        # VectorStore is initialised here for ingestion;
        # nodes.py lazily creates its own singleton for retrieval.
        self._store = VectorStore()

    # ── Ingestion (unchanged from Iteration 1) ────────────────────────

    def ingest(self, source: str) -> None:
        print(f"\n{'='*50}")
        print(f"[Pipeline] Ingesting: {source}")
        print(f"{'='*50}")
        docs   = load(source)
        chunks = chunk_documents(docs)
        self._store.add_documents(chunks)
        print(f"[Pipeline] Ingestion complete — {len(chunks)} chunks stored.\n")

    def clear(self) -> None:
        self._store.delete_collection()

    # ── Querying via LangGraph ────────────────────────────────────────

    def query(self, question: str, verbose: bool = False) -> str:
        """
        Run the agentic RAG graph and return the final answer.

        The graph will:
          1. Rewrite the question for better retrieval
          2. Retrieve top-k chunks from PGVector
          3. Grade each chunk for relevance (Groq)
          4. Retry with a fresh rewrite if zero relevant chunks found
          5. Generate a grounded answer from surviving chunks

        Args:
            question: Natural language question
            verbose:  Print full graph state after execution

        Returns:
            Answer string
        """
        initial_state = {
            "question":        question,
            "rewritten_query": question,
            "documents":       [],
            "generation":      None,
            "rewrite_count":   0,
            "messages":        [HumanMessage(content=question)],
        }

        final_state = rag_graph.invoke(initial_state)

        if verbose:
            print("\n── Agent trace ───────────────────────────────────")
            print(f"  Original query : {final_state['question']}")
            print(f"  Rewritten query: {final_state['rewritten_query']}")
            print(f"  Docs kept      : {len(final_state['documents'])}")
            print(f"  Rewrite count  : {final_state['rewrite_count']}")
            print("──────────────────────────────────────────────────\n")

        return final_state["generation"] or "No answer generated."

    def stream_steps(self, question: str):
        """
        Yield each graph step as it executes.
        Useful for building streaming UIs or debugging.

        Usage:
            for step in rag.stream_steps("What is X?"):
                node_name, state = step
                print(f"[{node_name}] done")
        """
        initial_state = {
            "question":        question,
            "rewritten_query": question,
            "documents":       [],
            "generation":      None,
            "rewrite_count":   0,
            "messages":        [HumanMessage(content=question)],
        }
        for step in rag_graph.stream(initial_state):
            # step is {node_name: partial_state}
            node_name  = next(iter(step))
            node_state = step[node_name]
            yield node_name, node_state
