"""
nodes.py — every LangGraph node in the agentic RAG graph.

Node contract: receives AgentState, returns a dict of fields to update.

Nodes
-----
rewrite_query   : use Groq to rephrase the question for better retrieval
retrieve        : similarity search against PGVector
grade_documents : Groq decides whether each retrieved chunk is relevant
generate        : Groq produces the final answer from graded docs
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state import AgentState
from vector_store import VectorStore


# ── Shared LLM ────────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Node: rewrite_query ───────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at improving search queries for a RAG system.\n"
     "Rewrite the user's question to maximise retrieval recall.\n"
     "- Expand acronyms, resolve pronouns, add synonyms if helpful.\n"
     "- Return ONLY the rewritten query — no explanation, no quotes."),
    ("human", "Original question: {question}"),
])

_rewrite_chain = _REWRITE_PROMPT | _llm(temperature=0.3) | StrOutputParser()


def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rephrase the question to improve retrieval recall."""
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    # Only rewrite once to avoid runaway loops
    if rewrite_count >= 1:
        print("[Node] rewrite_query: max rewrites reached, skipping.")
        return {"rewritten_query": state.get("rewritten_query", question)}

    rewritten = _rewrite_chain.invoke({"question": question})
    print(f"[Node] rewrite_query: '{question}' → '{rewritten}'")
    return {
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count + 1,
    }


# ── Node: retrieve ────────────────────────────────────────────────────────────

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(state: AgentState) -> dict[str, Any]:
    """Retrieve top-k chunks from PGVector using the (rewritten) query."""
    query = state.get("rewritten_query") or state["question"]
    k     = int(os.getenv("TOP_K", 5))

    docs = _get_store().similarity_search(query, k=k)
    print(f"[Node] retrieve: found {len(docs)} chunk(s) for '{query}'")
    return {"documents": docs}


# ── Node: grade_documents ─────────────────────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading whether a retrieved document chunk is relevant to the question.\n"
     "Respond with ONLY valid JSON: {{\"relevant\": true}} or {{\"relevant\": false}}.\n"
     "No explanation. No markdown fences."),
    ("human",
     "Question: {question}\n\nDocument chunk:\n{document}"),
])

_grade_chain = _GRADE_PROMPT | _llm() | JsonOutputParser()


def grade_documents(state: AgentState) -> dict[str, Any]:
    """
    Filter retrieved chunks: keep only those Groq marks as relevant.
    Sets 'documents' to the filtered list.
    """
    question  = state["question"]
    documents = state["documents"]
    relevant  = []

    for doc in documents:
        try:
            result = _grade_chain.invoke({
                "question": question,
                "document": doc.page_content,
            })
            if result.get("relevant", False):
                relevant.append(doc)
        except Exception as e:
            print(f"[Node] grade_documents: parse error ({e}), keeping chunk.")
            relevant.append(doc)  # keep on error — fail open

    print(f"[Node] grade_documents: {len(relevant)}/{len(documents)} chunks kept.")
    return {"documents": relevant}


# ── Node: generate ────────────────────────────────────────────────────────────

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions using only the provided context.\n\n"
     "Rules:\n"
     "- Answer ONLY from the context. Do not use prior knowledge.\n"
     "- If context is insufficient, say so clearly.\n"
     "- Cite source filenames when available.\n"
     "- Be concise and factual.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

_generate_chain = _GENERATE_PROMPT | _llm() | StrOutputParser()


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        score  = doc.metadata.get("relevance_score", "—")
        parts.append(f"[{i}] (source: {source}, score: {score})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate(state: AgentState) -> dict[str, Any]:
    """Generate a grounded answer from the graded documents."""
    question  = state["question"]
    documents = state["documents"]

    if not documents:
        answer = "I could not find relevant information to answer your question."
        print("[Node] generate: no relevant docs, returning fallback.")
        return {"generation": answer}

    context = _format_context(documents)
    answer  = _generate_chain.invoke({"context": context, "question": question})
    print(f"[Node] generate: answer produced ({len(answer)} chars).")
    return {"generation": answer}
