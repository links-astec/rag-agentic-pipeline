"""
vector_store.py — PGVector-backed store using LangChain + HuggingFace embeddings.

Responsibilities:
  - Initialise PGVector collection (creates table + index if needed)
  - Embed documents via HuggingFaceEmbeddings (local, no API key)
  - Store and query embeddings
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector


def _get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    print(f"[VectorStore] Loading embedding model '{model_name}'...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_connection_string() -> str:
    conn = os.getenv(
        "PGVECTOR_CONNECTION_STRING",
        "postgresql+psycopg://rag:rag_secret@localhost:5432/rag_db",
    )
    return conn


class VectorStore:
    """
    Thin wrapper around LangChain's PGVector store.

    Usage:
        store = VectorStore()
        store.add_documents(chunks)
        results = store.similarity_search("What is X?", k=5)
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(self):
        self.embeddings = _get_embeddings()
        self.store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            connection=_get_connection_string(),
            use_jsonb=True,
        )
        print(f"[VectorStore] Connected. Collection: '{self.COLLECTION_NAME}'")

    def add_documents(self, docs: List[Document]) -> None:
        """Embed and persist a list of Documents."""
        print(f"[VectorStore] Embedding and storing {len(docs)} chunks...")
        self.store.add_documents(docs)
        print(f"[VectorStore] Done.")

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> List[Document]:
        """
        Return top-k most similar chunks.

        Args:
            query:  Natural language question
            k:      Number of results (default from env TOP_K or 5)
            filter: Optional metadata filter, e.g. {"source": "doc.pdf"}

        Returns:
            List of Documents with similarity scores in metadata
        """
        k = k or int(os.getenv("TOP_K", 5))
        results = self.store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )
        # Attach score to metadata for transparency
        docs = []
        for doc, score in results:
            doc.metadata["relevance_score"] = round(score, 4)
            docs.append(doc)
        return docs

    def delete_collection(self) -> None:
        """Drop and recreate the collection (useful for re-ingestion)."""
        self.store.delete_collection()
        print(f"[VectorStore] Collection '{self.COLLECTION_NAME}' deleted.")
