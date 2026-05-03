"""
chunker.py — splits LangChain Documents into overlapping chunks.
Uses RecursiveCharacterTextSplitter which respects paragraph and sentence
boundaries before falling back to character-level splitting.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        docs:          LangChain Document list from loader
        chunk_size:    Characters per chunk (default from env or 500)
        chunk_overlap: Overlap between chunks (default from env or 50)

    Returns:
        List of chunk Documents with updated metadata
    """
    chunk_size    = chunk_size    or int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 50))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Tag each chunk with its index within its source document
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]

    print(f"[Chunker] {len(docs)} doc(s) → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
