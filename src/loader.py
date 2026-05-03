"""
loader.py — multi-source document loader using LangChain document loaders.
Supports: .txt, .pdf, web URLs, and directories.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)


def load(source: str) -> List[Document]:
    """
    Auto-detect source type and return a list of LangChain Documents.

    Args:
        source: file path, directory path, or http(s) URL

    Returns:
        List[Document] with .page_content and .metadata
    """
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[Loader] Fetching URL: {source}")
        loader = WebBaseLoader(source)

    elif os.path.isdir(source):
        print(f"[Loader] Scanning directory: {source}")
        # Load .txt and .pdf files recursively
        docs = []
        for ext, loader_cls in [("**/*.txt", TextLoader), ("**/*.pdf", PyPDFLoader)]:
            loader = DirectoryLoader(
                source,
                glob=ext,
                loader_cls=loader_cls,
                silent_errors=True,
            )
            docs.extend(loader.load())
        print(f"[Loader] Found {len(docs)} document(s)")
        return docs

    elif source.lower().endswith(".pdf"):
        print(f"[Loader] Loading PDF: {source}")
        loader = PyPDFLoader(source)

    else:
        print(f"[Loader] Loading text file: {source}")
        loader = TextLoader(source, encoding="utf-8")

    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} document(s)")
    return docs
