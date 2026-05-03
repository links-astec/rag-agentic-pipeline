"""
state.py — typed state object that flows through every LangGraph node.

All nodes read from and write to AgentState. LangGraph merges partial
updates returned by each node back into the running state dict.
"""

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Fields
    ------
    question        : original user question (never mutated)
    rewritten_query : query after the rewrite node (may equal question)
    documents       : chunks retrieved from PGVector
    generation      : final answer string
    rewrite_count   : how many times we've rewritten (guards infinite loops)
    messages        : full conversation history for multi-turn support
    """

    question:        str
    rewritten_query: str
    documents:       List[Document]
    generation:      Optional[str]
    rewrite_count:   int
    messages:        List[BaseMessage]
