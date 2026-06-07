"""
Shared dataclasses for the RAG pipeline.
All modules import from here — single source of truth for types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class Document:
    """Raw document loaded from disk before chunking."""
    doc_id: str
    content: str
    source: str                        # original file path
    format: str                        # "pdf" | "markdown" | "text" | "html"
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id!r}, source={self.source!r}, chars={len(self.content)})"


@dataclass
class Chunk:
    """A single text chunk produced by the chunker, ready for embedding."""
    chunk_id: str                      # deterministic hash of content
    doc_id: str                        # parent document
    text: str
    index: int                         # position within parent doc
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id!r}, tokens={self.token_count}, doc={self.doc_id!r})"


@dataclass
class EmbeddedChunk:
    """A chunk that has been embedded and is ready for vector store upsert."""
    chunk: Chunk
    embedding: list[float]
    embedding_model: str


@dataclass
class RetrievedChunk:
    """A chunk returned from vector similarity search, with its score."""
    chunk: Chunk
    score: float                       # cosine similarity 0-1
    rank: int                          # position in result list (1-indexed)


@dataclass
class GenerationResult:
    """Final output from the LLM, with provenance info for evals."""
    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    trace_id: str | None = None

    @property
    def source_texts(self) -> list[str]:
        return [r.chunk.text for r in self.retrieved_chunks]

    @property
    def sources(self) -> list[str]:
        return list({r.chunk.metadata.get("source", "unknown") for r in self.retrieved_chunks})