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
    source: str
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id!r}, source={self.source!r}, chars={len(self.content)})"


@dataclass
class Chunk:
    """A single text chunk produced by the chunker, ready for embedding."""
    chunk_id: str
    doc_id: str
    text: str
    index: int
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
    score: float
    rank: int


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


@dataclass
class EvalResult:
    """
    Evaluation scores for a single query-answer pair.
    All scores are 0.0 – 1.0. Higher is better.
    """
    trace_id: str
    query: str
    answer: str
    context_texts: list[str]

    # Core RAGAS-style metrics
    faithfulness: float         # are all claims grounded in the context?
    answer_relevance: float     # does the answer address the question?
    context_precision: float    # are retrieved chunks relevant?
    context_recall: float       # did we retrieve everything needed?

    # Metadata
    model: str = ""
    eval_model: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        """Weighted average of all four metrics."""
        return round(
            (self.faithfulness * 0.35 +
             self.answer_relevance * 0.30 +
             self.context_precision * 0.20 +
             self.context_recall * 0.15),
            4,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "model": self.model,
            "eval_model": self.eval_model,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }