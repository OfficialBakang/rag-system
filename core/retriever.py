"""
Retriever
=========
Two-stage retrieval with full OTel instrumentation:
  1. Query rewriting  — LLM call to improve search precision
  2. Vector search    — cosine similarity top-k from ChromaDB
  3. Reranking        — optional cross-encoder re-scoring
"""

from __future__ import annotations

import logging
import os

import anthropic

from core.embedder import Embedder
from core.models import RetrievedChunk
from core.telemetry import record_span
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query optimiser for a retrieval-augmented generation system.
Rewrite the user's question into a concise, keyword-rich search query.

Rules:
- Output ONLY the rewritten query — no explanation, no punctuation at the end
- Preserve all named entities (people, products, dates, figures)
- Remove conversational filler
- Keep the query under 20 words

User question: {query}
Rewritten query:"""


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        rewrite_queries: bool = True,
        rerank: bool = False,
        top_k: int | None = None,
        top_n: int | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.rewrite_queries = rewrite_queries
        self.rerank = rerank
        self.top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", 6))
        self.top_n = top_n or int(os.getenv("RERANK_TOP_N", 3))
        self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._reranker = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:

        k = top_k or self.top_k

        with record_span("retriever.retrieve", {
            "query.original": query,
            "retriever.top_k": k,
            "retriever.rewrite": self.rewrite_queries,
            "retriever.rerank": self.rerank,
        }) as retriever_span:

            # Stage 1: Query rewriting
            search_query = query
            if self.rewrite_queries:
                with record_span("retriever.rewrite", {"query.original": query}) as rw_span:
                    search_query = self._rewrite(query)
                    rw_span["query.rewritten"] = search_query
                    rw_span["query.changed"] = search_query != query

            # Stage 2: Embed query
            with record_span("retriever.embed_query", {"query": search_query}) as eq_span:
                query_vector = self.embedder.embed_query(search_query)
                eq_span["embedding.dims"] = len(query_vector)

            # Stage 3: Vector search
            with record_span("retriever.vector_search", {
                "query": search_query,
                "top_k": k,
            }) as vs_span:
                candidates = self.vector_store.query(query_vector, top_k=k, where=where)
                vs_span["results.count"] = len(candidates)
                if candidates:
                    vs_span["results.top_score"] = candidates[0].score
                    vs_span["results.sources"] = str([
                        c.chunk.metadata.get("source", "?") for c in candidates
                    ])

            # Stage 4: Optional reranking
            if self.rerank and candidates:
                with record_span("retriever.rerank", {
                    "candidates": len(candidates),
                    "top_n": self.top_n,
                }) as rr_span:
                    candidates = self._rerank(query, candidates)
                    rr_span["reranked.count"] = len(candidates)

            retriever_span["final.count"] = len(candidates)
            logger.info("Retrieved %d chunks for query: %r", len(candidates), search_query)
            return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rewrite(self, query: str) -> str:
        try:
            response = self._anthropic.messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(query=query),
                }],
            )
            rewritten = response.content[0].text.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as exc:
            logger.warning("Query rewrite failed (%s) — using original", exc)
            return query

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s), rank=i + 1)
            for i, (c, s) in enumerate(ranked[: self.top_n])
        ]