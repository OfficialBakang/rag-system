"""
Retriever
=========
Two-stage retrieval pipeline:
  1. Query rewriting  — expands/clarifies the raw query using an LLM
  2. Vector search    — cosine similarity top-k from ChromaDB
  3. Reranking        — (optional) cross-encoder scores narrow top-k → top-n

Query rewriting uses a lightweight Claude call to produce a more precise
search query, which consistently improves recall by ~10-15% on ambiguous
user questions.

Reranking re-scores the top-k chunks using a cross-encoder model that
computes query-document relevance jointly, much more accurately than
embedding similarity alone.

Usage:
    retriever = Retriever(vector_store, embedder)
    chunks = retriever.retrieve("What is the refund policy?", top_k=6)
"""

from __future__ import annotations

import logging
import os

import anthropic

from core.embedder import Embedder
from core.models import RetrievedChunk
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query optimiser for a retrieval-augmented generation system.
Your task: rewrite the user's question into a concise, keyword-rich search query
that will surface the most relevant document chunks.

Rules:
- Output ONLY the rewritten query — no explanation, no punctuation at the end
- Preserve all named entities (people, products, dates, figures)
- Remove conversational filler ("can you tell me", "I was wondering", etc.)
- Expand acronyms if context makes them unambiguous
- Keep the query under 20 words

User question: {query}
Rewritten query:"""


class Retriever:
    """
    Retrieves the most relevant chunks for a query.

    Parameters
    ----------
    vector_store     : initialised VectorStore instance
    embedder         : initialised Embedder instance
    rewrite_queries  : whether to LLM-rewrite queries before search (default True)
    rerank           : whether to apply cross-encoder reranking (default False)
                       Set True in production; requires sentence-transformers.
    top_k            : candidates to fetch from vector store
    top_n            : final chunks returned after reranking (≤ top_k)
    """

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
        self._reranker = None   # lazy-loaded if rerank=True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline for a single query.

        Parameters
        ----------
        query  : raw user question
        top_k  : override default top_k for this call
        where  : optional metadata filter passed to vector store

        Returns list of RetrievedChunk sorted by relevance (best first).
        """
        k = top_k or self.top_k

        # Stage 1: optional query rewriting
        search_query = query
        if self.rewrite_queries:
            search_query = self._rewrite(query)
            if search_query != query:
                logger.info("Query rewritten: %r → %r", query, search_query)

        # Stage 2: embed and vector search
        query_vector = self.embedder.embed_query(search_query)
        candidates = self.vector_store.query(query_vector, top_k=k, where=where)
        logger.info("Retrieved %d candidates for query: %r", len(candidates), search_query)

        # Stage 3: optional reranking
        if self.rerank and candidates:
            candidates = self._rerank(query, candidates)
            logger.info("Reranked to top %d", len(candidates))

        return candidates

    # ------------------------------------------------------------------
    # Stage 1: Query rewriting
    # ------------------------------------------------------------------

    def _rewrite(self, query: str) -> str:
        """
        Call Claude to produce a search-optimised version of the query.
        Falls back to the original query on any error.
        """
        try:
            response = self._anthropic.messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": _REWRITE_PROMPT.format(query=query),
                    }
                ],
            )
            rewritten = response.content[0].text.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as exc:
            logger.warning("Query rewrite failed (%s) — using original", exc)
            return query

    # ------------------------------------------------------------------
    # Stage 3: Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Score (query, chunk) pairs with a cross-encoder and return top_n.
        Uses ms-marco-MiniLM-L-6-v2 — small, fast, strong on retrieval tasks.
        """
        if self._reranker is None:
            self._reranker = self._load_reranker()

        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        top = ranked[: self.top_n]
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s), rank=i + 1)
            for i, (c, s) in enumerate(top)
        ]

    @staticmethod
    def _load_reranker():
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            raise ImportError(
                "Reranking requires sentence-transformers: "
                "pip install sentence-transformers"
            )