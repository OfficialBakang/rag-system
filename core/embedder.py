"""
Embedder
========
Converts Chunk objects into EmbeddedChunk objects by calling the
OpenAI embeddings API in batches.

Uses text-embedding-3-small by default (1536 dims, cheap, fast).
Batches up to 100 chunks per API call to stay within rate limits.

Usage:
    embedder = Embedder()
    embedded = embedder.embed(chunks)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

from openai import APIError, OpenAI

from core.models import Chunk, EmbeddedChunk
from rag_config import Limits, ResourceLimit

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_BATCH_SIZE = 100       # OpenAI allows up to 2048 inputs, 100 is conservative
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0      # seconds


def _batched(items: list, size: int) -> Iterator[list]:
    """Yield successive fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    """
    Batch embedder using the OpenAI embeddings endpoint.

    Parameters
    ----------
    model      : embedding model name (default: text-embedding-3-small)
    batch_size : chunks per API call (default: 100)
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        batch_size: int = _BATCH_SIZE,
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0, timeout=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of Chunk objects. Returns EmbeddedChunk objects
        in the same order as the input list.
        """
        if not chunks:
            return []

        logger.info(
            "Embedding %d chunks in batches of %d (model=%s)",
            len(chunks),
            self.batch_size,
            self.model,
        )

        embedded: list[EmbeddedChunk] = []
        total_tokens = 0
        t0 = time.perf_counter()

        max_calls = Limits.load().embedding_batches
        if len(chunks) > max_calls * self.batch_size:
            raise ResourceLimit()
        calls = iter(range(max_calls))
        for batch_num, batch in enumerate(_batched(chunks, self.batch_size), start=1):
            texts = [c.text for c in batch]
            vectors, usage = self._embed_with_retry(texts, calls)
            total_tokens += usage

            for chunk, vector in zip(batch, vectors):
                embedded.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        embedding=vector,
                        embedding_model=self.model,
                    )
                )
            logger.debug("Batch %d/%d done", batch_num, -(-len(chunks) // self.batch_size))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Embedded %d chunks — %d tokens — %.0f ms total",
            len(embedded),
            total_tokens,
            elapsed,
        )
        return embedded

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.
        Returns the raw embedding vector.
        """
        vectors, _ = self._embed_with_retry([query])
        return vectors[0]

    # ------------------------------------------------------------------
    # Internal: retry logic
    # ------------------------------------------------------------------

    def _embed_with_retry(
        self, texts: list[str], calls: Iterator[int] | None = None
    ) -> tuple[list[list[float]], int]:
        """
        Call the embeddings API with exponential-backoff retry.
        Returns (list_of_vectors, total_prompt_tokens).
        """
        last_exc: Exception | None = None

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            if calls is not None and next(calls, None) is None:
                raise ResourceLimit()
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",
                )
                vectors = [item.embedding for item in response.data]
                tokens = response.usage.total_tokens
                return vectors, tokens

            except Exception as exc:  # noqa: BLE001 - bounded retries for transport failures
                last_exc = exc
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding API error (attempt %d/%d) — retrying in %.1fs",
                    attempt,
                    _RETRY_ATTEMPTS,
                    wait,
                )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(wait)

        if isinstance(last_exc, APIError):
            raise last_exc
        raise RuntimeError(
            f"Embedding failed after {_RETRY_ATTEMPTS} attempts"
        ) from last_exc