"""
Ingestion Pipeline
==================
Orchestrates the full ingestion flow:
  DocumentLoader → Chunker → Embedder → VectorStore

This is the entry point for adding new documents to the system.
It handles batching, progress logging, and returns a summary dict
that the telemetry layer can attach to a span.

Usage:
    pipeline = IngestionPipeline()

    # Ingest a directory
    summary = pipeline.ingest_directory("./data/docs")

    # Ingest a single file
    summary = pipeline.ingest_file("./data/docs/report.pdf")

    # Ingest raw text (e.g. from an API endpoint)
    summary = pipeline.ingest_text("Some content...", source="api-upload")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from core.chunker import Chunker
from core.embedder import Embedder
from core.loader import DocumentLoader
from core.models import Document
from core.vector_store import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Parameters
    ----------
    chunk_size    : target tokens per chunk (default from env)
    chunk_overlap : overlap tokens between chunks (default from env)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = Chunker(
            chunk_size=chunk_size or int(os.getenv("CHUNK_SIZE", 400)),
            chunk_overlap=chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 80)),
        )
        self.embedder = Embedder()
        self.store = vector_store or VectorStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> dict:
        """Ingest a single file. Returns a telemetry-friendly summary dict."""
        doc = self.loader.load_file(path)
        return self._process_documents([doc])

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict:
        """Ingest all supported files in a directory."""
        docs = self.loader.load_directory(directory, recursive=recursive)
        if not docs:
            logger.warning("No documents found in %s", directory)
            return {"documents": 0, "chunks": 0, "tokens": 0, "elapsed_ms": 0}
        return self._process_documents(docs)

    def ingest_text(self, text: str, source: str = "inline") -> dict:
        """Ingest a raw text string (e.g. from an API upload endpoint)."""
        doc = self.loader.load_text(text, source=source)
        return self._process_documents([doc])

    def ingest_documents(self, docs: list[Document]) -> dict:
        """Ingest pre-loaded Document objects directly."""
        return self._process_documents(docs)

    # ------------------------------------------------------------------
    # Internal: pipeline execution
    # ------------------------------------------------------------------

    def _process_documents(self, docs: list[Document]) -> dict:
        t0 = time.perf_counter()
        logger.info("Starting ingestion of %d document(s)", len(docs))

        # Step 1: Chunk
        chunks = self.chunker.chunk_many(docs)
        total_tokens = sum(c.token_count for c in chunks)
        logger.info(
            "Chunking complete — %d chunks, %d total tokens", len(chunks), total_tokens
        )

        # Step 2: Embed
        embedded = self.embedder.embed(chunks)

        # Step 3: Store
        upserted = self.store.upsert(embedded)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        summary = {
            "documents": len(docs),
            "chunks": len(chunks),
            "tokens": total_tokens,
            "upserted": upserted,
            "collection_total": self.store.count(),
            "elapsed_ms": elapsed_ms,
            "sources": [d.source for d in docs],
        }
        logger.info(
            "Ingestion complete — %d docs, %d chunks, %.0f ms",
            len(docs),
            len(chunks),
            elapsed_ms,
        )
        return summary