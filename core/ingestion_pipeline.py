"""
Ingestion Pipeline
==================
Orchestrates the full ingestion flow:
  DocumentLoader → Chunker → Embedder → VectorStore

Every stage is wrapped in an OTel span via record_span so the full
ingestion trace is visible in Jaeger / the local dashboard.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from core.chunker import Chunker
from core.embedder import Embedder
from core.loader import DocumentLoader
from core.models import Document
from core.telemetry import record_span
from core.vector_store import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)


class IngestionPipeline:
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
        doc = self.loader.load_file(path)
        return self._process_documents([doc])

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict:
        docs = self.loader.load_directory(directory, recursive=recursive)
        if not docs:
            return {"documents": 0, "chunks": 0, "tokens": 0, "elapsed_ms": 0}
        return self._process_documents(docs)

    def ingest_text(self, text: str, source: str = "inline") -> dict:
        doc = self.loader.load_text(text, source=source)
        return self._process_documents([doc])

    def ingest_documents(self, docs: list[Document]) -> dict:
        return self._process_documents(docs)

    # ------------------------------------------------------------------
    # Internal: instrumented pipeline execution
    # ------------------------------------------------------------------

    def _process_documents(self, docs: list[Document]) -> dict:
        sources = [d.source for d in docs]

        with record_span("ingestion.pipeline", {
            "doc.count": len(docs),
            "doc.sources": str(sources),
        }) as pipeline_span:

            # Stage 1: Chunk
            with record_span("ingestion.chunk", {"doc.count": len(docs)}) as chunk_span:
                chunks = self.chunker.chunk_many(docs)
                total_tokens = sum(c.token_count for c in chunks)
                chunk_span["chunk.count"] = len(chunks)
                chunk_span["chunk.total_tokens"] = total_tokens
                chunk_span["chunk.avg_tokens"] = round(total_tokens / max(len(chunks), 1), 1)

            # Stage 2: Embed
            with record_span("ingestion.embed", {"chunk.count": len(chunks)}) as embed_span:
                embedded = self.embedder.embed(chunks)
                embed_span["embed.count"] = len(embedded)
                embed_span["embed.model"] = self.embedder.model

            # Stage 3: Store
            with record_span("ingestion.store", {"chunk.count": len(embedded)}) as store_span:
                upserted = self.store.upsert(embedded)
                store_span["store.upserted"] = upserted
                store_span["store.collection_total"] = self.store.count()

            summary = {
                "documents": len(docs),
                "chunks": len(chunks),
                "tokens": total_tokens,
                "upserted": upserted,
                "collection_total": self.store.count(),
                "sources": sources,
            }
            pipeline_span["ingestion.summary"] = str(summary)
            logger.info("Ingestion complete — %d docs, %d chunks", len(docs), len(chunks))
            return summary