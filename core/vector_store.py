"""
Vector Store
============
ChromaDB wrapper that handles upsert, similarity search, and deletion.

Collection schema per record:
  - id         : chunk_id (deterministic hash)
  - embedding  : float list from the embedder
  - document   : raw chunk text (returned alongside search results)
  - metadata   : source, doc_id, chunk_index, token_count, title, ...

Why ChromaDB?
  - Runs fully in-process, zero infrastructure to set up
  - Persistent on disk (sqlite + HNSW index)
  - Swappable with Qdrant/Pinecone in production without changing retriever logic

Usage:
    store = VectorStore()
    store.upsert(embedded_chunks)
    results = store.query(query_vector, top_k=6)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings

from core.models import Chunk, EmbeddedChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_DEFAULT_COLLECTION  = os.getenv("CHROMA_COLLECTION", "rag_documents")


class VectorStore:
    """
    Persistent ChromaDB collection with a clean upsert / query / delete API.

    Parameters
    ----------
    persist_dir  : directory where ChromaDB stores its sqlite + index files
    collection   : name of the ChromaDB collection
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        collection: str = _DEFAULT_COLLECTION,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "VectorStore ready — collection=%r, docs=%d",
            collection,
            self._col.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Insert or update embedded chunks in the collection.
        Idempotent: re-upserting the same chunk_id overwrites in place.
        Returns the number of records upserted.
        """
        if not embedded_chunks:
            return 0

        ids        = [ec.chunk.chunk_id for ec in embedded_chunks]
        embeddings = [ec.embedding      for ec in embedded_chunks]
        documents  = [ec.chunk.text     for ec in embedded_chunks]
        metadatas  = [self._safe_meta(ec.chunk) for ec in embedded_chunks]

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks — collection total: %d", len(ids), self._col.count())
        return len(ids)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 6,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Cosine similarity search.

        Parameters
        ----------
        query_vector : embedded query from Embedder.embed_query()
        top_k        : number of results to return
        where        : optional ChromaDB metadata filter, e.g. {"source": "report.pdf"}

        Returns a list of RetrievedChunk sorted by descending similarity score.
        """
        kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, self._col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._col.query(**kwargs)

        results: list[RetrievedChunk] = []
        for i, (doc_text, meta, dist) in enumerate(
            zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
        ):
            # ChromaDB returns L2 distance for cosine space — convert to similarity
            similarity = 1.0 - dist

            chunk = Chunk(
                chunk_id=raw["ids"][0][i],
                doc_id=meta.get("doc_id", "unknown"),
                text=doc_text,
                index=int(meta.get("chunk_index", i)),
                token_count=int(meta.get("token_count", 0)),
                metadata=meta,
            )
            results.append(
                RetrievedChunk(chunk=chunk, score=round(similarity, 4), rank=i + 1)
            )

        return results

    def delete_document(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document.
        Returns the number of chunks deleted.
        """
        existing = self._col.get(where={"doc_id": doc_id})
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            self._col.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for doc_id=%r", len(ids_to_delete), doc_id)
        return len(ids_to_delete)

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._col.count()

    def list_sources(self) -> list[str]:
        """Return unique source file names stored in the collection."""
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "unknown") for m in all_meta})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_meta(chunk: Chunk) -> dict:
        """
        ChromaDB metadata values must be str | int | float | bool.
        Flatten and sanitise the chunk metadata dict.
        """
        safe: dict = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.index,
            "token_count": chunk.token_count,
        }
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            else:
                safe[k] = str(v)   # coerce lists, dicts, etc.
        return safe