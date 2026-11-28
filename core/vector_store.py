"""Persistent single-process Qdrant boundary. Storage and backups must be trusted."""

import json
import math
import os
import re
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from core.models import Chunk, RetrievedChunk


class StorageUnavailable(RuntimeError):
    """The configured store cannot serve the requested operation."""


def metadata_filter(where):
    if not isinstance(where, dict) or len(where) != 1:
        raise ValueError("A filter must contain one field or logical operator")
    key, value = next(iter(where.items()))
    if key in {"$and", "$or"}:
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError("Logical filters require at least two conditions")
        conditions = [metadata_filter(item) for item in value]
        return models.Filter(**{"must" if key == "$and" else "should": conditions})
    if key.startswith("$") or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("Unsupported metadata field")
    operator, operand = ("$eq", value)
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError("Invalid filter operator")
        operator, operand = next(iter(value.items()))
    field = "metadata." + key
    if operator in {"$eq", "$ne"}:
        if isinstance(operand, float):
            condition = models.FieldCondition(
                key=field, range=models.Range(gte=operand, lte=operand)
            )
        elif isinstance(operand, (str, int, bool)):
            condition = models.FieldCondition(
                key=field, match=models.MatchValue(value=operand)
            )
        else:
            raise ValueError("Invalid filter value")
        return models.Filter(
            **{"must_not" if operator == "$ne" else "must": [condition]}
        )
    if operator in {"$gt", "$gte", "$lt", "$lte"} and type(operand) in {float, int}:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=field, range=models.Range(**{operator[1:]: operand})
                )
            ]
        )
    if operator in {"$in", "$nin"} and isinstance(operand, list) and operand:
        parts = [metadata_filter({key: item}) for item in operand]
        return models.Filter(**{"must_not" if operator == "$nin" else "should": parts})
    raise ValueError("Unsupported metadata filter")


class VectorStore:
    def __init__(self, persist_dir=None, collection=None):
        self.path = Path(
            persist_dir or os.getenv("QDRANT_PERSIST_DIR", "./data/qdrant")
        )
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "rag_documents")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.collection):
            raise ValueError("Invalid collection name")
        if (self.path / "chroma.sqlite3").exists():
            raise StorageUnavailable("Legacy storage requires explicit migration")
        legacy = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
        if (
            persist_dir is None
            and (legacy / "chroma.sqlite3").exists()
            and not (self.path / "meta.json").exists()
        ):
            raise StorageUnavailable(
                "Legacy storage detected; explicit migration required"
            )
        self.lock = threading.RLock()
        self.closed = False
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(
            path=str(self.path), force_disable_check_same_thread=True
        )
        self.dimension = None
        if self.client.collection_exists(self.collection):
            self.dimension = self.client.get_collection(
                self.collection
            ).config.params.vectors.size

    def _check(self):
        if self.closed:
            raise StorageUnavailable("Store closed")
        if self.dimension is not None and not self.client.collection_exists(
            self.collection
        ):
            raise StorageUnavailable("Collection unavailable")

    def point_id(self, chunk_id):
        return str(uuid5(NAMESPACE_URL, self.collection + ":" + chunk_id))

    @staticmethod
    def _vector(vector):
        values = list(vector)
        if not values or any(
            type(v) not in {int, float} or not math.isfinite(v) for v in values
        ):
            raise ValueError("Invalid vector")
        if not any(values):
            raise ValueError("Cosine vectors must be nonzero")
        return values

    def upsert(self, embedded_chunks):
        with self.lock:
            self._check()
            points = []
            dimension = self.dimension
            for item in embedded_chunks:
                vector = self._vector(item.embedding)
                dimension = dimension or len(vector)
                if len(vector) != dimension:
                    raise ValueError("Vector dimension mismatch")
                chunk = item.chunk
                if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id:
                    raise ValueError("Invalid chunk identifier")
                metadata = json.loads(json.dumps(chunk.metadata, allow_nan=False))
                points.append(
                    models.PointStruct(
                        id=self.point_id(chunk.chunk_id),
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "index": chunk.index,
                            "token_count": chunk.token_count,
                            "text": chunk.text,
                            "metadata": metadata,
                        },
                    )
                )
            if not points:
                return 0
            if self.dimension is None:
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
                self.dimension = dimension
            self.client.upsert(self.collection, points, wait=True)
            return len(points)

    def query(self, query_vector, top_k=6, where=None):
        vector = self._vector(query_vector)
        condition = metadata_filter(where) if where is not None else None
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Invalid retrieval limit")
        with self.lock:
            self._check()
            if self.dimension is None:
                return []
            if len(vector) != self.dimension:
                raise ValueError("Vector dimension mismatch")
            hits = self.client.query_points(
                self.collection, query=vector, limit=top_k, query_filter=condition
            ).points
            return [
                RetrievedChunk(
                    Chunk(
                        h.payload["chunk_id"],
                        h.payload["doc_id"],
                        h.payload["text"],
                        h.payload["index"],
                        h.payload["token_count"],
                        h.payload["metadata"],
                    ),
                    round(h.score, 4),
                    i + 1,
                )
                for i, h in enumerate(hits)
            ]

    def count(self):
        with self.lock:
            self._check()
            return (
                self.client.count(self.collection, exact=True).count
                if self.dimension
                else 0
            )

    def list_sources(self):
        with self.lock:
            self._check()
            if not self.dimension:
                return []
            sources = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    self.collection, offset=offset, limit=256, with_vectors=False
                )
                sources.update(
                    p.payload["metadata"].get("source", "unknown") for p in points
                )
                if offset is None:
                    return sorted(sources)

    def delete_document(self, doc_id):
        with self.lock:
            self._check()
            if not self.dimension:
                return 0
            condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    )
                ]
            )
            count = self.client.count(
                self.collection, count_filter=condition, exact=True
            ).count
            self.client.delete(
                self.collection, models.FilterSelector(filter=condition), wait=True
            )
            return count

    def close(self):
        with self.lock:
            if not self.closed:
                self.client.close()
                self.closed = True
