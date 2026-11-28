import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from core.vector_store import StorageUnavailable, VectorStore


def test_restart_and_threads(tmp_path, embedded):
    path = tmp_path / "store"
    store = VectorStore(str(path))
    store.upsert([embedded])
    identifier = store.point_id(embedded.chunk.chunk_id)
    store.close()
    store = VectorStore(str(path))
    assert store.point_id(embedded.chunk.chunk_id) == identifier
    with ThreadPoolExecutor(3) as pool:
        assert (
            list(
                pool.map(lambda _: store.query([1.0, 0.0])[0].chunk.chunk_id, range(9))
            )
            == [embedded.chunk.chunk_id] * 9
        )
    assert store.count() == 1
    store.close()
    store.close()
    with pytest.raises(StorageUnavailable):
        store.count()


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0], [1.0, 0.0, 0.0]]
)
def test_batch_rejection(vector_store, embedded, vector):
    vector_store.upsert([embedded])
    with pytest.raises(ValueError):
        vector_store.upsert(
            [
                replace(embedded, chunk=replace(embedded.chunk, chunk_id="new")),
                replace(embedded, embedding=vector),
            ]
        )
    assert vector_store.count() == 1


def test_typed_metadata_and_filters(vector_store, embedded):
    chunk = replace(
        embedded.chunk,
        metadata={
            "source": "a",
            "n": 1.5,
            "flag": True,
            "nested": {"a": 1},
            "doc_id": "user-value",
        },
    )
    vector_store.upsert([replace(embedded, chunk=chunk)])
    result = vector_store.query(
        [1.0, 0.0], where={"$and": [{"n": {"$gt": 1}}, {"flag": True}]}
    )[0]
    assert result.chunk.metadata == chunk.metadata
    assert result.chunk.doc_id == chunk.doc_id
    assert vector_store.query([1.0, 0.0], where={"source": {"$nin": ["a"]}}) == []
    with pytest.raises(ValueError):
        vector_store.query([1.0, 0.0], where={"source": {"$regex": "a"}})


def test_legacy_rejected(tmp_path):
    (tmp_path / "chroma.sqlite3").touch()
    with pytest.raises(StorageUnavailable):
        VectorStore(str(tmp_path))
