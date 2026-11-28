import hashlib
import json

import pytest

from core.vector_store import VectorStore
from scripts.migration.import_bundle import import_bundle, read_bundle


def bundle(path):
    path.mkdir()
    records = [
        {
            "id": str(i),
            "text": "synthetic " + str(i),
            "metadata": {
                "doc_id": "doc",
                "chunk_index": i,
                "token_count": 2,
                "source": "fixture",
                "flag": True,
            },
            "vector": [float(i + 1), 1.0],
        }
        for i in range(3)
    ]
    raw = "".join(json.dumps(row) + "\n" for row in records).encode()
    (path / "records.jsonl").write_bytes(raw)
    (path / "manifest.json").write_text(
        json.dumps(
            {"count": 3, "dimension": 2, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    )
    return records


def test_import_recovery_and_integrity(tmp_path, monkeypatch):
    source = tmp_path / "export"
    records = bundle(source)
    target = tmp_path / "target"
    original = VectorStore.upsert
    calls = []

    def interrupt(self, rows):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("interrupted")
        return original(self, rows)

    monkeypatch.setattr(VectorStore, "upsert", interrupt)
    with pytest.raises(OSError):
        import_bundle(source, str(target), batch_size=1)
    monkeypatch.setattr(VectorStore, "upsert", original)
    import_bundle(source, str(target), batch_size=1)
    import_bundle(source, str(target))
    store = VectorStore(str(target))
    hits = store.query([1.0, 1.0], top_k=3)
    assert {h.chunk.chunk_id for h in hits} == {r["id"] for r in records}
    for h in hits:
        row = records[int(h.chunk.chunk_id)]
        assert h.chunk.text == row["text"] and h.chunk.metadata == row["metadata"]
    assert store.count() == 3
    assert store.delete_document("doc") == 3
    store.close()


def test_corruption(tmp_path):
    source = tmp_path / "export"
    bundle(source)
    (source / "records.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(source)
