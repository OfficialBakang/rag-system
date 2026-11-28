"""Data-only migration bundle. Export must operate on a disposable legacy copy."""

import hashlib
import json
from pathlib import Path

from core.models import Chunk, EmbeddedChunk
from core.vector_store import VectorStore


def read_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    raw = (path / "records.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Bundle checksum mismatch")
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != manifest["count"]:
        raise ValueError("Incomplete bundle")
    ids = set()
    for record in records:
        if set(record) != {"id", "text", "metadata", "vector"} or record["id"] in ids:
            raise ValueError("Invalid or duplicate record")
        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError("Invalid identifier")
        ids.add(record["id"])
        vector = VectorStore._vector(record["vector"])
        if len(vector) != manifest["dimension"]:
            raise ValueError("Dimension mismatch")
        metadata = record["metadata"]
        if not isinstance(metadata, dict) or not isinstance(record["text"], str):
            raise TypeError("Invalid payload")
        if not isinstance(metadata.get("doc_id"), str):
            raise TypeError("Missing document identifier")
        if type(metadata.get("chunk_index")) is not int or type(metadata.get("token_count")) is not int:
            raise ValueError("Missing chunk fields")
        json.dumps(metadata, allow_nan=False)
    return records


def import_bundle(bundle, target, batch_size=100):
    # Validate the entire immutable bundle before opening the target.
    records = read_bundle(bundle)
    target = Path(target)
    fingerprint = (Path(bundle) / "manifest.json").read_text()
    marker = target / "migration.json"
    if target.exists() and any(target.iterdir()) and not marker.exists():
        raise ValueError("Import target must be empty or owned by this migration")
    if marker.exists() and marker.read_text() != fingerprint:
        raise ValueError("Target belongs to a different migration")
    target.mkdir(parents=True, exist_ok=True)
    marker.write_text(fingerprint)
    store = VectorStore(str(target))
    try:
        for start in range(0, len(records), batch_size):
            batch = []
            for row in records[start : start + batch_size]:
                meta = row["metadata"]
                chunk = Chunk(
                    row["id"],
                    meta["doc_id"],
                    row["text"],
                    int(meta["chunk_index"]),
                    int(meta["token_count"]),
                    meta,
                )
                batch.append(EmbeddedChunk(chunk, row["vector"], "migration"))
            store.upsert(batch)
    finally:
        store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    import_bundle(args.bundle, args.target)
