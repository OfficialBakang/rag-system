"""Run only against a disposable legacy copy with network disabled."""

import argparse
import hashlib
import json
from pathlib import Path


def export(source, output, collection):
    # This module is isolated from application imports and requirements.
    import chromadb
    from chromadb.config import Settings

    output = Path(output)
    output.mkdir(exist_ok=False)
    client = chromadb.PersistentClient(
        path=source, settings=Settings(anonymized_telemetry=False)
    )
    try:
        col = client.get_collection(collection, embedding_function=None)
        count = col.count()
        dimension = None
        digest = hashlib.sha256()
        with (output / "records.jsonl").open("wb") as stream:
            for offset in range(0, count, 100):
                batch = col.get(
                    offset=offset,
                    limit=100,
                    include=["documents", "metadatas", "embeddings"],
                )
                for identifier, text, meta, vector in zip(
                    batch["ids"],
                    batch["documents"],
                    batch["metadatas"],
                    batch["embeddings"],
                    strict=True,
                ):
                    vector = list(map(float, vector))
                    dimension = dimension or len(vector)
                    raw = (
                        json.dumps(
                            {
                                "id": identifier,
                                "text": text,
                                "metadata": meta,
                                "vector": vector,
                            },
                            allow_nan=False,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                    stream.write(raw)
                    digest.update(raw)
        (output / "manifest.json").write_text(
            json.dumps(
                {"count": count, "dimension": dimension, "sha256": digest.hexdigest()}
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disposable-copy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection", default="rag_documents")
    args = parser.parse_args()
    export(args.disposable_copy, args.output, args.collection)
