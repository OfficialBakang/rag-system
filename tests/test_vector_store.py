from dataclasses import replace


def test_empty_collection(vector_store):
    assert vector_store.count() == vector_store.upsert([]) == 0
    assert vector_store.query([1.0, 0.0]) == []
    assert vector_store.list_sources() == []


def test_similarity_filter_reconstruction_and_delete(vector_store, embedded):
    first = embedded.chunk
    second = replace(
        first,
        chunk_id="other",
        doc_id="two",
        index=3,
        text="Elsewhere",
        metadata={"source": "other.md"},
    )
    records = [replace(embedded, chunk=second, embedding=[0.0, 1.0]), embedded]
    for _ in range(2):
        assert vector_store.upsert(records) == 2
        assert vector_store.count() == 2
    hits = vector_store.query([1.0, 0.0], top_k=20)
    assert [(h.rank, h.score, h.chunk.chunk_id) for h in hits] == [
        (1, 1.0, first.chunk_id),
        (2, 0.0, second.chunk_id),
    ]
    (filtered,) = vector_store.query([1.0, 0.0], where={"source": "other.md"})
    assert (
        filtered.chunk.text,
        filtered.chunk.doc_id,
        filtered.chunk.index,
        filtered.chunk.token_count,
        filtered.chunk.metadata["source"],
    ) == ("Elsewhere", "two", 3, second.token_count, "other.md")
    assert vector_store.list_sources() == ["docs/report.txt", "other.md"]
    assert vector_store.delete_document(first.doc_id) == 1
    assert vector_store.delete_document(first.doc_id) == 0
    assert vector_store.count() == 1
