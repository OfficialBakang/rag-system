from dataclasses import replace

import pytest

from core.ingestion_pipeline import IngestionPipeline
from core.models import EmbeddedChunk


@pytest.fixture
def pipeline(vector_store, mocker):
    client = mocker.patch("core.ingestion_pipeline.Embedder").return_value
    client.batch_size = 100
    client.model = "offline"
    client.embed.side_effect = lambda chunks: [
        EmbeddedChunk(c, [1.0, 0.0], client.model) for c in chunks
    ]
    return IngestionPipeline(chunk_size=8, chunk_overlap=0, vector_store=vector_store)


def test_ingest_text_stage_order_and_summary(pipeline, mocker):
    calls = mocker.Mock()
    for stage, method in (
        (pipeline.loader, "load_text"),
        (pipeline.chunker, "chunk_many"),
        (pipeline.store, "upsert"),
    ):
        calls.attach_mock(mocker.spy(stage, method), method)
    calls.attach_mock(pipeline.embedder.embed, "embed")
    summary = pipeline.ingest_text("one two three", source="notes")
    assert [call[0] for call in calls.mock_calls] == [
        "load_text",
        "chunk_many",
        "embed",
        "upsert",
    ]
    assert summary == {
        "documents": 1,
        "chunks": 1,
        "tokens": 3,
        "upserted": 1,
        "collection_total": 1,
        "sources": ["notes"],
    }


def test_file_and_multiple_documents(pipeline, document_file, document):
    summary = pipeline.ingest_file(document_file)
    assert summary["sources"] == [str(document_file)]
    assert summary["upserted"] == 1
    docs = [document, replace(document, doc_id="other", source="second")]
    summary = pipeline.ingest_documents(docs)
    assert summary == {
        "documents": 2,
        "chunks": 2,
        "tokens": 4,
        "upserted": 2,
        "collection_total": 3,
        "sources": [document.source, "second"],
    }


def test_empty_directory_skips_embedding(pipeline, tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    assert pipeline.ingest_directory(directory) == {
        "documents": 0,
        "chunks": 0,
        "tokens": 0,
        "elapsed_ms": 0,
    }
    pipeline.embedder.embed.assert_not_called()


def test_embedding_failure_prevents_storage(pipeline, mocker):
    upsert = mocker.spy(pipeline.store, "upsert")
    pipeline.embedder.embed.side_effect = RuntimeError("embedding failed")
    with pytest.raises(RuntimeError, match="embedding failed"):
        pipeline.ingest_text("one two")
    upsert.assert_not_called()
    assert pipeline.store.count() == 0


@pytest.mark.parametrize("limit", ["CHUNKS", "EMBEDDING_BATCHES"])
def test_excessive_ingestion_never_embeds(pipeline, document, monkeypatch, limit):
    from rag_config import ResourceLimit

    monkeypatch.setenv("RAG_LIMIT_" + limit, "1")
    pipeline.embedder.batch_size = 1
    with pytest.raises(ResourceLimit):
        pipeline.ingest_documents([document, document])
    pipeline.embedder.embed.assert_not_called()
    assert pipeline.store.count() == 0
