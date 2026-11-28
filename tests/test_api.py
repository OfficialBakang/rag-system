import json
import os

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(services):
    with TestClient(api.app, headers={"X-API-Key": os.environ["RAG_ADMIN_API_KEY"]}) as session:
        yield session


def test_health_and_schema(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/query/stream" in schema.json()["paths"]
    assert (
        "answer" in schema.json()["components"]["schemas"]["QueryResponse"]["required"]
    )


def test_minimum_ingest_text(client, services):
    summary = {"documents": 1, "chunks": 2, "tokens": 12, "collection_total": 10}
    services.pipeline.ingest_text.return_value = summary
    response = client.post("/ingest/text", json={"text": "0123456789"})
    assert response.status_code == 200
    assert response.json() == dict(status="ok", elapsed_ms=0, **summary)
    services.pipeline.ingest_text.assert_called_once_with(
        "0123456789", source="api-upload"
    )


@pytest.mark.parametrize(
    ("route", "body", "field"),
    [
        ("/ingest/text", {"text": "123456789"}, "text"),
        ("/query", {"query": "ab"}, "query"),
        ("/query", {"query": "why", "top_k": 0}, "top_k"),
        ("/query", {"query": "why", "top_k": 21}, "top_k"),
    ],
)
def test_validation_before_service_calls(client, services, route, body, field):
    response = client.post(route, json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    services.retriever.retrieve.assert_not_called()
    services.pipeline.ingest_text.assert_not_called()


def test_unsupported_upload(client, services):
    response = client.post("/ingest/file", files={"file": ("image.png", b"abc")})
    assert response.status_code == 415 and response.json()["detail"] == "Unsupported document format"
    services.pipeline.ingest_file.assert_not_called()


@pytest.mark.parametrize(("top_k", "evaluate"), [(1, False), (20, True)])
def test_query_response(client, services, generation, top_k, evaluate):
    response = client.post(
        "/query", json={"query": "why", "top_k": top_k, "evaluate": evaluate}
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "answer",
        "query": "why",
        "sources": ["docs/report.txt"],
        "chunks": [
            {
                "text": "Local passage",
                "score": 0.9,
                "rank": 1,
                "source": "docs/report.txt",
            }
        ],
        "model": "model",
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "latency_ms": 1.5,
        "trace_id": "trace",
        "eval": {"overall_score": 0.75} if evaluate else None,
    }
    services.retriever.retrieve.assert_called_once_with("why", top_k=top_k)
    services.generator.generate.assert_called_once_with(
        "why", generation.retrieved_chunks
    )
    if evaluate:
        services.evaluator.evaluate.assert_called_once_with(generation)
    else:
        services.evaluator.evaluate.assert_not_called()


def test_stream_completion(client, services):
    services.generator.generate_stream.return_value = iter(["one", "two"])
    response = client.post("/query/stream", json={"query": "why"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == "data: one\n\ndata: two\n\ndata: [DONE]\n\n"


def test_manual_evaluation(client, services):
    response = client.post(
        "/evaluate", json={"query": "why", "answer": "yes", "context_texts": ["a", "b"]}
    )
    assert response.status_code == 200 and response.json() == {"overall_score": 0.75}
    (result,) = services.evaluator.evaluate.call_args.args
    assert (result.trace_id, result.model, result.source_texts) == (
        "manual",
        "manual",
        ["a", "b"],
    )


def test_metrics_history_and_limits(client, logs):
    traces = [
        {"name": "generator.llm_call", "attributes": {"llm.latency_ms": n}}
        for n in (2, 4)
    ]
    traces.append({"name": "other"})
    evals = [{"overall_score": n} for n in (0.1, 0.7)]
    for path, records in zip(logs, (traces, evals), strict=True):
        path.write_text("\n".join(map(json.dumps, records)))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "collection_size": 8,
        "sources": ["report"],
        "total_traces": 3,
        "total_evals": 2,
        "avg_latency_ms": 3,
        "avg_overall_score": 0.4,
    }
    for route, records in (("traces", traces), ("evals", evals)):
        for limit in (1, 500):
            response = client.get(f"/{route}", params={"limit": limit})
            assert response.status_code == 200
            assert response.json() == {
                route: records[-limit:],
                "count": len(records[-limit:]),
            }
        assert client.get(f"/{route}?limit=501").status_code == 422


def test_sources_and_deletion(client, services):
    services.pipeline.store.delete_document.return_value = 2
    assert client.get("/sources").json() == {"sources": ["report"]}
    response = client.delete("/documents/abc")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "deleted_chunks": 2, "doc_id": "abc"}
    services.pipeline.store.delete_document.assert_called_once_with("abc")


@pytest.mark.parametrize("route", ["/query", "/ingest/text", "/evaluate"])
def test_dependency_errors(client, services, route):
    error = RuntimeError("dependency unavailable")
    services.generator.generate.side_effect = error
    services.pipeline.ingest_text.side_effect = error
    services.evaluator.evaluate.side_effect = error
    response = client.post(
        route,
        json={
            "query": "why",
            "answer": "answer",
            "text": "0123456789",
            "context_texts": [],
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Request failed"
    assert "dependency unavailable" not in response.text


def test_delete_missing_document_is_idempotent(client, services, vector_store):
    services.pipeline.store = vector_store
    response = client.delete("/documents/absent")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "deleted_chunks": 0, "doc_id": "absent"}


def test_delete_missing_collection_returns_unavailable(
    client, services, vector_store, caplog
):
    services.pipeline.store = vector_store
    vector_store._client.delete_collection(vector_store._col.name)
    response = client.delete("/documents/abc")
    assert response.status_code == 503
    assert response.json() == {"detail": "Document store unavailable"}
    assert response.headers["X-Request-ID"] in caplog.text


def test_unexpected_deletion_failure_is_private(client, services, caplog):
    error = RuntimeError("private diagnostic from the storage backend")
    services.pipeline.store.delete_document.side_effect = error
    response = client.delete("/documents/abc")
    assert response.status_code == 500
    assert response.json() == {"detail": "Request failed"}
    assert response.headers["X-Request-ID"] in caplog.text
    assert str(error) not in caplog.text
