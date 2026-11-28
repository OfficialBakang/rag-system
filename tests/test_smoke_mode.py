import os

import pytest
from fastapi.testclient import TestClient

import api
from core import telemetry
from core.vector_store import VectorStore


@pytest.fixture
def smoke(monkeypatch, mocker, tmp_path):
    monkeypatch.setenv("RAG_SMOKE_TEST_MODE", "true")
    for target in ("core.embedder.OpenAI", "anthropic.Anthropic"):
        mocker.patch(target, side_effect=AssertionError("AI client initialised"))
    factory = mocker.patch(
        "core.vector_store.VectorStore",
        side_effect=lambda: VectorStore(str(tmp_path / "db")),
    )
    with TestClient(api.app, headers={"X-API-Key": os.environ["RAG_ADMIN_API_KEY"]}) as client:
        yield client
    factory.assert_called_once_with()


def test_smoke_startup_and_storage(smoke):
    assert smoke.get("/live").json() == {"status": "ok"}
    assert smoke.get("/health").json() == {"status": "ok"}
    assert "/query" in smoke.get("/openapi.json").json()["paths"]
    response = smoke.get("/ready?RAG_SMOKE_TEST_MODE=false")
    assert response.status_code == 200
    assert response.json() == {
        "database_available": True,
        "providers_configured": False,
        "mode": "smoke",
        "rag_ready": False,
    }
    assert smoke.get("/sources").json() == {"sources": []}
    assert smoke.get("/metrics").json()["collection_size"] == 0


@pytest.mark.parametrize(
    "path,kwargs",
    [
        ("/ingest/text", {"json": {"text": "0123456789"}}),
        ("/ingest/file", {"files": {"file": ("report.txt", b"sample")}}),
        ("/query", {"json": {"query": "why"}}),
        ("/query/stream", {"json": {"query": "why"}}),
        ("/evaluate", {"json": {"query": "why", "answer": "a", "context_texts": []}}),
    ],
)
def test_provider_routes_disabled(smoke, path, kwargs):
    response = smoke.post(path, **kwargs)
    assert response.status_code == 503
    assert response.json() == {"detail": "AI operations unavailable in smoke-test mode"}


def test_store_failure(smoke):
    store = api.app.state.store
    store._client.delete_collection(store._col.name)
    assert smoke.get("/live").status_code == 200
    assert smoke.get("/health").status_code == 503
    ready = smoke.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["database_available"] is False
    assert ready.json()["rag_ready"] is False


def test_startup_fails_on_inaccessible_store(monkeypatch, mocker):
    monkeypatch.setenv("RAG_SMOKE_TEST_MODE", "true")
    mocker.patch("core.vector_store.VectorStore", side_effect=OSError("unavailable"))
    with pytest.raises(OSError, match="unavailable"), TestClient(api.app, headers={"X-API-Key": os.environ["RAG_ADMIN_API_KEY"]}):
        pass


def test_normal_mode_default_and_configuration(monkeypatch, mocker):
    for name in ("pipeline", "retriever", "generator"):
        mocker.patch.object(api, "get_" + name)
    with TestClient(api.app, headers={"X-API-Key": os.environ["RAG_ADMIN_API_KEY"]}) as client:
        result = client.get("/ready?RAG_SMOKE_TEST_MODE=true")
        assert result.status_code == 503
        assert result.json()["mode"] == "normal"
        assert result.json()["rag_ready"] is False
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.setenv(key, "test-placeholder")
        assert client.get("/ready").json()["rag_ready"] is True


def test_invalid_mode(monkeypatch):
    monkeypatch.setenv("RAG_SMOKE_TEST_MODE", "sometimes")
    with pytest.raises(ValueError, match="must be true or false"), TestClient(api.app, headers={"X-API-Key": os.environ["RAG_ADMIN_API_KEY"]}):
        pass


def test_smoke_suppresses_otlp(monkeypatch, mocker):
    monkeypatch.setenv("RAG_SMOKE_TEST_MODE", "true")
    monkeypatch.setenv("OTEL_ENABLED", "true")
    exporter = mocker.patch(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
    )
    mocker.patch.object(telemetry.trace, "set_tracer_provider")
    telemetry._build_provider().shutdown()
    exporter.assert_not_called()
