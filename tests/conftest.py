"""Install the offline boundary before pytest imports application modules."""

import socket
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from weakref import WeakSet

import dotenv
import pytest


def pytest_configure(config):
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    # Chroma settings can also use python-dotenv through pydantic-settings.
    patch.setattr(dotenv.main.DotEnv, "dict", lambda self: {})
    patch.delenv("RAG_SMOKE_TEST_MODE", raising=False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        patch.delenv(key, raising=False)
    for key, value in {
        "OTEL_ENABLED": "false",
        "OTEL_SDK_DISABLED": "false",
        "OTEL_TRACES_SAMPLER": "always_on",
        "MAX_TOKENS": "1024",
        "RETRIEVAL_TOP_K": "6",
        "RERANK_TOP_N": "3",
        "ANONYMIZED_TELEMETRY": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        patch.setenv(key, value)

    # Windows asyncio uses the stdlib's authenticated loopback socket pair.
    # Permit only its internal connect and writes on the resulting pair.
    creating_pair = ContextVar("creating_socketpair", default=False)
    pairs = WeakSet()
    original_socketpair = socket.socketpair

    def local_socketpair(*args, **kwargs):
        token = creating_pair.set(True)
        try:
            pair = original_socketpair(*args, **kwargs)
        finally:
            creating_pair.reset(token)
        pairs.update(pair)
        return pair

    patch.setattr(socket, "socketpair", local_socketpair)

    def guard(method):
        def checked(sock, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                internal_connect = method.__name__ == "connect" and creating_pair.get()
                pair_write = method.__name__ in ("send", "sendall") and sock in pairs
                if not (internal_connect or pair_write):
                    pytest.fail(f"Network access blocked: socket.{method.__name__}")
            return method(sock, *args, **kwargs)

        return checked

    for name in ("connect", "connect_ex", "sendto", "sendmsg", "send", "sendall"):
        if hasattr(socket.socket, name):
            patch.setattr(socket.socket, name, guard(getattr(socket.socket, name)))

    def block_dns(*args, **kwargs):
        pytest.fail("Network access blocked: DNS lookup")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        patch.setattr(socket, name, block_dns)

    runtime = TemporaryDirectory(prefix="rag-tests-")
    config.add_cleanup(runtime.cleanup)
    root = Path(runtime.name)
    for name, file in (
        ("TRACE_LOG_PATH", "trace.jsonl"),
        ("EVAL_LOG_PATH", "eval.jsonl"),
        ("CHROMA_PERSIST_DIR", "chroma"),
    ):
        patch.setenv(name, str(root / file))


@pytest.fixture(autouse=True)
def logs(tmp_path, monkeypatch):
    from core import evals, telemetry

    trace, evaluation = tmp_path / "traces.jsonl", tmp_path / "evals.jsonl"
    monkeypatch.setattr(telemetry, "_TRACE_LOG", trace)
    monkeypatch.setattr(evals, "_EVAL_LOG", evaluation)
    return trace, evaluation


@pytest.fixture
def document():
    from core.models import Document

    return Document(
        "doc",
        "Local passage",
        "docs/report.txt",
        "txt",
        {"source": "docs/report.txt", "filename": "report.txt"},
    )


@pytest.fixture
def chunk(document):
    from core.models import Chunk

    return Chunk("chunk", document.doc_id, document.content, 0, 2, document.metadata)


@pytest.fixture
def embedded(chunk):
    from core.models import EmbeddedChunk

    return EmbeddedChunk(chunk, [1.0, 0.0], "synthetic")


@pytest.fixture
def retrieved(chunk):
    from core.models import RetrievedChunk

    return RetrievedChunk(chunk, 0.9, 1)


@pytest.fixture
def generation(retrieved):
    from core.models import GenerationResult

    return GenerationResult("answer", "why", [retrieved], "model", 7, 2, 1.5, "trace")


@pytest.fixture
def document_file(tmp_path, document):
    path = tmp_path / "report.txt"
    path.write_text(document.content, encoding="utf-8")
    return path


@pytest.fixture
def vector_store(tmp_path):
    from core.vector_store import VectorStore

    store = VectorStore(str(tmp_path / "database"), "test_documents")
    yield store
    close = getattr(store._client, "close", None)
    if close:
        close()


@pytest.fixture
def anthropic_response():
    def response(text):
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=7, output_tokens=2),
        )

    return response


@pytest.fixture
def messages(mocker):
    return mocker.patch("anthropic.Anthropic").return_value.messages


@pytest.fixture
def openai_response():
    def response(*vectors):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=v) for v in vectors],
            usage=SimpleNamespace(total_tokens=len(vectors) * 2),
        )

    return response
