"""
RAG System — FastAPI Backend
=============================
Exposes the full pipeline as a REST API.

Endpoints:
  POST /ingest/text          — ingest raw text
  POST /ingest/file          — ingest an uploaded file
  POST /query                — run a full RAG query
  POST /query/stream         — streaming query (SSE)
  POST /evaluate             — evaluate a generation result
  GET  /metrics              — pipeline health + aggregate stats
  GET  /traces               — recent OTel span records
  GET  /evals                — recent eval results
  GET  /sources              — list ingested documents
  DELETE /documents/{doc_id} — remove a document

Usage:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import anthropic
import openai
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api_security import Access, Boundary, Role, ai_slot, authorize, origins
from core.vector_store import StorageUnavailable
from rag_config import ResourceLimit, flag, load_settings, smoke_test_mode

load_settings()
# SDK and HTTP transport debug logs can include request content, headers or URLs.
for component in (
    "openai",
    "anthropic",
    "httpx",
    "httpcore",
    "pypdf",
    "python_multipart",
    "multipart",
):
    logging.getLogger(component).addHandler(logging.NullHandler())
    logging.getLogger(component).propagate = False
logging.getLogger("uvicorn.access").disabled = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline components (initialised once on startup)
# ---------------------------------------------------------------------------
_pipeline = None
_embedder = None
_retriever = None
_generator = None
_evaluator = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from core.ingestion_pipeline import IngestionPipeline

        _pipeline = IngestionPipeline()
    return _pipeline


def get_retriever():
    global _retriever, _embedder
    if _retriever is None:
        from core.embedder import Embedder
        from core.retriever import Retriever

        _embedder = Embedder()
        _retriever = Retriever(
            get_pipeline().store,
            _embedder,
            rewrite_queries=True,
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "6")),
        )
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        from core.generator import Generator

        _generator = Generator()
    return _generator


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        from core.evals import EvalRunner

        _evaluator = EvalRunner()
    return _evaluator


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.access = Access()
    if not 1 <= int(os.getenv("MAX_TOKENS", "1024")) <= 8192:
        raise ValueError("MAX_TOKENS must be between 1 and 8192")
    app.state.smoke = smoke_test_mode()
    if app.state.smoke:
        from core.vector_store import VectorStore

        logger.info("Smoke-test mode enabled. Initialising local ChromaDB only.")
        app.state.store = VectorStore()
        try:
            app.state.store.count()
            yield
        finally:
            app.state.store.close()
        return

    get_pipeline()
    get_retriever()
    get_generator()
    logger.info(
        "Pipeline ready — %d chunks in collection", get_pipeline().store.count()
    )
    yield
    logger.info("Shutting down RAG API.")


def get_store():
    return app.state.store if app.state.smoke else get_pipeline().store


docs_enabled = flag("RAG_PUBLIC_DOCS", True)
app = FastAPI(
    title="RAG System API",
    description="Production RAG pipeline with telemetry and evals",
    version="1.0.0",
    lifespan=lifespan,
    telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
    openapi_url="/openapi.json" if docs_enabled else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins(),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)


app.add_middleware(Boundary)
protected = APIRouter(dependencies=[Depends(authorize)])


@app.exception_handler(RequestValidationError)
async def invalid_input(request, exc):
    # Never echo Pydantic input values, context or attacker-controlled field names.
    fields = {
        "body",
        "query",
        "text",
        "top_k",
        "file",
        "limit",
        "evaluate",
        "answer",
        "source",
        "context_texts",
    }
    errors = [
        {"loc": [part for part in e["loc"] if part in fields], "msg": "Invalid value"}
        for e in exc.errors()[:10]
    ]
    return JSONResponse({"detail": errors}, 422)


@app.exception_handler(ResourceLimit)
async def over_limit(request, exc):
    return JSONResponse({"detail": "Document exceeds processing limits"}, 413)


async def provider_error(request, exc):
    status = 503 if getattr(exc, "status_code", None) in (401, 403, 429) else 502
    return JSONResponse({"detail": "AI provider unavailable"}, status)


for provider_exception in (openai.APIError, anthropic.APIError):
    app.add_exception_handler(provider_exception, provider_error)


class InputModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IngestTextRequest(InputModel):
    text: str = Field(..., min_length=10, description="Raw text to ingest")
    source: str = Field(
        "api-upload",
        min_length=1,
        max_length=256,
        description="Source label for this document",
    )


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    tokens: int
    collection_total: int
    elapsed_ms: float


class QueryRequest(InputModel):
    query: str = Field(..., min_length=3, description="Question to answer")
    top_k: int = Field(default=6, ge=1)
    evaluate: bool = Field(default=False, description="Run evals on this query")


class RetrievedChunkOut(BaseModel):
    text: str
    score: float
    rank: int
    source: str


class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: list[str]
    chunks: list[RetrievedChunkOut]
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    trace_id: str | None
    eval: dict | None = None


class EvalRequest(InputModel):
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    context_texts: list[str] = Field(max_length=100)
    trace_id: str = Field(default="manual", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class MetricsResponse(BaseModel):
    status: str
    collection_size: int
    sources: list[str]
    total_traces: int
    total_evals: int
    avg_latency_ms: float | None
    avg_overall_score: float | None


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------


@protected.post(
    "/ingest/text",
    response_model=IngestResponse,
    dependencies=[Depends(ai_slot, scope="request")],
    tags=["Ingestion"],
)
def ingest_text(request: IngestTextRequest):
    """Ingest a plain-text string into the vector store."""
    if len(request.text) > app.state.access.limits.text_chars:
        raise HTTPException(413, "Text too large")
    summary = get_pipeline().ingest_text(request.text, source=request.source)
    return IngestResponse(
        status="ok",
        documents=summary["documents"],
        chunks=summary["chunks"],
        tokens=summary["tokens"],
        collection_total=summary["collection_total"],
        elapsed_ms=summary.get("elapsed_ms", 0),
    )


@protected.post(
    "/ingest/file",
    response_model=IngestResponse,
    dependencies=[Depends(ai_slot, scope="request")],
    tags=["Ingestion"],
)
def ingest_file(file: Annotated[UploadFile, File()]):
    """Validate bytes before ingestion. Only server-generated paths reach the loader."""
    name = file.filename or ""
    if (
        not name
        or len(name) > 255
        or re.search(r"[\\/:\x00-\x1f]", name)
        or name in {".", ".."}
    ):
        raise HTTPException(422, "Invalid upload filename")
    suffix = Path(name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".html", ".htm"}:
        raise HTTPException(415, "Unsupported document format")
    limits = app.state.access.limits
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("document" + suffix)
        size = 0
        with path.open("wb") as output:
            while block := file.file.read(65536):
                size += len(block)
                if size > limits.upload_bytes:
                    raise HTTPException(413, "Upload too large")
                output.write(block)
        if not size:
            raise HTTPException(422, "Empty document")
        from core.loader import DocumentLoader

        try:
            doc = DocumentLoader().load_file(path)
        except ResourceLimit:
            raise
        except Exception:  # noqa: BLE001 - sanitise untrusted parser and backend failures
            raise HTTPException(422, "Invalid or unreadable document") from None
        if not doc.content.strip():
            raise HTTPException(422, "Document contains no text")
        doc.source = name
        doc.metadata.update(source=name, filename=name)
        summary = get_pipeline().ingest_documents([doc])
    return IngestResponse(
        status="ok",
        elapsed_ms=summary.get("elapsed_ms", 0),
        **{
            k: summary[k] for k in ("documents", "chunks", "tokens", "collection_total")
        },
    )


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------


def validate_query(request: QueryRequest, http: Request):
    access = app.state.access
    if request.top_k > access.limits.top_k:
        raise HTTPException(
            422, [{"loc": ["body", "top_k"], "msg": "Retrieval limit exceeded"}]
        )
    if len(request.query) > access.limits.query_chars:
        raise HTTPException(422, "Query too long")
    if request.evaluate:
        if http.state.role != Role.ADMIN:
            raise HTTPException(403, "Evaluation requires administrator permission")
        if http.url.path == "/query/stream":
            raise HTTPException(422, "Evaluation requires the non-streaming query endpoint")
        access.budget.charge(Role.ADMIN, "evaluate")


@protected.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(validate_query), Depends(ai_slot, scope="request")],
    tags=["Query"],
)
def query(request: QueryRequest):
    """Run a full RAG query and return a grounded answer."""
    chunks = get_retriever().retrieve(request.query, top_k=request.top_k)
    result = get_generator().generate(request.query, chunks)

    chunks_out = [
        RetrievedChunkOut(
            text=rc.chunk.text,
            score=rc.score,
            rank=rc.rank,
            source=rc.chunk.metadata.get("source", "unknown"),
        )
        for rc in chunks
    ]

    eval_out = None
    if request.evaluate:
        eval_result = get_evaluator().evaluate(result)
        eval_out = eval_result.to_dict()

    return QueryResponse(
        answer=result.answer,
        query=result.query,
        sources=result.sources,
        chunks=chunks_out,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        trace_id=result.trace_id,
        eval=eval_out,
    )


@protected.post(
    "/query/stream", dependencies=[Depends(ai_slot, scope="request")], tags=["Query"]
)
def query_stream(request: QueryRequest, http: Request):
    """
    Streaming RAG query — returns answer tokens as Server-Sent Events.
    Connect with EventSource in the browser.
    """
    chunks = get_retriever().retrieve(request.query, top_k=request.top_k)

    def token_stream():
        try:
            for token in get_generator().generate_stream(request.query, chunks):
                yield "".join(f"data: {line}\n" for line in token.split("\n")) + "\n"
        except Exception:  # noqa: BLE001 - sanitise untrusted parser and backend failures
            logger.warning("stream_failed request_id=%s", http.state.request_id)
            yield "event: error\ndata: Generation interrupted\n\n"
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Eval endpoint
# ---------------------------------------------------------------------------


@protected.post(
    "/evaluate", dependencies=[Depends(ai_slot, scope="request")], tags=["Evals"]
)
def evaluate(request: EvalRequest):
    """Evaluate a query-answer pair. Provide context_texts from a prior /query call."""
    from core.models import Chunk, GenerationResult, RetrievedChunk

    limits = app.state.access.limits
    if (
        len(request.query) > limits.query_chars
        or len(request.answer) + sum(map(len, request.context_texts))
        > limits.context_chars
        or any(not text.strip() for text in request.context_texts)
    ):
        raise HTTPException(422, "Invalid evaluation input")

    # Build a minimal GenerationResult from the request
    fake_chunks = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"manual-{i}",
                doc_id="manual",
                text=text,
                index=i,
                token_count=len(text.split()),
                metadata={"source": "manual"},
            ),
            score=1.0,
            rank=i + 1,
        )
        for i, text in enumerate(request.context_texts)
    ]

    gen_result = GenerationResult(
        answer=request.answer,
        query=request.query,
        retrieved_chunks=fake_chunks,
        model="manual",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0,
        trace_id=request.trace_id,
    )

    eval_result = get_evaluator().evaluate(gen_result)
    return eval_result.to_dict()


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------


@protected.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
async def metrics():
    """Pipeline health check + aggregate statistics."""
    from core.evals import load_evals
    from core.telemetry import load_traces

    traces = load_traces()
    evals = load_evals()

    llm_spans = [
        t
        for t in traces
        if t["name"] == "generator.llm_call"
        and "llm.latency_ms" in t.get("attributes", {})
    ]
    avg_latency = (
        round(
            sum(t["attributes"]["llm.latency_ms"] for t in llm_spans) / len(llm_spans),
            1,
        )
        if llm_spans
        else None
    )
    avg_score = (
        round(sum(e["overall_score"] for e in evals) / len(evals), 3) if evals else None
    )

    return MetricsResponse(
        status="ok",
        collection_size=get_store().count(),
        sources=get_store().list_sources(),
        total_traces=len(traces),
        total_evals=len(evals),
        avg_latency_ms=avg_latency,
        avg_overall_score=avg_score,
    )


@protected.get("/traces", tags=["Observability"])
async def traces(limit: int = Query(default=50, ge=1, le=500)):
    """Return recent OTel span records from traces.jsonl."""
    from core.telemetry import load_traces

    records = load_traces(limit=limit)
    return {"traces": records, "count": len(records)}


@protected.get("/evals", tags=["Observability"])
async def evals(limit: int = Query(default=50, ge=1, le=500)):
    """Return recent eval results from evals.jsonl."""
    from core.evals import load_evals

    records = load_evals(limit=limit)
    return {"evals": records, "count": len(records)}


@protected.get("/sources", tags=["Observability"])
async def sources():
    """List all source documents currently in the vector store."""
    return {"sources": get_store().list_sources()}


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


@protected.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove all chunks belonging to a document by doc_id."""
    try:
        deleted = get_store().delete_document(doc_id)
    except StorageUnavailable:
        raise HTTPException(503, "Document store unavailable") from None
    return {"status": "ok", "deleted_chunks": deleted, "doc_id": doc_id}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health():
    try:
        get_store().count()
    except Exception:  # noqa: BLE001 - sanitise untrusted parser and backend failures
        raise HTTPException(503, "Service unavailable") from None
    return {"status": "ok"}


@app.get("/live", tags=["Health"])
async def live():
    return {"status": "ok"}


@protected.get("/ready", tags=["Health"])
async def ready():
    try:
        get_store().count()
        database = True
    except Exception:  # noqa: BLE001 - sanitise untrusted parser and backend failures
        logger.warning("database_probe_failed")
        database = False
    smoke = app.state.smoke
    configured = all(
        os.getenv(key, "").strip() for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    rag_ready = database and configured and not smoke
    return JSONResponse(
        status_code=200 if database and (smoke or configured) else 503,
        content={
            "mode": "smoke" if smoke else "normal",
            "database_available": database,
            "providers_configured": configured,
            "rag_ready": rag_ready,
        },
    )


app.include_router(protected)
