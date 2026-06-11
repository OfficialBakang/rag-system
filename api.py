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
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline components (initialised once on startup)
# ---------------------------------------------------------------------------
_pipeline  = None
_embedder  = None
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
        _embedder  = Embedder()
        _retriever = Retriever(
            get_pipeline().store,
            _embedder,
            rewrite_queries=True,
            top_k=int(os.getenv("RETRIEVAL_TOP_K", 6)),
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
    logger.info("Starting RAG API — warming up pipeline components...")
    get_pipeline()
    get_retriever()
    get_generator()
    logger.info(
        "Pipeline ready — %d chunks in collection",
        get_pipeline().store.count(),
    )
    yield
    logger.info("Shutting down RAG API.")


app = FastAPI(
    title="RAG System API",
    description="Production RAG pipeline with telemetry and evals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text to ingest")
    source: str = Field("api-upload", description="Source label for this document")


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    tokens: int
    collection_total: int
    elapsed_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Question to answer")
    top_k: int = Field(default=6, ge=1, le=20)
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


class EvalRequest(BaseModel):
    query: str
    answer: str
    context_texts: list[str]
    trace_id: str = "manual"


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

@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """Ingest a plain-text string into the vector store."""
    try:
        summary = get_pipeline().ingest_text(request.text, source=request.source)
        return IngestResponse(
            status="ok",
            documents=summary["documents"],
            chunks=summary["chunks"],
            tokens=summary["tokens"],
            collection_total=summary["collection_total"],
            elapsed_ms=summary.get("elapsed_ms", 0),
        )
    except Exception as exc:
        logger.exception("Ingest text failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a file (PDF, Markdown, TXT, HTML)."""
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "upload.txt").suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".html", ".htm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        summary = get_pipeline().ingest_file(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        return IngestResponse(
            status="ok",
            documents=summary["documents"],
            chunks=summary["chunks"],
            tokens=summary["tokens"],
            collection_total=summary["collection_total"],
            elapsed_ms=summary.get("elapsed_ms", 0),
        )
    except Exception as exc:
        logger.exception("Ingest file failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Run a full RAG query and return a grounded answer."""
    try:
        t0 = time.perf_counter()
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
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Streaming RAG query — returns answer tokens as Server-Sent Events.
    Connect with EventSource in the browser.
    """
    try:
        chunks = get_retriever().retrieve(request.query, top_k=request.top_k)

        async def token_stream():
            for token in get_generator().generate_stream(request.query, chunks):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")
    except Exception as exc:
        logger.exception("Stream query failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Eval endpoint
# ---------------------------------------------------------------------------

@app.post("/evaluate", tags=["Evals"])
async def evaluate(request: EvalRequest):
    """Evaluate a query-answer pair. Provide context_texts from a prior /query call."""
    from core.models import GenerationResult, RetrievedChunk, Chunk

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

    try:
        eval_result = get_evaluator().evaluate(gen_result)
        return eval_result.to_dict()
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
async def metrics():
    """Pipeline health check + aggregate statistics."""
    from core.evals import load_evals
    from core.telemetry import load_traces

    traces = load_traces()
    evals  = load_evals()

    llm_spans = [
        t for t in traces
        if t["name"] == "generator.llm_call" and "llm.latency_ms" in t.get("attributes", {})
    ]
    avg_latency = (
        round(sum(t["attributes"]["llm.latency_ms"] for t in llm_spans) / len(llm_spans), 1)
        if llm_spans else None
    )
    avg_score = (
        round(sum(e["overall_score"] for e in evals) / len(evals), 3)
        if evals else None
    )

    return MetricsResponse(
        status="ok",
        collection_size=get_pipeline().store.count(),
        sources=get_pipeline().store.list_sources(),
        total_traces=len(traces),
        total_evals=len(evals),
        avg_latency_ms=avg_latency,
        avg_overall_score=avg_score,
    )


@app.get("/traces", tags=["Observability"])
async def traces(limit: int = Query(default=50, ge=1, le=500)):
    """Return recent OTel span records from traces.jsonl."""
    from core.telemetry import load_traces
    return {"traces": load_traces(limit=limit), "count": limit}


@app.get("/evals", tags=["Observability"])
async def evals(limit: int = Query(default=50, ge=1, le=500)):
    """Return recent eval results from evals.jsonl."""
    from core.evals import load_evals
    records = load_evals(limit=limit)
    return {"evals": records, "count": len(records)}


@app.get("/sources", tags=["Observability"])
async def sources():
    """List all source documents currently in the vector store."""
    return {"sources": get_pipeline().store.list_sources()}


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove all chunks belonging to a document by doc_id."""
    try:
        deleted = get_pipeline().store.delete_document(doc_id)
        return {"status": "ok", "deleted_chunks": deleted, "doc_id": doc_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "collection_size": get_pipeline().store.count(),
        "timestamp": time.time(),
    }