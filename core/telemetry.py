"""
Telemetry
=========
OpenTelemetry setup and instrumentation for the RAG pipeline.

Every stage gets a span:
  - ingest        : document loading, chunking, embedding, upsert
  - retrieve      : query rewriting, vector search, reranking
  - generate      : prompt building, LLM call, token counts

Spans are exported to:
  - Console (always on, for local development)
  - OTLP endpoint (when OTEL_ENABLED=true, for Jaeger/Grafana)

Trace data is also written to data/traces.jsonl for the dashboard
to read without needing a running collector.

Usage:
    from core.telemetry import tracer, record_span

    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("key", "value")
        result = do_work()
        span.set_attribute("result.count", len(result))
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Status, StatusCode

from rag_config import load_settings, smoke_test_mode

load_settings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace log file — dashboard reads this
# ---------------------------------------------------------------------------
_TRACE_LOG = Path(os.getenv("TRACE_LOG_PATH", "./data/traces.jsonl"))
_TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

def _build_provider() -> TracerProvider:
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "rag-system"),
        "service.version": "1.0.0",
        "deployment.environment": "development",
    })

    provider = TracerProvider(resource=resource)

    # Always export to console in development
    provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    # Export to OTLP collector if configured
    if (not smoke_test_mode()
            and os.getenv("OTEL_ENABLED", "true").lower() == "true"):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            otlp = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                insecure=True,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp))
            logger.info("OTLP exporter configured → %s", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        except Exception:
            logger.exception("OTLP exporter unavailable — console only")

    trace.set_tracer_provider(provider)
    return provider


_provider = _build_provider()
tracer = trace.get_tracer("rag-system")


# ---------------------------------------------------------------------------
# JSONL trace logger (for the dashboard)
# ---------------------------------------------------------------------------

def _log_span(
    name: str,
    trace_id: str,
    span_id: str,
    parent_id: str | None,
    start_time: float,
    end_time: float,
    attributes: dict[str, Any],
    status: str = "OK",
) -> None:
    """Append a span record to the JSONL trace log."""
    record = {
        "name": name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": round((end_time - start_time) * 1000, 2),
        "attributes": attributes,
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(_TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Context manager — the main instrumentation primitive
# ---------------------------------------------------------------------------

@contextmanager
def record_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    parent_trace_id: str | None = None,
):
    """
    Context manager that creates an OTel span AND writes to traces.jsonl.

    Usage:
        with record_span("retriever.search", {"query": q, "top_k": k}) as span:
            results = vector_store.query(...)
            span["result.count"] = len(results)   # add attrs mid-span
    """
    attrs = dict(attributes or {})
    start = time.perf_counter()
    start_wall = time.time()
    status = "OK"

    with tracer.start_as_current_span(name) as otel_span:
        # Attach initial attributes to OTel span
        for k, v in attrs.items():
            otel_span.set_attribute(k, v)

        # Expose a simple dict-like interface for adding attributes mid-span
        class SpanProxy:
            def __setitem__(self, key: str, value: Any):
                otel_span.set_attribute(key, value)
                attrs[key] = value

            def set_error(self, exc: Exception):
                nonlocal status
                status = "ERROR"
                otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
                otel_span.record_exception(exc)

        proxy = SpanProxy()
        try:
            yield proxy
            otel_span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            proxy.set_error(exc)
            raise
        finally:
            end = time.perf_counter()
            end_wall = time.time()

            ctx = otel_span.get_span_context()
            trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else "unknown"
            span_id  = format(ctx.span_id,  "016x") if ctx.span_id  else "unknown"

            attrs["duration_ms"] = round((end - start) * 1000, 2)
            _log_span(
                name=name,
                trace_id=trace_id,
                span_id=span_id,
                parent_id=parent_trace_id,
                start_time=start_wall,
                end_time=end_wall,
                attributes=attrs,
                status=status,
            )


# ---------------------------------------------------------------------------
# Decorator — wrap a function in a span automatically
# ---------------------------------------------------------------------------

def traced(span_name: str | None = None, attributes: dict | None = None):
    """
    Decorator that wraps a function in a record_span call.

    Usage:
        @traced("embedder.embed")
        def embed(self, chunks):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        name = span_name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with record_span(name, attributes or {}):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Utility: read traces back for the dashboard / evals
# ---------------------------------------------------------------------------

def load_traces(limit: int = 200) -> list[dict]:
    """Read the most recent `limit` span records from the trace log."""
    if not _TRACE_LOG.exists():
        return []
    lines = _TRACE_LOG.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def clear_traces() -> None:
    """Wipe the trace log (useful between test runs)."""
    _TRACE_LOG.write_text("", encoding="utf-8")
    logger.info("Trace log cleared.")