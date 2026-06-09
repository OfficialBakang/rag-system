"""
Phase 2 Telemetry Smoke Test
=============================
Runs a full query through the instrumented pipeline and prints:
  - The answer
  - Every span that was recorded with its duration
  - The full trace log entry

Usage:
    python test_telemetry.py
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise so output is clean
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
if missing:
    print(f"\n❌  Missing env vars: {', '.join(missing)}\n")
    sys.exit(1)

from core.embedder import Embedder
from core.generator import Generator
from core.ingestion_pipeline import IngestionPipeline
from core.retriever import Retriever
from core.telemetry import clear_traces, load_traces, record_span

# ---------------------------------------------------------------------------
# Setup — ingest sample docs if collection is empty
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RAG System — Phase 2 Telemetry Smoke Test")
print("=" * 60)

pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=40)

if pipeline.store.count() == 0:
    print("\n▶  Collection empty — ingesting sample docs...")
    DOCS = [
        ("RAG combines a dense retrieval system with a large language model. "
         "The retriever fetches relevant passages; the generator conditions its output "
         "on those passages. RAG reduces hallucination compared to pure parametric models.",
         "rag_overview.txt"),
        ("ChromaDB is an open-source vector database for AI applications. "
         "It supports cosine similarity and persists to disk using SQLite and HNSW.",
         "chromadb_docs.txt"),
        ("OpenTelemetry provides APIs and SDKs for traces, metrics, and logs. "
         "In a RAG pipeline, OTel spans wrap each stage capturing latency and token counts.",
         "otel_guide.txt"),
        ("RAGAS evaluates RAG pipelines with metrics: Faithfulness, Answer Relevance, "
         "Context Precision, and Context Recall. Scores range from 0 to 1.",
         "ragas_overview.txt"),
    ]
    for text, source in DOCS:
        pipeline.ingest_text(text, source=source)
    print(f"   Collection size: {pipeline.store.count()} chunks")

# ---------------------------------------------------------------------------
# Run a traced query
# ---------------------------------------------------------------------------
clear_traces()

query = "How does RAG reduce hallucinations?"
print(f"\n▶  Query: {query!r}\n")

embedder = Embedder()
retriever = Retriever(pipeline.store, embedder, rewrite_queries=True, top_k=4)
generator = Generator()

with record_span("test.full_query", {"query": query}) as root:
    chunks = retriever.retrieve(query)
    result = generator.generate(query, chunks)
    root["answer.chars"] = len(result.answer)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("─" * 60)
print("ANSWER:")
print(result.answer)
print("─" * 60)

print(f"\nTokens   : {result.prompt_tokens} prompt + {result.completion_tokens} completion")
print(f"Latency  : {result.latency_ms:.0f} ms")
print(f"Sources  : {', '.join(result.sources)}")
print(f"Trace ID : {result.trace_id}")

# ---------------------------------------------------------------------------
# Print span summary
# ---------------------------------------------------------------------------
traces = load_traces()
if traces:
    print(f"\n{'─'*60}")
    print(f"{'SPAN':<40} {'STATUS':<8} {'DURATION':>10}")
    print(f"{'─'*40} {'─'*8} {'─'*10}")
    for t in traces:
        name = t["name"]
        status = t["status"]
        dur = f"{t['duration_ms']:.1f} ms"
        print(f"{name:<40} {status:<8} {dur:>10}")
    print(f"\n✅  {len(traces)} spans recorded → data/traces.jsonl")
else:
    print("\n⚠️  No traces found in log.")