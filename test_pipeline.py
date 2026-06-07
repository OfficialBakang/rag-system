"""
Phase 1 smoke test — run this after setting up your .env file.

Usage:
    python test_pipeline.py

What it does:
    1. Creates a handful of in-memory documents (no files needed)
    2. Runs the full ingestion pipeline (chunk → embed → store)
    3. Fires a retrieval query
    4. Generates a grounded answer
    5. Prints a full summary

Requires: ANTHROPIC_API_KEY and OPENAI_API_KEY in .env
"""

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Guard: check API keys before trying anything
# ---------------------------------------------------------------------------
missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
if missing:
    print(f"\n❌  Missing env vars: {', '.join(missing)}")
    print("    Copy .env.example → .env and fill in your keys.\n")
    sys.exit(1)

from core import Generator, IngestionPipeline, Retriever, VectorStore
from core.embedder import Embedder

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    {
        "text": """\
Retrieval-Augmented Generation (RAG) is a technique that combines a
dense retrieval system with a large language model. The retriever fetches
relevant passages from a knowledge base; the generator conditions its
output on those passages. RAG was introduced by Lewis et al. (2020) and
significantly reduces hallucination compared to pure parametric models.
        """,
        "source": "rag_overview.txt",
    },
    {
        "text": """\
ChromaDB is an open-source vector database designed for AI applications.
It supports cosine, L2, and inner-product similarity metrics. ChromaDB
stores embeddings, raw documents, and arbitrary metadata in a single
collection. It can run in-memory or persist to disk using SQLite and an
HNSW index. The Python client is pip-installable with no external services.
        """,
        "source": "chromadb_docs.txt",
    },
    {
        "text": """\
OpenTelemetry (OTel) is a vendor-neutral observability framework providing
APIs, SDKs, and tooling for traces, metrics, and logs. In a RAG pipeline,
OTel spans can wrap each stage — loading, chunking, embedding, retrieval,
and generation — capturing latency, token counts, and error rates in a
single distributed trace visible in tools like Jaeger or Grafana Tempo.
        """,
        "source": "otel_guide.txt",
    },
    {
        "text": """\
RAGAS (Retrieval Augmented Generation Assessment) is an evaluation
framework for RAG pipelines. Key metrics include:
  - Faithfulness: are all claims in the answer supported by the context?
  - Answer Relevance: does the answer address the question?
  - Context Precision: are the retrieved chunks actually needed?
  - Context Recall: are all relevant chunks retrieved?
RAGAS uses an LLM-as-judge approach and returns scores between 0 and 1.
        """,
        "source": "ragas_overview.txt",
    },
]

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RAG System — Phase 1 Smoke Test")
print("=" * 60 + "\n")

# 1. Ingest
print("▶  Ingesting sample documents...")
pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=40)
for d in SAMPLE_DOCS:
    pipeline.ingest_text(d["text"], source=d["source"])
print(f"   Collection size: {pipeline.store.count()} chunks\n")

# 2. Retrieve
query = "What is RAG and how does it reduce hallucinations?"
print(f"▶  Query: {query!r}")

embedder = Embedder()
retriever = Retriever(
    vector_store=pipeline.store,
    embedder=embedder,
    rewrite_queries=True,
    rerank=False,
    top_k=4,
)
chunks = retriever.retrieve(query)
print(f"   Retrieved {len(chunks)} chunks:")
for rc in chunks:
    source = rc.chunk.metadata.get("source", "?")
    print(f"   [{rc.rank}] score={rc.score:.3f}  source={source}")
print()

# 3. Generate
print("▶  Generating answer...")
generator = Generator()
result = generator.generate(query, chunks)

print("\n" + "─" * 60)
print("ANSWER:")
print(result.answer)
print("─" * 60)
print(f"\nSources   : {', '.join(result.sources)}")
print(f"Tokens    : {result.prompt_tokens} prompt + {result.completion_tokens} completion")
print(f"Latency   : {result.latency_ms:.0f} ms")
print(f"Trace ID  : {result.trace_id}\n")