"""
Phase 3 Eval Smoke Test
========================
Runs 4 queries through the full pipeline and evaluates each one.
Prints a score report and saves results to data/evals.jsonl.

Usage:
    python test_evals.py
"""

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
if missing:
    print(f"\n❌  Missing env vars: {', '.join(missing)}\n")
    sys.exit(1)

from core.embedder import Embedder
from core.evals import EvalRunner, load_evals
from core.generator import Generator
from core.ingestion_pipeline import IngestionPipeline
from core.retriever import Retriever

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RAG System — Phase 3 Eval Smoke Test")
print("=" * 60)

pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=40)

if pipeline.store.count() == 0:
    print("\n▶  Ingesting sample docs...")
    DOCS = [
        ("Retrieval-Augmented Generation (RAG) combines a dense retrieval "
         "system with a large language model. The retriever fetches relevant "
         "passages from a knowledge base; the generator conditions its output "
         "on those passages. RAG was introduced by Lewis et al. (2020) and "
         "significantly reduces hallucination compared to pure parametric models.",
         "rag_overview.txt"),
        ("ChromaDB is an open-source vector database designed for AI applications. "
         "It supports cosine, L2, and inner-product similarity metrics. ChromaDB "
         "stores embeddings, raw documents, and arbitrary metadata. It persists "
         "to disk using SQLite and an HNSW index.",
         "chromadb_docs.txt"),
        ("OpenTelemetry (OTel) is a vendor-neutral observability framework providing "
         "APIs, SDKs, and tooling for traces, metrics, and logs. In a RAG pipeline, "
         "OTel spans can wrap each stage capturing latency and token counts.",
         "otel_guide.txt"),
        ("RAGAS evaluates RAG pipelines with four key metrics: Faithfulness measures "
         "whether all claims in the answer are supported by the context. Answer "
         "Relevance measures whether the answer addresses the question. Context "
         "Precision measures whether retrieved chunks are relevant. Context Recall "
         "measures whether all needed information was retrieved. All scores are 0-1.",
         "ragas_overview.txt"),
    ]
    for text, source in DOCS:
        pipeline.ingest_text(text, source=source)
    print(f"   Collection size: {pipeline.store.count()} chunks\n")

embedder  = Embedder()
retriever = Retriever(pipeline.store, embedder, rewrite_queries=True, top_k=4)
generator = Generator()
evaluator = EvalRunner()

# ---------------------------------------------------------------------------
# Test queries — mix of good, broad, and out-of-scope questions
# ---------------------------------------------------------------------------
QUERIES = [
    "What is RAG and how does it reduce hallucinations?",
    "What metrics does RAGAS use to evaluate RAG pipelines?",
    "How does ChromaDB store and search embeddings?",
    "What is OpenTelemetry used for in a RAG system?",
]

print("▶  Running queries and evaluating...\n")
generation_results = []

for i, query in enumerate(QUERIES, 1):
    print(f"  [{i}/{len(QUERIES)}] {query}")
    chunks = retriever.retrieve(query)
    result = generator.generate(query, chunks)
    generation_results.append(result)

# ---------------------------------------------------------------------------
# Evaluate all results
# ---------------------------------------------------------------------------
print("\n▶  Scoring with LLM judge...\n")
eval_results = evaluator.evaluate_batch(generation_results)

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
evaluator.print_report(eval_results)

# Show one full answer as example
print("SAMPLE ANSWER:")
print(f"Q: {eval_results[0].query}")
print(f"A: {eval_results[0].answer}")
print(f"\nScores → Faithfulness: {eval_results[0].faithfulness:.2f} | "
      f"Relevance: {eval_results[0].answer_relevance:.2f} | "
      f"Precision: {eval_results[0].context_precision:.2f} | "
      f"Recall: {eval_results[0].context_recall:.2f}")
print(f"Overall: {eval_results[0].overall_score:.3f}")

saved = load_evals()
print(f"\n✅  {len(eval_results)} eval results saved → data/evals.jsonl")
print(f"   Total evals on disk: {len(saved)}")