# RAG System with Telemetry & Evals

A production-grade Retrieval-Augmented Generation pipeline built for a portfolio.  
Demonstrates: vector search, LLM grounding, OpenTelemetry instrumentation, and automated evaluation.

---

## Project structure

```
rag-system/
├── core/
│   ├── models.py            # Shared dataclasses (Document, Chunk, RetrievedChunk, …)
│   ├── loader.py            # Document loader — PDF, Markdown, text, HTML
│   ├── chunker.py           # Sentence-aware, overlapping token chunker
│   ├── embedder.py          # OpenAI batch embedder (text-embedding-3-small)
│   ├── vector_store.py      # ChromaDB wrapper — upsert, query, delete
│   ├── retriever.py         # Two-stage retrieval: query rewriting + vector search
│   ├── generator.py         # Grounded answer generation with Claude
│   └── ingestion_pipeline.py # Orchestrates loader → chunker → embedder → store
│
├── data/
│   ├── docs/                # Drop your source documents here
│   └── chroma/              # ChromaDB persists here (auto-created)
│
├── tests/                   # Unit tests (Phase 4)
│
├── .env.example             # Copy to .env and fill in API keys
├── requirements.txt
├── test_pipeline.py         # Phase 1 smoke test
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd rag-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 3. Run the smoke test

```bash
python test_pipeline.py
```

Expected output:
```
▶  Ingesting sample documents...
   Collection size: N chunks

▶  Query: 'What is RAG and how does it reduce hallucinations?'
   Retrieved 4 chunks:
   [1] score=0.847  source=rag_overview.txt
   ...

ANSWER:
RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM...
```

---

## Architecture

```
INGESTION                          QUERY
─────────────────────────────      ─────────────────────────────────────────
Documents → Loader → Chunker       User Query → Query Rewriter → Embedder
              ↓                                    ↓
           Embedder               Vector Store ← similarity search
              ↓                                    ↓
         Vector Store             Reranker (optional cross-encoder)
                                            ↓
                                   Prompt Builder → Claude LLM → Answer
```

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | ChromaDB | Zero infra, persistent, swappable |
| Embedding model | text-embedding-3-small | Best price/perf ratio |
| Chunk size | 400 tokens | Balances context vs retrieval precision |
| Overlap | 80 tokens (20%) | Prevents meaning loss at boundaries |
| Query rewriting | Claude (lightweight call) | ~10-15% recall improvement |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Optional; improves precision |
| LLM | Claude Sonnet 4 | Strong grounding, citation-following |

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Core pipeline | ✅ Done | Load → chunk → embed → store → retrieve → generate |
| 2 — Telemetry | 🔜 Next | OTel spans on every stage |
| 3 — Evals | 🔜 | RAGAS faithfulness, relevance, precision |
| 4 — API | 🔜 | FastAPI endpoints |
| 5 — Dashboard | 🔜 | React dashboard with live metrics |

---

## Tech stack

- **Python 3.11+**
- **ChromaDB** — vector store
- **OpenAI** — embeddings (text-embedding-3-small)
- **Anthropic Claude** — generation + query rewriting
- **tiktoken** — accurate token counting
- **FastAPI** — API layer (Phase 4)
- **OpenTelemetry** — distributed tracing (Phase 2)
- **RAGAS** — retrieval evaluation (Phase 3)