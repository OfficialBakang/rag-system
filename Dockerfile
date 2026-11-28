
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip check


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="RAG System" \
      org.opencontainers.image.description="Retrieval-Augmented Generation API" \
      org.opencontainers.image.source="https://github.com/OfficialBakang/rag-system"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tokenizer-cache \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    TRACE_LOG_PATH=/app/data/traces.jsonl \
    EVAL_LOG_PATH=/app/data/evals.jsonl \
    OTEL_ENABLED=false

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 \
       --create-home appuser \
    && mkdir -p /app/data/chroma /opt/tokenizer-cache

COPY --from=builder /opt/venv /opt/venv

RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser api.py rag_config.py api_security.py ./

RUN chown -R appuser:appuser /app/data /opt/tokenizer-cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--no-access-log", "--no-proxy-headers"]
