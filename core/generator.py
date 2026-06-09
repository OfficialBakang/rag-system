"""
Generator
=========
Grounded answer generation with full OTel instrumentation.
Every LLM call emits a span with token counts, latency, and model name.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import anthropic

from core.models import GenerationResult, RetrievedChunk
from core.telemetry import record_span

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant powered by retrieval-augmented generation.

Rules:
1. Answer ONLY using the provided context passages. Do not use outside knowledge.
2. If the answer is not in the context, say: "I don't have enough information to answer that."
3. Cite sources inline using [Source: <filename>] at the end of any sentence that draws on a passage.
4. Be concise. Prefer 2-4 sentences unless the question genuinely requires more detail.
5. Never fabricate facts, statistics, or quotes.
"""

_CONTEXT_TEMPLATE = """\
<context>
{passages}
</context>

Question: {query}
"""

_PASSAGE_TEMPLATE = "[{rank}] (Source: {source})\n{text}"


def _build_context_block(query: str, chunks: list[RetrievedChunk]) -> str:
    passages = []
    for rc in chunks:
        source = rc.chunk.metadata.get("filename", rc.chunk.metadata.get("source", "unknown"))
        passages.append(_PASSAGE_TEMPLATE.format(rank=rc.rank, source=source, text=rc.chunk.text))
    return _CONTEXT_TEMPLATE.format(passages="\n\n".join(passages), query=query)


class Generator:
    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", 1024))
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        trace_id: str | None = None,
    ) -> GenerationResult:

        trace_id = trace_id or str(uuid.uuid4())

        with record_span("generator.generate", {
            "query": query,
            "context.chunks": len(retrieved_chunks),
            "model": self.model,
            "trace_id": trace_id,
        }) as gen_span:

            # Build prompt
            with record_span("generator.build_prompt", {
                "chunks": len(retrieved_chunks)
            }) as prompt_span:
                user_message = _build_context_block(query, retrieved_chunks)
                prompt_span["prompt.chars"] = len(user_message)

            # LLM call
            with record_span("generator.llm_call", {
                "model": self.model,
                "max_tokens": self.max_tokens,
            }) as llm_span:
                t0 = time.perf_counter()
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)

                answer = response.content[0].text
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens

                llm_span["llm.prompt_tokens"] = prompt_tokens
                llm_span["llm.completion_tokens"] = completion_tokens
                llm_span["llm.total_tokens"] = prompt_tokens + completion_tokens
                llm_span["llm.latency_ms"] = latency_ms
                llm_span["llm.answer_chars"] = len(answer)

            gen_span["result.latency_ms"] = latency_ms
            gen_span["result.total_tokens"] = prompt_tokens + completion_tokens

            logger.info(
                "Generated answer — %d prompt + %d completion tokens, %.0f ms",
                prompt_tokens, completion_tokens, latency_ms,
            )

            return GenerationResult(
                answer=answer,
                query=query,
                retrieved_chunks=retrieved_chunks,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )

    def generate_stream(self, query: str, retrieved_chunks: list[RetrievedChunk]):
        """Streaming variant — yields text tokens as they arrive."""
        user_message = _build_context_block(query, retrieved_chunks)
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text