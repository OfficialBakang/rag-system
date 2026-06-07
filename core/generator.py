"""
Generator
=========
Takes a user query + retrieved chunks and calls Claude to produce a
grounded, cited answer.

Prompt design:
  - System: strict grounding instruction — answer from context only
  - Context block: numbered chunks with source attribution
  - User: original question

The generator returns a GenerationResult with full provenance:
  - The answer text
  - Which chunks were used
  - Token counts (for cost tracking + telemetry)
  - Wall-clock latency

Usage:
    generator = Generator()
    result = generator.generate(query, retrieved_chunks)
    print(result.answer)
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import anthropic

from core.models import GenerationResult, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant powered by retrieval-augmented generation.

Rules you must follow:
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
        passages.append(
            _PASSAGE_TEMPLATE.format(rank=rc.rank, source=source, text=rc.chunk.text)
        )
    return _CONTEXT_TEMPLATE.format(
        passages="\n\n".join(passages),
        query=query,
    )


class Generator:
    """
    Grounded answer generator using Claude.

    Parameters
    ----------
    model      : Claude model string (default from env: LLM_MODEL)
    max_tokens : max answer length in tokens (default 1024)
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
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
        """
        Generate a grounded answer.

        Parameters
        ----------
        query            : original user question
        retrieved_chunks : output of Retriever.retrieve()
        trace_id         : optional OTel trace ID to attach to the result

        Returns a GenerationResult with the answer + full provenance.
        """
        if not retrieved_chunks:
            logger.warning("Generator called with no retrieved chunks")

        trace_id = trace_id or str(uuid.uuid4())
        user_message = _build_context_block(query, retrieved_chunks)

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.content[0].text
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        logger.info(
            "Generated answer — %d prompt tokens, %d completion tokens, %.0f ms",
            prompt_tokens,
            completion_tokens,
            latency_ms,
        )

        return GenerationResult(
            answer=answer,
            query=query,
            retrieved_chunks=retrieved_chunks,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            trace_id=trace_id,
        )

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ):
        """
        Streaming variant — yields text tokens as they arrive.
        Use this for the API endpoint to reduce perceived latency.
        """
        user_message = _build_context_block(query, retrieved_chunks)
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text