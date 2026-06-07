"""
Chunker
=======
Splits Document objects into overlapping Chunk objects ready for embedding.

Strategy (in priority order):
  1. Split on paragraph breaks (\n\n) first — keeps logical sections together.
  2. If a paragraph exceeds chunk_size, split on sentence boundaries.
  3. As a last resort, split on whitespace.

Overlap is implemented by re-including the last `overlap` tokens of the
previous chunk at the start of the next one — this prevents meaning loss
at chunk boundaries.

Usage:
    chunker = Chunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk(doc)
    all_chunks = chunker.chunk_many(docs)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace

import tiktoken

from core.models import Chunk, Document

logger = logging.getLogger(__name__)

# Tokeniser for counting — cl100k_base covers GPT-4 and Claude tokenisation
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _split_sentences(text: str) -> list[str]:
    """Naive but fast sentence splitter. Good enough for RAG chunking."""
    # Split after . ! ? followed by space + capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Deterministic chunk ID — stable across re-ingestion."""
    payload = f"{doc_id}:{index}:{text[:64]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Chunker:
    """
    Sentence-aware, token-counted chunker with configurable overlap.

    Parameters
    ----------
    chunk_size    : target chunk size in tokens (default 400)
    chunk_overlap : tokens to repeat from previous chunk (default 80)
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk a single Document. Returns a list of Chunk objects."""
        raw_segments = self._split_into_segments(doc.content)
        token_windows = self._build_windows(raw_segments)
        chunks = []

        for i, text in enumerate(token_windows):
            chunk_id = _make_chunk_id(doc.doc_id, i, text)
            token_count = _count_tokens(text)
            meta = {
                **doc.metadata,
                "chunk_index": i,
                "chunk_total": len(token_windows),  # filled in after loop
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=text,
                    index=i,
                    token_count=token_count,
                    metadata=meta,
                )
            )

        # Back-fill the final total count now that we know it
        total = len(chunks)
        for c in chunks:
            c.metadata["chunk_total"] = total

        logger.debug(
            "Chunked %r → %d chunks (avg %.0f tokens)",
            doc.doc_id,
            total,
            sum(c.token_count for c in chunks) / max(total, 1),
        )
        return chunks

    def chunk_many(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a list of Documents. Returns a flat list of Chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        logger.info(
            "Chunked %d documents → %d total chunks", len(docs), len(all_chunks)
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal: paragraph / sentence splitting
    # ------------------------------------------------------------------

    def _split_into_segments(self, text: str) -> list[str]:
        """
        First split on blank lines (paragraphs).
        If any paragraph exceeds chunk_size, further split on sentences.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        segments: list[str] = []

        for para in paragraphs:
            if _count_tokens(para) <= self.chunk_size:
                segments.append(para)
            else:
                # Too long — break into sentences
                sentences = _split_sentences(para)
                segments.extend(sentences)

        return segments

    # ------------------------------------------------------------------
    # Internal: sliding window assembly
    # ------------------------------------------------------------------

    def _build_windows(self, segments: list[str]) -> list[str]:
        """
        Pack segments into token windows of size chunk_size.
        Each window starts with the overlap tail of the previous one.
        """
        windows: list[str] = []
        current_tokens: list[str] = []   # raw words / tokens in current window
        current_count = 0

        # We work with raw words for simplicity; token count stays accurate
        # because we re-count at the end.
        overlap_buffer: str = ""

        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_tokens = _count_tokens(seg)

            # If adding this segment would exceed the limit, flush
            if current_count + seg_tokens > self.chunk_size and current_tokens:
                window_text = " ".join(current_tokens)
                windows.append(window_text)

                # Build overlap: take the tail of the flushed window
                overlap_buffer = self._tail_tokens(window_text, self.chunk_overlap)
                current_tokens = [overlap_buffer] if overlap_buffer else []
                current_count = _count_tokens(overlap_buffer) if overlap_buffer else 0
                # Don't advance i — re-process this segment
                continue

            current_tokens.append(seg)
            current_count += seg_tokens
            i += 1

        # Flush the last window
        if current_tokens:
            windows.append(" ".join(current_tokens))

        return windows

    @staticmethod
    def _tail_tokens(text: str, n_tokens: int) -> str:
        """Return the last n_tokens worth of text from a string."""
        words = text.split()
        result_words: list[str] = []
        count = 0
        for word in reversed(words):
            word_tokens = _count_tokens(word)
            if count + word_tokens > n_tokens:
                break
            result_words.append(word)
            count += word_tokens
        return " ".join(reversed(result_words))