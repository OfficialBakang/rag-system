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
from collections.abc import Iterator

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
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
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
        current = ""
        for segment in segments:
            for separator, piece in self._bounded_pieces(segment):
                candidate = f"{current}{separator}{piece}" if current else piece
                if current and _count_tokens(candidate) > self.chunk_size:
                    windows.append(current)
                    tail = self._tail_tokens(current, self.chunk_overlap)
                    candidate = f"{tail}{separator}{piece}" if tail else piece
                    # Retain only overlap that leaves room for new content.
                    while tail and _count_tokens(candidate) > self.chunk_size:
                        tail = tail.partition(" ")[2]
                        candidate = f"{tail}{separator}{piece}" if tail else piece
                current = candidate
        if current:
            windows.append(current)
        return windows

    def _bounded_pieces(self, segment: str) -> Iterator[tuple[str, str]]:
        if _count_tokens(segment) <= self.chunk_size:
            yield " ", segment
            return

        # Pack whitespace-delimited words first. For long unbroken strings,
        # split on character boundaries so Unicode text remains intact.
        for word in segment.split():
            if _count_tokens(word) <= self.chunk_size:
                yield " ", word
                continue
            start = 0
            while start < len(word):
                end = min(start + self.chunk_size, len(word))
                while (token_count := _count_tokens(word[start:end])) > self.chunk_size:
                    if end == start + 1:
                        raise ValueError(
                            f"Character U+{ord(word[start]):04X} requires {token_count} tokens "
                            f"but chunk_size={self.chunk_size}"
                        )
                    end = start + (end - start) // 2
                yield " " if start == 0 else "", word[start:end]
                start = end

    @staticmethod
    def _tail_tokens(text: str, n_tokens: int) -> str:
        """Return the last n_tokens worth of text from a string."""
        words = text.split()
        result_words: list[str] = []
        for word in reversed(words):
            candidate = " ".join([word, *reversed(result_words)])
            if _count_tokens(candidate) > n_tokens:
                break
            result_words.append(word)
        return " ".join(reversed(result_words))
