"""
Document Loader
===============
Loads raw files from disk into Document objects.
Supports: PDF, Markdown, plain text, HTML.

Each loader extracts text + metadata. The metadata dict always contains:
  - source      : original file path
  - format      : file type string
  - char_count  : length of extracted text
  - page_count  : (PDF only)
  - title       : inferred from filename or <title> tag

Usage:
    loader = DocumentLoader()
    docs = loader.load_directory("./data/docs")
    doc  = loader.load_file("./data/docs/report.pdf")
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable

from core.models import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, {}


def _load_markdown(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip front-matter (---...---) if present
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        try:
            end = lines.index("---", 1)
            raw = "\n".join(lines[end + 1:])
        except ValueError:
            pass
    return raw, {}


def _load_pdf(path: Path) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n\n".join(pages)
    meta = {
        "page_count": len(reader.pages),
        "title": reader.metadata.get("/Title", path.stem) if reader.metadata else path.stem,
    }
    return full_text, meta


def _load_html(path: Path) -> tuple[str, dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else path.stem
    text = soup.get_text(separator="\n", strip=True)
    return text, {"title": title}


# ---------------------------------------------------------------------------
# Loader dispatch table
# ---------------------------------------------------------------------------

_LOADERS: dict[str, Callable[[Path], tuple[str, dict]]] = {
    ".txt":  _load_text,
    ".md":   _load_markdown,
    ".mdx":  _load_markdown,
    ".pdf":  _load_pdf,
    ".html": _load_html,
    ".htm":  _load_html,
}


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Loads files into Document objects.

    Deterministic doc_id: sha256 of the file's absolute path, so the same
    file always maps to the same ID even across restarts.
    """

    def __init__(self, supported_formats: list[str] | None = None):
        self._formats = supported_formats or list(_LOADERS.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> Document:
        """Load a single file. Raises ValueError for unsupported formats."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in _LOADERS:
            raise ValueError(
                f"Unsupported format '{suffix}'. Supported: {list(_LOADERS.keys())}"
            )

        t0 = time.perf_counter()
        loader_fn = _LOADERS[suffix]
        text, extra_meta = loader_fn(path)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        doc_id = self._make_id(path)
        metadata = {
            "source": str(path),
            "filename": path.name,
            "format": suffix.lstrip("."),
            "char_count": len(text),
            "load_ms": round(elapsed_ms, 2),
            **extra_meta,
        }
        metadata.setdefault("title", path.stem)

        doc = Document(
            doc_id=doc_id,
            content=text,
            source=str(path),
            format=suffix.lstrip("."),
            metadata=metadata,
        )
        logger.info(
            "Loaded %s (%d chars, %.1f ms)", path.name, len(text), elapsed_ms
        )
        return doc

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """
        Load all supported files in a directory.
        Returns documents in sorted order (reproducible).
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        paths = sorted(
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in self._formats
        )

        docs: list[Document] = []
        for path in paths:
            try:
                docs.append(self.load_file(path))
            except Exception as exc:
                logger.warning("Skipping %s — %s", path.name, exc)

        logger.info(
            "Loaded %d/%d files from %s", len(docs), len(paths), directory
        )
        return docs

    def load_text(self, text: str, source: str = "inline") -> Document:
        """
        Create a Document directly from a string.
        Useful for testing and API ingestion endpoints.
        """
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return Document(
            doc_id=doc_id,
            content=text,
            source=source,
            format="text",
            metadata={
                "source": source,
                "filename": source,
                "format": "text",
                "char_count": len(text),
                "title": source,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(path: Path) -> str:
        """Deterministic ID: sha256 of absolute path, first 16 hex chars."""
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]