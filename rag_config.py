"""Shared startup configuration for the API and standalone pipeline modules."""

import os
import re
from dataclasses import dataclass, fields

from dotenv import load_dotenv


class ResourceLimit(ValueError):
    """Document exceeds a configured processing budget."""


def flag(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def smoke_test_mode() -> bool:
    return flag("RAG_SMOKE_TEST_MODE", False)


def load_settings() -> None:
    if not smoke_test_mode():
        load_dotenv()


@dataclass(frozen=True)
class Limits:
    general: int = 60
    query: int = 10
    ingest: int = 5
    evaluate: int = 5
    anonymous: int = 120
    concurrent: int = 3
    query_chars: int = 2000
    text_chars: int = 100000
    top_k: int = 20
    upload_bytes: int = 10 * 1024**2
    body_bytes: int = 12 * 1024**2
    pdf_pages: int = 100
    extracted_chars: int = 200000
    chunks: int = 256
    embedding_batches: int = 3
    context_chars: int = 32000

    @classmethod
    def load(cls):
        values = {}
        for field in fields(cls):
            name = "RAG_LIMIT_" + field.name.upper()
            raw = os.getenv(name, str(field.default))
            if not re.fullmatch(r"[1-9][0-9]{0,8}", raw):
                raise ValueError(f"Invalid positive integer for {name}")
            values[field.name] = int(raw)
        result = cls(**values)
        if result.upload_bytes > result.body_bytes:
            raise ValueError("Upload limit exceeds body limit")
        return result
