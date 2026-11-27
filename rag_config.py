"""Shared startup configuration for the API and standalone pipeline modules."""

import os

from dotenv import load_dotenv


def smoke_test_mode() -> bool:
    value = os.getenv("RAG_SMOKE_TEST_MODE", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("RAG_SMOKE_TEST_MODE must be true or false")
    return value == "true"


def load_settings() -> None:
    if not smoke_test_mode():
        load_dotenv()
