import json
import re

import pytest

from core.telemetry import load_traces, record_span


def test_span_jsonl_and_history(logs):
    assert load_traces() == []
    with record_span("operation", {"source": "local"}) as span:
        span["items"] = 3
    with record_span("second"):
        pass
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    first, _ = records
    assert first["name"] == "operation" and first["status"] == "OK"
    assert first["attributes"]["source"] == "local"
    assert first["attributes"]["items"] == 3
    assert first["duration_ms"] >= 0 and first["attributes"]["duration_ms"] >= 0
    for key, width in (("trace_id", 32), ("span_id", 16)):
        assert re.fullmatch(f"[0-9a-f]{{{width}}}", first[key])
        assert int(first[key], 16) > 0
    assert load_traces(limit=1) == records[-1:]


def test_span_records_and_propagates_error():
    with pytest.raises(ValueError, match="broken"), record_span("failure"):
        raise ValueError("broken")
    assert load_traces()[0]["status"] == "ERROR"
