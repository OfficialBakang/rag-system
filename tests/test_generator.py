from uuid import UUID

import pytest

from core.generator import Generator


@pytest.mark.parametrize("trace_id", [None, "trace"])
def test_generation_prompt_usage_and_provenance(
    trace_id, messages, anthropic_response, retrieved
):
    messages.create.return_value = anthropic_response("Answer")
    result = Generator("test-model", 51).generate(
        "why?", [retrieved], trace_id=trace_id
    )
    params = messages.create.call_args.kwargs
    assert (params["model"], params["max_tokens"]) == ("test-model", 51)
    prompt = params["messages"][0]["content"]
    assert "[1] (Source: report.txt)\nLocal passage" in prompt
    assert "Question: why?" in prompt
    assert (result.answer, result.query, result.model) == (
        "Answer",
        "why?",
        "test-model",
    )
    assert (result.prompt_tokens, result.completion_tokens) == (7, 2)
    assert result.retrieved_chunks == [retrieved]
    if trace_id:
        assert result.trace_id == trace_id
    else:
        assert UUID(result.trace_id).version == 4


def test_generation_error_propagates(messages):
    messages.create.side_effect = TimeoutError("judge unavailable")
    with pytest.raises(TimeoutError, match="unavailable"):
        Generator().generate("why?", [])


def test_stream(messages, retrieved):
    manager = messages.stream.return_value
    manager.__enter__.return_value.text_stream = iter(["first", " second"])
    assert list(Generator("stream-model", 12).generate_stream("why?", [retrieved])) == [
        "first",
        " second",
    ]
    assert messages.stream.call_args.kwargs["max_tokens"] == 12
    manager.__exit__.assert_called_once()
