from dataclasses import replace

import pytest

from core.embedder import Embedder


@pytest.mark.parametrize("size", [1, 3])
def test_ordered_batch_requests(size, chunk, mocker, openai_response):
    create = mocker.patch("core.embedder.OpenAI").return_value.embeddings.create
    chunks = [replace(chunk, text=str(i)) for i in range(size)]
    create.side_effect = lambda **kw: openai_response(
        *[[float(t)] for t in kw["input"]]
    )
    result = Embedder("model", batch_size=2).embed(chunks)
    assert [r.chunk for r in result] == chunks
    assert [r.embedding for r in result] == [[float(i)] for i in range(size)]
    assert all(r.embedding_model == "model" for r in result)
    assert create.call_args_list == [
        mocker.call(
            model="model",
            input=[c.text for c in chunks[i : i + 2]],
            encoding_format="float",
        )
        for i in range(0, size, 2)
    ]


def test_empty_and_query_retry(mocker, openai_response):
    create = mocker.patch("core.embedder.OpenAI").return_value.embeddings.create
    sleep = mocker.patch("core.embedder.time.sleep")
    service = Embedder("model")
    assert service.embed([]) == []
    create.assert_not_called()
    create.side_effect = [TimeoutError(), openai_response([0.5])]
    assert service.embed_query("question") == [0.5]
    assert (
        create.call_args_list
        == [mocker.call(model="model", input=["question"], encoding_format="float")] * 2
    )
    sleep.assert_called_once()


def test_exhausted_retries_preserve_cause(mocker):
    create = mocker.patch("core.embedder.OpenAI").return_value.embeddings.create
    mocker.patch("core.embedder.time.sleep")
    error = TimeoutError("offline")
    create.side_effect = error
    with pytest.raises(RuntimeError, match="3 attempts") as exc:
        Embedder().embed_query("why")
    assert exc.value.__cause__ is error
    assert create.call_count == 3


def test_batch_budget_includes_retries(chunk, mocker, monkeypatch):
    from rag_config import ResourceLimit

    monkeypatch.setenv("RAG_LIMIT_EMBEDDING_BATCHES", "1")
    mocker.patch("core.embedder.time.sleep")
    calls = mocker.patch("core.embedder.OpenAI").return_value.embeddings.create
    calls.side_effect = TimeoutError("private")
    with pytest.raises(ResourceLimit):
        Embedder().embed([chunk])
    assert calls.call_count == 1
