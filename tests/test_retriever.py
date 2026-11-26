from dataclasses import replace

import pytest

from core.embedder import Embedder
from core.retriever import Retriever
from core.vector_store import VectorStore


@pytest.mark.parametrize(
    ("rewrite", "reply", "expected"),
    [
        (True, ' "rewritten" ', "rewritten"),
        (True, "   ", "original"),
        (True, TimeoutError(), "original"),
        (False, None, "original"),
    ],
)
def test_retrieval_forwards_query_and_filter(
    rewrite, reply, expected, messages, anthropic_response, mocker
):
    store, embedder = mocker.Mock(spec=VectorStore), mocker.Mock(spec=Embedder)
    store.query.return_value = []
    embedder.embed_query.return_value = [0.1, 0.9]
    if isinstance(reply, Exception):
        messages.create.side_effect = reply
    else:
        messages.create.return_value = anthropic_response(reply)
    service = Retriever(store, embedder, rewrite_queries=rewrite)
    assert service.retrieve("original", top_k=4, where={"source": "one"}) == []
    assert messages.create.call_count == int(rewrite)
    embedder.embed_query.assert_called_once_with(expected)
    store.query.assert_called_once_with([0.1, 0.9], top_k=4, where={"source": "one"})


def test_reranking_orders_and_limits_candidates(messages, retrieved, mocker):
    store, embedder = mocker.Mock(spec=VectorStore), mocker.Mock(spec=Embedder)
    candidates = [
        replace(retrieved, chunk=replace(retrieved.chunk, text=t)) for t in "abc"
    ]
    store.query.return_value = candidates
    embedder.embed_query.return_value = [1.0]
    service = Retriever(store, embedder, rewrite_queries=False, rerank=True, top_n=2)
    service._reranker = mocker.Mock()
    service._reranker.predict.return_value = [0.2, 0.8, 0.5]
    result = service.retrieve("why")
    service._reranker.predict.assert_called_once_with([["why", t] for t in "abc"])
    assert [(r.chunk.text, r.score, r.rank) for r in result] == [
        ("b", 0.8, 1),
        ("c", 0.5, 2),
    ]
