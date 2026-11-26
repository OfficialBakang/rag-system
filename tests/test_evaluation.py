import json
from dataclasses import replace

from core.evals import EvalRunner, load_evals


def test_batch_judging_weights_and_jsonl(
    messages, anthropic_response, generation, logs
):
    replies = [
        '{"score": 0.2}',
        '```json\n{"score": 0.4}\n```',
        '{"score": 1.4}',
        '{"score": -0.2}',
        "not-json",
        "{}",
        TimeoutError(),
        '{"score": 0.8}',
    ]
    messages.create.side_effect = [
        r if isinstance(r, Exception) else anthropic_response(r) for r in replies
    ]
    results = EvalRunner("judge").evaluate_batch(
        [generation, replace(generation, query="second")]
    )
    assert messages.create.call_count == 8
    for call, metric in zip(
        messages.create.call_args_list[:4],
        ["FAITHFULNESS", "ANSWER RELEVANCE", "CONTEXT PRECISION", "CONTEXT RECALL"],
        strict=True,
    ):
        assert call.kwargs["model"] == "judge"
        assert metric in call.kwargs["messages"][0]["content"]
    assert [r.query for r in results] == ["why", "second"]
    first, second = results
    assert (
        first.faithfulness,
        first.answer_relevance,
        first.context_precision,
        first.context_recall,
        first.overall_score,
    ) == (0.2, 0.4, 1.0, 0.0, 0.39)
    assert (
        second.faithfulness,
        second.answer_relevance,
        second.context_precision,
        second.context_recall,
        second.overall_score,
    ) == (0, 0, 0, 0.8, 0.12)
    rows = [json.loads(line) for line in logs[1].read_text().splitlines()]
    assert rows == [r.to_dict() for r in results]
    assert rows[0] == {
        "trace_id": "trace",
        "query": "why",
        "answer": "answer",
        "faithfulness": 0.2,
        "answer_relevance": 0.4,
        "context_precision": 1,
        "context_recall": 0,
        "overall_score": 0.39,
        "model": "model",
        "eval_model": "judge",
        "timestamp": first.timestamp,
        "latency_ms": first.latency_ms,
    }
    assert load_evals(limit=1) == rows[-1:]
