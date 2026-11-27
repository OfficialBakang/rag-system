"""
Eval Runner
===========
LLM-as-judge evaluation of RAG pipeline outputs.

Scores each query-answer pair on four metrics:

  Faithfulness      — Every claim in the answer must be traceable to the
                      retrieved context. Hallucinated facts score 0.
                      Weight: 35% of overall score.

  Answer Relevance  — Does the answer directly address what was asked?
                      Off-topic or partial answers score lower.
                      Weight: 30%

  Context Precision — Of the chunks we retrieved, how many were actually
                      useful for answering the question?
                      Weight: 20%

  Context Recall    — Did the retrieved chunks contain all the information
                      needed? Missing key facts score lower.
                      Weight: 15%

Each metric uses a separate LLM prompt that returns a JSON score + reasoning.
Results are appended to data/evals.jsonl for the dashboard to read.

Usage:
    evaluator = EvalRunner()
    result = evaluator.evaluate(generation_result)
    print(result.overall_score)

    # Batch evaluate
    results = evaluator.evaluate_batch(generation_results)
    evaluator.print_report(results)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import anthropic

from core.models import EvalResult, GenerationResult
from core.telemetry import record_span
from rag_config import load_settings

load_settings()
logger = logging.getLogger(__name__)

_EVAL_LOG = Path(os.getenv("EVAL_LOG_PATH", "./data/evals.jsonl"))
_EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)

_EVAL_MODEL = os.getenv("EVAL_MODEL", "claude-sonnet-4-5")

# ---------------------------------------------------------------------------
# Eval prompts — each returns {"score": float, "reasoning": str}
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to its source context.

QUESTION: {query}

CONTEXT (retrieved passages):
{context}

ANSWER: {answer}

Task: Score the answer's FAITHFULNESS — the degree to which every claim in the
answer is directly supported by the context passages. Penalise any statement
that cannot be verified from the context, even if it seems plausible.

Scoring guide:
  1.0 — Every claim is explicitly supported by the context
  0.7 — Most claims supported; minor unsupported details
  0.4 — Some claims supported; notable unsupported statements
  0.0 — Answer contradicts context or is entirely unsupported

Respond with ONLY valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating whether an AI-generated answer addresses the user's question.

QUESTION: {query}

ANSWER: {answer}

Task: Score ANSWER RELEVANCE — how directly and completely the answer addresses
what was asked. Ignore factual accuracy; focus only on relevance.

Scoring guide:
  1.0 — Answer directly and completely addresses the question
  0.7 — Mostly relevant with minor tangents or gaps
  0.4 — Partially relevant; misses key aspects of the question
  0.0 — Irrelevant or refuses to answer without good reason

Respond with ONLY valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


_CONTEXT_PRECISION_PROMPT = """\
You are evaluating the quality of retrieved context for answering a question.

QUESTION: {query}

RETRIEVED CONTEXT:
{context}

Task: Score CONTEXT PRECISION — what fraction of the retrieved passages are
actually relevant and useful for answering the question. Penalise noise.

Scoring guide:
  1.0 — All retrieved passages are relevant to the question
  0.7 — Most passages relevant; 1-2 noisy chunks
  0.4 — About half the passages are relevant
  0.0 — Retrieved passages are entirely irrelevant

Respond with ONLY valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


_CONTEXT_RECALL_PROMPT = """\
You are evaluating whether retrieved context contains sufficient information
to answer a question.

QUESTION: {query}

ANSWER GIVEN: {answer}

RETRIEVED CONTEXT:
{context}

Task: Score CONTEXT RECALL — whether the retrieved passages contain all the
key information needed to produce a complete answer. If important facts
required for a full answer are missing from the context, score lower.

Scoring guide:
  1.0 — Context contains all information needed for a complete answer
  0.7 — Context mostly sufficient; minor details missing
  0.4 — Context partially sufficient; key information missing
  0.0 — Context lacks the information needed to answer the question

Respond with ONLY valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """
    Evaluates GenerationResult objects using an LLM-as-judge approach.

    Parameters
    ----------
    model : Claude model to use as judge (default: claude-sonnet-4-5)
    """

    def __init__(self, model: str = _EVAL_MODEL):
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, result: GenerationResult) -> EvalResult:
        """
        Run all four eval metrics on a single GenerationResult.
        Returns an EvalResult with scores and appends to evals.jsonl.
        """
        context_texts = result.source_texts
        context_str = self._format_context(context_texts)
        t0 = time.perf_counter()

        with record_span("eval.evaluate", {
            "query": result.query,
            "model": self.model,
            "context.chunks": len(context_texts),
        }) as eval_span:

            faithfulness     = self._score("faithfulness",     _FAITHFULNESS_PROMPT,     result.query, result.answer, context_str)
            answer_relevance = self._score("answer_relevance", _ANSWER_RELEVANCE_PROMPT, result.query, result.answer, context_str)
            context_precision = self._score("context_precision", _CONTEXT_PRECISION_PROMPT, result.query, result.answer, context_str)
            context_recall   = self._score("context_recall",   _CONTEXT_RECALL_PROMPT,   result.query, result.answer, context_str)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            eval_result = EvalResult(
                trace_id=result.trace_id or "unknown",
                query=result.query,
                answer=result.answer,
                context_texts=context_texts,
                faithfulness=faithfulness,
                answer_relevance=answer_relevance,
                context_precision=context_precision,
                context_recall=context_recall,
                model=result.model,
                eval_model=self.model,
                latency_ms=latency_ms,
            )

            eval_span["eval.faithfulness"]      = faithfulness
            eval_span["eval.answer_relevance"]  = answer_relevance
            eval_span["eval.context_precision"] = context_precision
            eval_span["eval.context_recall"]    = context_recall
            eval_span["eval.overall"]           = eval_result.overall_score
            eval_span["eval.latency_ms"]        = latency_ms

        self._log(eval_result)
        logger.info(
            "Eval complete — overall=%.3f (F=%.2f AR=%.2f CP=%.2f CR=%.2f)",
            eval_result.overall_score,
            faithfulness, answer_relevance, context_precision, context_recall,
        )
        return eval_result

    def evaluate_batch(self, results: list[GenerationResult]) -> list[EvalResult]:
        """Evaluate a list of GenerationResults. Returns eval results in order."""
        eval_results = []
        for i, result in enumerate(results, 1):
            logger.info("Evaluating %d/%d — %r", i, len(results), result.query[:50])
            eval_results.append(self.evaluate(result))
        return eval_results

    def print_report(self, results: list[EvalResult]) -> None:
        """Print a formatted summary table of eval results."""
        if not results:
            print("No eval results to report.")
            return

        print(f"\n{'─'*80}")
        print(f"{'QUERY':<35} {'FAITH':>6} {'RELEV':>6} {'PREC':>6} {'REC':>6} {'OVERALL':>8}")
        print(f"{'─'*35} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

        for r in results:
            query_short = r.query[:33] + ".." if len(r.query) > 35 else r.query
            print(
                f"{query_short:<35} "
                f"{r.faithfulness:>6.2f} "
                f"{r.answer_relevance:>6.2f} "
                f"{r.context_precision:>6.2f} "
                f"{r.context_recall:>6.2f} "
                f"{r.overall_score:>8.3f}"
            )

        avg_overall = sum(r.overall_score for r in results) / len(results)
        avg_faith   = sum(r.faithfulness  for r in results) / len(results)
        avg_relev   = sum(r.answer_relevance for r in results) / len(results)
        avg_prec    = sum(r.context_precision for r in results) / len(results)
        avg_rec     = sum(r.context_recall for r in results) / len(results)

        print(f"{'─'*35} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        print(
            f"{'AVERAGE':<35} "
            f"{avg_faith:>6.2f} "
            f"{avg_relev:>6.2f} "
            f"{avg_prec:>6.2f} "
            f"{avg_rec:>6.2f} "
            f"{avg_overall:>8.3f}"
        )
        print(f"{'─'*80}\n")

    # ------------------------------------------------------------------
    # Internal: LLM scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        metric: str,
        prompt_template: str,
        query: str,
        answer: str,
        context: str,
    ) -> float:
        """Call the LLM judge for a single metric. Returns a 0-1 float."""
        prompt = prompt_template.format(
            query=query,
            answer=answer,
            context=context,
        )

        with record_span(f"eval.{metric}", {"metric": metric}):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                # Strip markdown fences if present
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                score = float(parsed["score"])
                score = max(0.0, min(1.0, score))   # clamp to [0, 1]
                logger.debug("%s score=%.3f — %s", metric, score, parsed.get("reasoning", ""))
                return score
            except Exception:
                logger.exception("Eval metric %r failed — defaulting to 0.0", metric)
                return 0.0

    @staticmethod
    def _format_context(context_texts: list[str]) -> str:
        parts = [f"[{i+1}] {text}" for i, text in enumerate(context_texts)]
        return "\n\n".join(parts)

    @staticmethod
    def _log(result: EvalResult) -> None:
        """Append eval result to evals.jsonl."""
        with open(_EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Utility: load eval history
# ---------------------------------------------------------------------------

def load_evals(limit: int = 200) -> list[dict]:
    """Read the most recent eval results from evals.jsonl."""
    if not _EVAL_LOG.exists():
        return []
    lines = _EVAL_LOG.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records