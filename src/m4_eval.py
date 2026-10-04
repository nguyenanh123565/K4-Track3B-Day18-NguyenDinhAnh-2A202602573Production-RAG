from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from math import isfinite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ("faithfulness", "answer_relevancy",
                    "context_precision", "context_recall")
    fallback = {**dict.fromkeys(metric_names, 0.0), "per_question": []}
    try:
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("All evaluation inputs must have the same length")
        if not questions:
            return fallback

        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                           context_precision, context_recall])
        frame = result.to_pandas()
        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
            **{name: float(row[name]) for name in metric_names},
        ) for _, row in frame.iterrows()]
        return {**{name: float(result[name]) for name in metric_names},
                "per_question": per_question}
    except Exception as exc:
        print(f"RAGAS evaluation failed: {exc}", file=sys.stderr)
        # Compatibility fallback for failed evaluation, not measured scores.
        return fallback



def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("LLM hallucination", "Tighten the prompt and set temperature=0"),
        "context_recall": ("Retrieval missed relevant chunks", "Improve chunking and BM25 retrieval"),
        "context_precision": ("Irrelevant context", "Improve reranking and metadata filtering"),
        "answer_relevancy": ("Answer is off-topic", "Improve the generation prompt"),
    }
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name)) for name in diagnostic_tree}
        # RAGAS may return NaN for failed metrics; those rows cannot be ranked.
        if not all(isfinite(score) for score in scores.values()):
            continue
        average = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({"question": result.question, "worst_metric": worst_metric,
                         "score": average, "diagnosis": diagnosis,
                         "suggested_fix": suggested_fix})
    return sorted(failures, key=lambda failure: failure["score"])[:bottom_n]



def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
