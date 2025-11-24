"""Baseline precision/recall sanity check on the LoCoMo10 eval set.

This uses a simple keyword-overlap retriever to score evidence recall. It is
intended as a lightweight regression guard to ensure the evaluation utilities
can handle larger QA sets.
"""
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from q_recall.eval import aggregate_prf, precision_recall_f1

DATA_PATH = Path(__file__).resolve().parents[1] / "data_eval" / "locomo10.json"
TOKEN_PATTERN = re.compile(r"\w+")


def iter_turns(conversation: dict) -> Iterator[dict]:
    """Yield individual dialogue turns with ids and text."""
    for key, val in conversation.items():
        if key.startswith("session_") and isinstance(val, list):
            for turn in val:
                if isinstance(turn, dict) and turn.get("dia_id") and turn.get("text"):
                    yield turn


def tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def prepare_turns(conversation: dict) -> list[tuple[str, set[str]]]:
    """Flatten and tokenize conversation turns once per sample."""
    return [(turn["dia_id"], tokenize(turn["text"])) for turn in iter_turns(conversation)]


def rank_by_overlap(question: str, turns: Iterable[tuple[str, set[str]]]) -> list[str]:
    """Score dialogue turns by keyword overlap with the question."""
    q_tokens = tokenize(question)
    scored: list[tuple[str, int]] = []
    for dia_id, tokens in turns:
        score = len(q_tokens & tokens)
        if score:
            scored.append((dia_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [dia for dia, _ in scored]


def run_locomo_baseline(
    dataset_path: str | Path = DATA_PATH,
    top_k: int = 5,
    categories: set[int] | None = None,
) -> dict[str, float]:
    """Run the keyword baseline over the LoCoMo10 QA pairs."""
    raw = Path(dataset_path).read_text(encoding="utf-8")
    dataset = json.loads(raw)

    per_question: list[dict[str, float]] = []
    for sample in dataset:
        turns = prepare_turns(sample["conversation"])
        for qa in sample["qa"]:
            if categories and qa.get("category") not in categories:
                continue
            preds = rank_by_overlap(qa["question"], turns)[:top_k]
            per_question.append(precision_recall_f1(qa["evidence"], preds))

    aggregated = aggregate_prf(per_question)
    aggregated["questions"] = len(per_question)
    aggregated["top_k"] = top_k
    aggregated["total_true_positives"] = sum(r["true_positives"] for r in per_question)
    aggregated["total_predicted"] = sum(r["predicted"] for r in per_question)
    aggregated["total_truth"] = sum(r["truth"] for r in per_question)
    return aggregated


if __name__ == "__main__":
    metrics = run_locomo_baseline()
    print(f"LoCoMo10 keyword baseline (top_k={metrics['top_k']})")
    print(f"Questions: {metrics['questions']}")
    print(
        "Macro P/R/F1: "
        f"{metrics['macro_precision']:.4f} / "
        f"{metrics['macro_recall']:.4f} / "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        "Micro P/R/F1: "
        f"{metrics['micro_precision']:.4f} / "
        f"{metrics['micro_recall']:.4f} / "
        f"{metrics['micro_f1']:.4f}"
    )
