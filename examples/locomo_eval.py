"""Baseline precision/recall sanity check on the LoCoMo10 eval set.

This uses a simple keyword-overlap retriever to score evidence recall. It is
intended as a lightweight regression guard to ensure the evaluation utilities
can handle larger QA sets.
"""
import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import q_recall as qr
from q_recall.eval import aggregate_prf, precision_recall_f1, summarize_latencies
from q_recall.ops_agent import Op

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
    return [
        (turn["dia_id"], tokenize(turn["text"])) for turn in iter_turns(conversation)
    ]


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
    capture_latency: bool = False,
    max_questions: int | None = None,
) -> dict[str, float]:
    """Run the keyword baseline over the LoCoMo10 QA pairs."""
    raw = Path(dataset_path).read_text(encoding="utf-8")
    dataset = json.loads(raw)

    per_question: list[dict[str, float]] = []
    latencies: list[float] = []
    seen = 0
    for sample in dataset:
        turns = prepare_turns(sample["conversation"])
        for qa in sample["qa"]:
            if categories and qa.get("category") not in categories:
                continue
            if max_questions is not None and seen >= max_questions:
                break
            started = time.perf_counter()
            preds = rank_by_overlap(qa["question"], turns)[:top_k]
            if capture_latency:
                latencies.append((time.perf_counter() - started) * 1000.0)
            per_question.append(precision_recall_f1(qa["evidence"], preds))
            seen += 1
        if max_questions is not None and seen >= max_questions:
            break

    aggregated = aggregate_prf(per_question)
    aggregated["questions"] = len(per_question)
    aggregated["top_k"] = top_k
    aggregated["total_true_positives"] = sum(r["true_positives"] for r in per_question)
    aggregated["total_predicted"] = sum(r["predicted"] for r in per_question)
    aggregated["total_truth"] = sum(r["truth"] for r in per_question)
    if capture_latency:
        aggregated["latency_ms"] = summarize_latencies(latencies)
    return aggregated


# ---------- Stack-friendly evaluation helpers ----------
def _default_pred_extractor(state: qr.State, top_k: int) -> list[str]:
    preds: list[str] = []
    for cand in state.candidates:
        cid = cand.meta.get("dia_id") if cand.meta else None
        cid = cid or cand.uri
        if cid:
            preds.append(cid.replace("file://", ""))
        if len(preds) >= top_k:
            break
    return preds


def run_locomo_stack(
    pipeline: Op | Callable[[qr.State | str], qr.State],
    dataset_path: str | Path = DATA_PATH,
    top_k: int = 5,
    categories: set[int] | None = None,
    predict: Callable[[qr.State, int], list[str]] | None = None,
    capture_latency: bool = False,
    max_questions: int | None = None,
) -> dict[str, float]:
    """Evaluate a qr.Stack/Op over LoCoMo10.

    The pipeline should accept a State or str. Each State gets the conversation
    injected via `state.query.meta['conversation']` so custom Ops can read the
    turns directly.
    """
    raw = Path(dataset_path).read_text(encoding="utf-8")
    dataset = json.loads(raw)
    predict = predict or _default_pred_extractor

    per_question: list[dict[str, float]] = []
    latencies: list[float] = []
    seen = 0
    for sample in dataset:
        for qa in sample["qa"]:
            if categories and qa.get("category") not in categories:
                continue
            if max_questions is not None and seen >= max_questions:
                break
            state = qr.State(
                query=qr.Query(
                    text=qa["question"], meta={"conversation": sample["conversation"]}
                )
            )
            started = time.perf_counter()
            out = pipeline(state) if callable(pipeline) else pipeline
            if not isinstance(out, qr.State):
                raise ValueError("pipeline must return a q_recall.State")
            preds = predict(out, top_k)
            per_question.append(precision_recall_f1(qa["evidence"], preds))
            if capture_latency:
                latencies.append((time.perf_counter() - started) * 1000.0)
            seen += 1
        if max_questions is not None and seen >= max_questions:
            break

    aggregated = aggregate_prf(per_question)
    aggregated["questions"] = len(per_question)
    aggregated["top_k"] = top_k
    aggregated["total_true_positives"] = sum(r["true_positives"] for r in per_question)
    aggregated["total_predicted"] = sum(r["predicted"] for r in per_question)
    aggregated["total_truth"] = sum(r["truth"] for r in per_question)
    if capture_latency:
        aggregated["latency_ms"] = summarize_latencies(latencies)
    return aggregated


class LoCoMoRetriever(Op):
    """Simple keyword-overlap retriever usable inside a Stack."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.name = "LoCoMoRetriever"

    def __call__(self, state: qr.State | str) -> qr.State:
        match state:
            case qr.State():
                return self.forward(state)
            case str():
                return self.forward(qr.State(query=qr.Query(text=state)))
            case _:
                raise ValueError("Expected State or str")

    def forward(self, state: qr.State) -> qr.State:
        convo = state.query.meta.get("conversation")
        if not isinstance(convo, dict):
            state.log("locomo_retriever_error", reason="missing_conversation")
            return state

        turns = prepare_turns(convo)
        preds = rank_by_overlap(state.query.text, turns)[: self.top_k]
        state.candidates = [
            qr.Candidate(uri=pid, score=1.0, snippet=None, meta={"dia_id": pid})
            for pid in preds
        ]
        state.log("locomo_retriever", n=len(state.candidates))
        return state


def make_locomo_stack(top_k: int = 5) -> qr.Stack:
    """Construct a Stack that runs the LoCoMo keyword retriever."""
    return qr.Stack(LoCoMoRetriever(top_k=top_k))


def benchmark_locomo_stack(
    top_k: int = 5,
    categories: set[int] | None = None,
    max_questions: int | None = 50,
) -> dict[str, float]:
    """Lightweight benchmark with latency stats for the LoCoMo stack."""
    stack = make_locomo_stack(top_k=top_k)
    return run_locomo_stack(
        stack,
        top_k=top_k,
        categories=categories,
        capture_latency=True,
        max_questions=max_questions,
    )


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
