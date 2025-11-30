import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from .core import Query, State
from .ops_agent import Op


@dataclass
class Case:
    query: str
    must_include: list[str] = field(default_factory=list)
    must_hit_files: list[str] = field(default_factory=list)
    name: str | None = None

    def evaluate(self, state: State) -> "CaseResult":
        failures: list[str] = []
        answer = (state.answer or "").lower()

        for term in self.must_include:
            if term.lower() not in answer:
                failures.append(f"answer missing '{term}'")

        if self.must_hit_files:
            uris = [c.uri for c in state.candidates if c.uri]
            for fname in self.must_hit_files:
                if not any(fname in uri for uri in uris):
                    failures.append(f"missing file hit '{fname}'")

        return CaseResult(
            case=self, passed=not failures, failures=failures, state=state
        )


@dataclass
class CaseResult:
    case: Case
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None
    state: State | None = None
    latency_ms: float | None = None


class EvalSuite:
    def __init__(self, name: str, cases: Sequence[Case]):
        self.name = name
        self.cases = list(cases)

    def run(
        self,
        pipeline: Callable[[str], State] | Op,
        stop_on_fail: bool = False,
    ) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in self.cases:
            started = time.perf_counter()
            try:
                state = self._run_case(pipeline, case.query)
                result = case.evaluate(state)
            except Exception as e:
                result = CaseResult(case=case, passed=False, error=str(e))
            result.latency_ms = (time.perf_counter() - started) * 1000.0
            results.append(result)
            if stop_on_fail and not result.passed:
                break
        return results

    def report(self, results: Iterable[CaseResult]) -> dict:
        results = list(results)
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        print(f"{self.name}: {passed}/{total} passed")
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        if latencies:
            print(
                f"  latency ms: p50={percentile(latencies, 50):.1f}, "
                f"p95={percentile(latencies, 95):.1f}, "
                f"avg={mean(latencies):.1f}"
            )
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            label = r.case.name or r.case.query
            print(f"- {status}: {label}")
            if r.error:
                print(f"    error: {r.error}")
            for fail in r.failures:
                print(f"    missing: {fail}")
        return {"passed": passed, "total": total}

    def _run_case(self, pipeline, query: str) -> State:
        if isinstance(pipeline, Op):
            return pipeline(query)

        if callable(pipeline):
            out = pipeline(query)
            if isinstance(out, State):
                return out

        # Allow users to pass in Ops or simple callables; fail loudly otherwise.
        raise ValueError("pipeline must be an Op or callable returning a State")


def precision_recall_f1(
    truth: Sequence[str], predicted: Sequence[str]
) -> dict[str, float]:
    """Compute retrieval metrics for a single example.

    Inputs are sequences of identifiers (e.g., evidence ids). Predictions are
    deduplicated in-order before scoring.
    """
    truth_set = set(truth)
    pred_dedup = list(dict.fromkeys(predicted))
    pred_set = set(pred_dedup)

    tp = len(truth_set & pred_set)
    precision = tp / len(pred_dedup) if pred_dedup else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    hit = 1.0 if tp > 0 else 0.0
    exact_match = 1.0 if truth_set and truth_set.issubset(pred_set) else 0.0

    mrr = 0.0
    for idx, pred in enumerate(pred_dedup, start=1):
        if pred in truth_set:
            mrr = 1.0 / idx
            break

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "exact_match": exact_match,
        "mrr": mrr,
        "true_positives": tp,
        "predicted": len(pred_dedup),
        "truth": len(truth_set),
    }


def aggregate_prf(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Aggregate a list of precision/recall rows into micro + macro metrics."""
    if not rows:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "hit_rate": 0.0,
            "exact_match_rate": 0.0,
            "mrr": 0.0,
        }

    macro_precision = sum(r["precision"] for r in rows) / len(rows)
    macro_recall = sum(r["recall"] for r in rows) / len(rows)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    macro_hit = sum(r["hit"] for r in rows) / len(rows)
    macro_exact = sum(r["exact_match"] for r in rows) / len(rows)
    macro_mrr = sum(r["mrr"] for r in rows) / len(rows)

    tp = sum(r["true_positives"] for r in rows)
    predicted = sum(r["predicted"] for r in rows)
    truth = sum(r["truth"] for r in rows)

    micro_precision = tp / predicted if predicted else 0.0
    micro_recall = tp / truth if truth else 0.0
    micro_f1 = (
        0.0
        if (micro_precision + micro_recall) == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "hit_rate": macro_hit,
        "exact_match_rate": macro_exact,
        "mrr": macro_mrr,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = k - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(latencies: Sequence[float]) -> dict[str, float]:
    """Return common latency statistics in milliseconds."""
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": mean(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "max_ms": max(latencies),
    }
