"""Latency-aware smoke tests for the `data_xl` transcript corpus.

These helpers keep the evaluation lightweight while still verifying that the
pipelines can search across the large JRE subtitle dump.
"""
from pathlib import Path
from statistics import mean

import q_recall as qr
from q_recall.eval import Case, EvalSuite, summarize_latencies

DATA_XL_JRE = Path(__file__).resolve().parents[1] / "data_xl" / "jre"

DEFAULT_CASES = [
    Case(query="Elon Musk starship launch", must_include=["elon"]),
    Case(query="Bernie Sanders medicare plan", must_include=["sanders"]),
    Case(query="Charlie Sheen tiger blood", must_include=["sheen"]),
    Case(query="Fedor Gorst billiards", must_include=["fedor"]),
]


def make_data_xl_stack(
    max_candidates: int = 6, context_lines: int = 1, root: Path = DATA_XL_JRE
) -> qr.Stack:
    """Simple search stack tuned for subtitles (.en.srt)."""
    return qr.Stack(
        qr.MultilingualNormalizer(),
        qr.WidenSearchTerms(extra=[]),
        qr.Grep(
            dir=str(root),
            file_glob="**/*.en.srt",
            context=context_lines,
            case_insensitive=True,
        ),
        qr.BloomKeywordBooster(boost_per_keyword=1.0, max_terms=24),
        qr.Deduplicate(),
        qr.Ranking(max_candidates=max_candidates),
        qr.ContextEnricher(max_tokens=600),
        qr.AdaptiveConcat(max_window_size=4000),
        qr.ComposeAnswer(prompt="Transcript evidence:"),
    )


def run_data_xl_eval(
    top_k: int = 6,
    cases: list[Case] | None = None,
    root: Path = DATA_XL_JRE,
) -> dict[str, float]:
    """Run an EvalSuite over the data_xl corpus and return summary metrics."""
    if not root.exists():
        raise FileNotFoundError(f"data_xl folder not found at {root}")

    suite = EvalSuite(name="data_xl_jre", cases=cases or DEFAULT_CASES)
    results = suite.run(make_data_xl_stack(max_candidates=top_k, root=root))

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    cand_counts = [
        len(r.state.candidates) for r in results if r.state and r.state.candidates
    ]
    failures = [
        {"query": r.case.query, "failures": r.failures}
        for r in results
        if not r.passed
    ]

    return {
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "latency_ms": summarize_latencies(latencies),
        "avg_candidates": mean(cand_counts) if cand_counts else 0.0,
        "failures": failures,
    }


if __name__ == "__main__":
    metrics = run_data_xl_eval()
    print("data_xl JRE eval")
    print(f"Passed: {metrics['passed']}/{metrics['total']}")
    print(
        f"Latency ms avg={metrics['latency_ms']['avg_ms']:.1f} "
        f"p95={metrics['latency_ms']['p95_ms']:.1f}"
    )
    if metrics["failures"]:
        print("Failures:", metrics["failures"])
