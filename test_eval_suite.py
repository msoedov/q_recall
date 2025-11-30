import time

import q_recall as qr


def _demo_pipeline(query: str) -> qr.State:
    state = qr.State(query=qr.Query(text=query))
    if "spec" in query:
        state.answer = "Spec is written prior to implementation."
    else:
        state.answer = "Lease obligations summary."
    if "lease" in query:
        state.candidates.append(
            qr.Candidate(uri="file://reports/lease_note.md", score=0.9)
        )
    return state


def test_eval_suite_passes_expectations():
    suite = qr.EvalSuite(
        name="demo",
        cases=[
            qr.Case(
                query="spec process",
                must_include=["spec", "prior to implementation"],
            ),
            qr.Case(query="lease obligations", must_hit_files=["lease_note.md"]),
        ],
    )

    results = suite.run(_demo_pipeline)

    assert [r.passed for r in results] == [True, True]


def test_eval_suite_surfaces_failures():
    suite = qr.EvalSuite(
        name="demo",
        cases=[
            qr.Case(query="spec process", must_include=["nonexistent"]),
            qr.Case(query="lease obligations", must_hit_files=["missing.md"]),
        ],
    )

    results = suite.run(_demo_pipeline)

    assert results[0].passed is False
    assert "answer missing 'nonexistent'" in results[0].failures
    assert results[1].passed is False
    assert "missing file hit 'missing.md'" in results[1].failures


def test_eval_suite_records_latency_ms():
    suite = qr.EvalSuite(
        name="latency",
        cases=[qr.Case(query="ping")],
    )

    def slow_pipeline(q: str) -> qr.State:
        time.sleep(0.002)
        return qr.State(query=qr.Query(text=q), answer="pong")

    results = suite.run(slow_pipeline)

    assert results[0].latency_ms is not None
    assert results[0].latency_ms >= 1.0
