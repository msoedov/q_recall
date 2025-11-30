from pathlib import Path

import pytest

from examples import data_xl_eval

DATA_ROOT = Path(__file__).resolve().parent / "data_xl"

if not DATA_ROOT.exists():
    pytest.skip(
        "data_xl corpus not available; skipping large-folder eval",
        allow_module_level=True,
    )


def test_data_xl_eval_smoke():
    metrics = data_xl_eval.run_data_xl_eval(top_k=6, root=DATA_ROOT)

    assert metrics["passed"] == metrics["total"]
    assert metrics["avg_candidates"] > 0
    assert metrics["latency_ms"]["max_ms"] >= 0.0
    assert metrics["failures"] == []
