import pytest

import q_recall as qr
import q_recall.utils as utils
from q_recall.core import TraceEvent


def test_timer_yields_elapsed_callable(monkeypatch):
    timeline = iter([1.0, 1.05, 1.15])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(timeline))

    with utils.timer() as elapsed:
        assert elapsed() == pytest.approx(0.05)
        assert elapsed() == pytest.approx(0.15)


def test_summarize_timings_rolls_up_per_op_and_overall():
    state = qr.State(query=qr.Query(text="demo"))
    state.log("time", name="A", seconds=0.1)
    state.log("time", name="A", seconds=0.2)
    state.log("time", name="B", seconds=0.05)
    state.log("time_overall", name="Stack", seconds=0.5)

    summary = utils.summarize_timings(state)

    assert summary["per_op"] == {"A": pytest.approx(0.3), "B": 0.05}
    assert summary["overall"] == pytest.approx(0.5)


def test_summarize_timings_defaults_unknown_for_missing_name():
    state = qr.State(query=qr.Query(text="demo"))
    state.trace.append(TraceEvent(op="time", payload={"seconds": 0.12}))

    summary = utils.summarize_timings(state)

    assert summary["per_op"] == {"unknown": 0.12}
    assert summary["overall"] is None
