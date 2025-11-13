# Placeholder for future budgets, caching, tracing helpers.
import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield lambda: (time.perf_counter() - start)
    finally:
        pass


def summarize_timings(state):
    """Return a dict of total time per op name (seconds) based on 'time' trace events."""
    totals = {}
    for ev in getattr(state, "trace", []):
        if ev.op == "time":
            name = ev.payload.get("name", "unknown")
            dt = ev.payload.get("seconds", 0.0)
            totals[name] = totals.get(name, 0.0) + dt
    overall = next(
        (ev.payload.get("seconds") for ev in state.trace if ev.op == "time_overall"),
        None,
    )
    return {"per_op": totals, "overall": overall}
