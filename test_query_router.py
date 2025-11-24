import pytest

import q_recall as qr


def _mark(label: str):
    def inner(state: qr.State) -> qr.State:
        state.query.meta["route"] = label
        return state

    inner.__name__ = f"mark_{label}"
    return qr.Lambda(inner)


def test_query_router_picks_first_matching_route():
    router = qr.QueryRouter(
        ("lease", lambda s: "lease" in s.query.text, _mark("lease")),
        ("spec", lambda s: "spec" in s.query.text, _mark("spec")),
    )

    state = router("lease obligations summary")

    assert state.query.meta["route"] == "lease"
    assert any(
        ev.op == "QueryRouter" and ev.payload.get("route") == "lease"
        for ev in state.trace
    )


def test_query_router_uses_default_when_no_match():
    router = qr.QueryRouter(
        ("lease", lambda s: "lease" in s.query.text, _mark("lease")),
        default=_mark("fallback"),
    )

    state = router("unrelated question")

    assert state.query.meta["route"] == "fallback"
    assert any(
        ev.op == "QueryRouter" and ev.payload.get("route") == "default"
        for ev in state.trace
    )


def test_query_router_can_require_match():
    router = qr.QueryRouter(
        ("lease", lambda s: "lease" in s.query.text, _mark("lease")),
        require_match=True,
    )

    with pytest.raises(ValueError):
        router("random topic")
