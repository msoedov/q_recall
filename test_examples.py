import json
import importlib

from inline_snapshot import snapshot


def test_basic():
    from examples.basic import mem0

    state = mem0("what is spec-driven development?")
    assert len(state.trace) == snapshot(37)


def test_sec_lease():
    from examples.sec_lease import lease_agent

    state = lease_agent("what is spec-driven development?")
    assert len(state.trace) == snapshot(27)


def test_self_healing():
    from examples.self_healing import mem0

    state = mem0("what is spec-driven development?")
    assert len(state.trace) == snapshot(78)


def test_persist_history_example(tmp_path, monkeypatch):
    monkeypatch.setenv("Q_RECALL_HISTORY", str(tmp_path / "history.jsonl"))
    ph = importlib.reload(importlib.import_module("examples.persist_history"))

    state = ph.mem_with_history("what is spec-driven development?")

    history_path = tmp_path / "history.jsonl"
    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["query"]["text"] == "what is spec-driven development?"
    assert record["answer"] == state.answer


def test_query_router_example():
    from examples.query_router import router

    state = router("lease obligations overview")
    assert state.query.meta.get("route") == "lease"

    fallback = router("totally unrelated question")
    assert fallback.query.meta.get("route") == "general"
