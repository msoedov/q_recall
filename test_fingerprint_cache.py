from pathlib import Path

import q_recall as qr
from q_recall.core import Candidate, Evidence, Query, State


def _fp_event(state: State) -> dict:
    for ev in reversed(state.trace):
        if ev.op == "FingerprintCache":
            return ev.payload
    return {}


def test_fingerprint_cache_hits_similar_candidates(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Alpha cache avoids re-reading fragments. Local cache keeps blocks alive."
    )
    (tmp_path / "b.txt").write_text(
        "Alpha cache keeps blocks alive and avoids rereading the same fragment twice."
    )

    cache = qr.FingerprintCache(ttl="session", match="similar_blocks")
    pipeline = qr.Stack(qr.Grep(dir=str(tmp_path)), cache)

    first = pipeline("Alpha cache")
    assert _fp_event(first)["hits"] == 0

    second = pipeline("Alpha cache")
    payload = _fp_event(second)
    assert payload["hits"] >= 1
    assert any(
        c.meta.get("fingerprint_cache", {}).get("hit") for c in second.candidates
    )


def test_fingerprint_cache_reuses_snippet_without_disk(monkeypatch, tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Cache keeps fragments warm for reuse.")
    uri = f"file://{path}"

    cache = qr.FingerprintCache(ttl="session")
    state1 = State(
        query=Query(text="demo"),
        candidates=[
            Candidate(uri=uri, snippet="Cache keeps fragments warm for reuse.")
        ],
    )
    cache(state1)

    state2 = State(
        query=Query(text="demo"),
        candidates=[Candidate(uri=uri, snippet=None, meta={"line": 1})],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("disk read")

    monkeypatch.setattr(Path, "read_text", boom)

    out = cache(state2)

    assert out.candidates[0].snippet is not None
    assert _fp_event(out)["hits"] >= 1


def test_fingerprint_cache_drops_duplicate_evidence():
    cache = qr.FingerprintCache(ttl="session")
    text = "Repeated fragment about caching and reuse."

    cache(State(query=Query(text="demo"), evidence=[Evidence(text=text)]))

    state = State(
        query=Query(text="demo"),
        evidence=[Evidence(text=text), Evidence(text="Fresh block goes through.")],
    )

    out = cache(state)
    payload = _fp_event(out)

    assert len(out.evidence) == 1
    assert out.evidence[0].text == "Fresh block goes through."
    assert payload["evidence_dropped"] == 1
