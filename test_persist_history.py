import json

import q_recall as qr


def test_persist_history_writes_jsonl(tmp_path):
    path = tmp_path / "history.jsonl"
    op = qr.PersistHistory(path=path, max_text=100)

    state = qr.State(query=qr.Query(text="hello world"))
    state.candidates.append(
        qr.Candidate(uri="file://demo.txt", score=0.7, snippet="snippet text")
    )
    state.evidence.append(qr.Evidence(text="evidence text", uri="file://demo.txt"))
    state.log("probe", value=1)

    out = op(state)
    assert out is state

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["query"]["text"] == "hello world"
    assert record["candidates"][0]["uri"].endswith("demo.txt")
    assert record["evidence"][0]["text"] == "evidence text"
    assert any(ev["op"] == "probe" for ev in record["trace"])
