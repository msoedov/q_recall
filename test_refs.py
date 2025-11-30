from pathlib import Path

import q_recall as qr


def test_reference_follower_multi_hop(tmp_path):
    main = tmp_path / "main.txt"
    main.write_text("See Note 2 for Appendix A details.")
    note = tmp_path / "note2.txt"
    note.write_text("Note 2 revenue guidance. Appendix A describes middleware.")
    appendix = tmp_path / "appendix_a.txt"
    appendix.write_text("Appendix A points to middleware.py and config sections.")
    middleware = tmp_path / "middleware.py"
    middleware.write_text("# middleware.py config\nmiddleware = True\n")

    seed = qr.Candidate(uri=f"file://{main}", snippet=main.read_text())
    state = qr.State(query=qr.Query(text="lease obligations"), candidates=[seed])

    follower = qr.ReferenceFollower(
        dir=str(tmp_path), mode="aggressive", max_hops=3, prune=False
    )
    out = follower(state)

    names = {Path(c.uri.replace("file://", "")).name for c in out.candidates}
    assert "note2.txt" in names
    assert "appendix_a.txt" in names
    assert "middleware.py" in names


def test_reference_follower_prunes_when_stalled(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("plain text without references")
    seed = qr.Candidate(uri=f"file://{doc}", snippet=doc.read_text())
    state = qr.State(query=qr.Query(text="nothing"), candidates=[seed])

    follower = qr.ReferenceFollower(
        dir=str(tmp_path), mode="aggressive", max_hops=5, prune=True
    )
    out = follower(state)

    assert len(out.candidates) == 1
    assert out.trace[-1].payload["added"] == 0
