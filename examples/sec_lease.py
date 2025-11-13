import q_recall as qr

lease_agent = qr.Stack(
    qr.Branch(
        qr.Stack(qr.Grep(dir="sec"), qr.Ranking(max_candidates=20)),
        qr.Stack(qr.Glob(dir="sec", pattern="**/*.txt"), qr.Ranking(max_candidates=5)),
    ),
    qr.Deduplicate(),
    qr.ContextEnricher(max_tokens=2000),
    qr.Concat(max_window_size=80_000),
    # A simple ref follower could be added here (left minimal to keep example runnable)
    qr.Ranking(max_candidates=30, keyword_boost=["lease", "Note", "Item 7", "MD&A"]),
    qr.Concat(max_window_size=160_000),
    qr.ComposeAnswer(
        prompt="Compute final lease obligations with adjustments and show the arithmetic:"
    ),
)

if __name__ == "__main__":
    q = "Total lease obligations including discontinued ops and terminations"
    state = lease_agent(qr.State(query=qr.Query(text=q)))
    print(state.answer)
    for ev in state.trace:
        print(ev.op, ev.payload)
