import q_recall as qr

mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.Branch(
        qr.Stack(qr.Grep(dir="../data"), qr.Ranking(max_candidates=10)),
        qr.Stack(qr.Glob(dir="../data"), qr.Ranking(max_candidates=5)),
    ),
    qr.Deduplicate(),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer(),
)

if __name__ == "__main__":
    state = mem0("What is the meaning of life?")
    print(state.answer)
    for ev in state.trace:
        print(ev.op, ev.payload)
