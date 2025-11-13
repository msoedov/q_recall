import q_recall as qr

# Register a data directory (create ./data and add some .txt files to try this).
db = qr.ParadigmDB()
db.register_fs(
    "data", "../data"
)  # (not strictly needed for Grep/Glob which take dir directly)

mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.Branch(
        qr.Stack(qr.Grep(dir="../data"), qr.Ranking(max_candidates=10)),
        qr.Stack(qr.Glob(dir="../data"), qr.Ranking(max_candidates=5)),
    ),
    qr.ReferenceFollower(dir="../data"),
    qr.Deduplicate(),
    qr.Lambda(lambda state: state),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer(),
)

if __name__ == "__main__":
    state = mem0(qr.State(query=qr.Query(text="what is spec-driven development?")))
    print(state.answer)
    for ev in state.trace:
        print(ev.op, ev.payload)
