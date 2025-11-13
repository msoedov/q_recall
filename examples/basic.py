import q_recall as qr

mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.Branch(
        qr.Stack(qr.Grep(dir="./data"), qr.Ranking(max_candidates=10)),
        qr.Stack(qr.Glob(dir="./data_xl"), qr.Ranking(max_candidates=5)),
    ),
    qr.Lambda(lambda state: state),
    qr.Deduplicate(),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer(),
)

if __name__ == "__main__":
    state = mem0("What is the meaning of life?")
    print(state.answer)
    state.explain_trace()
