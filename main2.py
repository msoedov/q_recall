import q_recall as qr


PlanB = qr.Stack(
    qr.Glob(dir="../data", pattern="**/*.(md|txt|py|ts|srt)"),
    qr.Ranking(max_candidates=30),
)

mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.SelfHeal(
        op=qr.Stack(qr.Grep(dir="../data"), qr.Ranking(max_candidates=10)),
        fallback=PlanB,
        post_condition=lambda s: qr.has_candidates(s, 1),
        on_weak=lambda s: qr.widen_search_terms(
            s,
            extra=[
                "summary",
                "abstract",
                "description",
                "characteristic",
                "pattern",
                "example",
                "illustration",
            ],
        ),
        retries=1,
        backoff=0.25,
    ),
    qr.Deduplicate(),
    qr.SelfHeal(
        op=qr.ContextEnricher(max_tokens=1000),
        fallback=qr.ContextEnricher(max_tokens=3000),
        post_condition=lambda s: qr.has_candidates(s, 1),
    ),
    qr.SelfHeal(
        op=qr.AdaptiveConcat(max_window_size=10_000),
        fallback=qr.AdaptiveConcat(max_window_size=6_000),
        post_condition=lambda s: qr.has_evidence(s, min_chars=600),
    ),
    qr.StagnationGuard(
        min_gain=1,
        on_stall=lambda s: qr.widen_search_terms(s, extra=["summary", "abstract"]),
    ),
    qr.SelfHeal(
        op=qr.ComposeAnswer(prompt="provide a concise answer based on the context"),
        fallback=qr.ComposeAnswer(prompt="provide a concise answer with direct quotes"),
        post_condition=lambda s: s.answer is not None and len(s.answer) > 200,
    ),
)


if __name__ == "__main__":
    state = mem0(qr.State(query=qr.Query(text="what is spec-driven development?")))
    print(state.answer)
    for ev in state.trace:
        print(ev.op, ev.payload)
