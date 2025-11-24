import q_recall as qr


def _mark(label: str):
    def inner(state: qr.State) -> qr.State:
        state.query.meta["route"] = label
        return state

    inner.__name__ = f"mark_{label}"
    return qr.Lambda(inner)


lease_route = qr.Stack(
    _mark("lease"),
    qr.Grep(dir="data"),
    qr.Ranking(max_candidates=20, keyword_boost=["lease", "Note"]),
    qr.ContextEnricher(max_tokens=2000),
    qr.Concat(max_window_size=80_000),
    qr.ComposeAnswer(prompt="Lease-focused answer:"),
)

spec_route = qr.Stack(
    _mark("spec"),
    qr.Grep(dir="data"),
    qr.Ranking(max_candidates=12, keyword_boost=["spec", "design"]),
    qr.ContextEnricher(max_tokens=1500),
    qr.Concat(max_window_size=60_000),
    qr.ComposeAnswer(prompt="Spec-focused answer:"),
)

fallback_route = qr.Stack(
    _mark("general"),
    qr.Glob(dir="data"),
    qr.Ranking(max_candidates=6),
    qr.ContextEnricher(max_tokens=800),
    qr.Concat(max_window_size=8_000),
    qr.ComposeAnswer(prompt="General answer from recent files:"),
)


router = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.WidenSearchTerms(),
    qr.QueryRouter(
        ("lease_route", lambda s: "lease" in s.query.text.lower(), lease_route),
        ("spec_route", lambda s: "spec" in s.query.text.lower(), spec_route),
        default=fallback_route,
    ),
)


if __name__ == "__main__":
    query = "What are our lease obligations?"
    state = router(query)
    print(f"Route picked: {state.query.meta.get('route')}")
    print(state.answer)
    state.explain_trace()
