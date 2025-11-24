import q_recall as qr

# Guarded pipeline: ensure we have candidates; if not, widen the search terms.
guarded = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.WidenSearchTerms(extra=["overview", "introduction", "summary"]),
    qr.Grep(dir="data"),
    qr.Gate(
        predicate=lambda s: qr.has_candidates(s, 1),
        on_fail=qr.WidenSearchTerms(extra=["appendix", "background"]),
    ),
    qr.Ranking(max_candidates=8),
    qr.ContextEnricher(max_tokens=800),
    qr.Concat(max_window_size=8_000),
    qr.ComposeAnswer(prompt="Answer using only gathered context:"),
)


if __name__ == "__main__":
    state = guarded("what is spec-driven development?")
    print(state.answer)
    state.explain_trace()
