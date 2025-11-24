import q_recall as qr


def test_keyword_booster_recovers_and_boosts():
    booster = qr.BloomKeywordBooster(boost_per_keyword=2.0, max_terms=6)
    state = qr.State(
        query=qr.Query(text="lease obligations", meta={"search_terms": ["lease"]}),
        candidates=[
            qr.Candidate(
                uri="file://a",
                score=0.0,
                snippet="Lease obligations equipment rentals",
            ),
            qr.Candidate(uri="file://b", score=0.0, snippet="irrelevant text"),
        ],
    )

    out = booster(state)

    assert out.candidates[0].score == 8.0
    terms = [t.lower() for t in out.query.meta["search_terms"]]
    assert {"lease", "obligations", "equipment", "rentals"}.issubset(terms)
    assert out.candidates[0].meta["keyword_hits"] == [
        "Lease",
        "obligations",
        "equipment",
        "rentals",
    ]


def test_keyword_booster_seeds_from_query_and_respects_limit():
    booster = qr.BloomKeywordBooster(boost_per_keyword=1.0, max_terms=3)
    state = qr.State(
        query=qr.Query(text="alpha beta gamma delta"),
        candidates=[qr.Candidate(uri="file://c", score=0.0, snippet="delta epsilon")],
    )

    out = booster(state)

    assert out.query.meta["search_terms"] == ["alpha", "beta", "gamma"]
    assert out.candidates[0].score == 0.0
