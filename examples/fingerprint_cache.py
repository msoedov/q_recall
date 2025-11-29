import q_recall as qr

# Session-local cache: remembers fragments by fingerprint so repeated runs
# skip re-reading similar blocks.
cache = qr.FingerprintCache(ttl="session", match="similar_blocks")

mem_cache = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.WidenSearchTerms(),
    qr.Grep(dir="./data"),
    cache,
    qr.Ranking(max_candidates=8),
    qr.ContextEnricher(max_tokens=800),
    qr.AdaptiveConcat(max_window_size=4000),
    qr.ComposeAnswer(prompt="Cache-aware summary:"),
)


def cached_search(query: str) -> qr.State:
    return mem_cache(query)


if __name__ == "__main__":
    first = cached_search("dopamine motivation")
    print("First call candidates:", len(first.candidates))

    second = cached_search("motivation dopamine focus")
    fp_event = next(ev.payload for ev in reversed(second.trace) if ev.op == "FingerprintCache")
    print("Second call cache hits:", fp_event.get("hits"))
    print(second.answer)
