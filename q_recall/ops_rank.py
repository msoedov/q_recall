from .core import Evidence, State, dedup_candidates


class Ranking:
    def __init__(self, max_candidates=20, keyword_boost=None):
        self.k = max_candidates
        self.boost = keyword_boost or []
        self.name = "Ranking"

    def __call__(self, state: State) -> State:
        for c in state.candidates:
            for kw in self.boost:
                if kw.lower() in (c.snippet or "").lower():
                    c.score += 0.3
        state.candidates = sorted(
            state.candidates, key=lambda c: c.score, reverse=True
        )[: self.k]
        state.log("rank", kept=len(state.candidates))
        return state


class Concat:
    def __init__(self, max_window_size=100_000):
        self.max = max_window_size
        self.name = "Concat"

    def __call__(self, state: State) -> State:
        text = ""
        for c in state.candidates:
            chunk = c.snippet or _safe_read(c.uri)
            if chunk is None:
                continue
            if len(text) + len(chunk) > self.max:
                break
            text += f"\n\n----- {c.uri}\n{chunk}"
        if text:
            state.evidence.append(Evidence(text=text, uri=None))
        state.log("concat", size=len(text))
        return state


def _safe_read(uri: str):
    from pathlib import Path

    try:
        p = Path(uri.replace("file://", ""))
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


class ContextEnricher:
    def __init__(self, max_tokens=1000):
        self.max = max_tokens
        self.name = "ContextEnricher"

    def __call__(self, state: State) -> State:
        new = []
        for c in state.candidates:
            if c.snippet and len(c.snippet) < self.max:
                new.append(c)
                continue
            full = _safe_read(c.uri)
            if full:
                c.snippet = full[: self.max * 4]  # generous bytes approximation
            new.append(c)
        state.candidates = new
        state.log("enrich", n=len(new))
        return state


class Lambda:
    def __init__(self, func):
        self.func = func
        self.name = "Lambda"

    def __call__(self, state: State) -> State:
        state.log("lambda", func=self.func.__name__)
        return self.func(state)


class Deduplicate:
    def __call__(self, state: State) -> State:
        state.candidates = dedup_candidates(state.candidates)
        state.log("dedup", n=len(state.candidates))
        return state
