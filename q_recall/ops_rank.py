import re

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


class BloomKeywordBooster:
    """Recover keywords from query/candidates and boost matches.

    - Seeds the bloom with existing search_terms or tokens from the query.
    - Boosts candidates that hit known/recovered keywords.
    - Expands search_terms (bounded) with newly discovered keywords for later ops.
    """

    def __init__(self, boost_per_keyword=0.5, max_terms=64, min_len=4):
        self.boost_per_keyword = boost_per_keyword
        self.max_terms = max_terms
        self.min_len = min_len
        self._pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{%d,}" % (min_len - 1))
        self.name = "BloomKeywordBooster"

    def __call__(self, state: State) -> State:
        seeds = state.query.meta.get("search_terms") or self._extract_keywords(
            state.query.text
        )
        terms = self._dedup(seeds)[: self.max_terms]
        bloom = {t.lower() for t in terms}

        added_terms: list[str] = []
        boosted_candidates = 0

        for cand in state.candidates:
            text = cand.snippet or ""
            tokens = self._extract_keywords(text)
            hits = 0
            local_seen: set[str] = set()
            local_hits: list[str] = []
            for tok in tokens:
                key = tok.lower()
                if key in local_seen:
                    continue
                local_seen.add(key)
                local_hits.append(tok)
                if key in bloom:
                    hits += 1
                elif len(terms) + len(added_terms) < self.max_terms:
                    bloom.add(key)
                    added_terms.append(tok)
                    hits += 1  # credit candidates that surface new useful terms
            if hits:
                cand.score += self.boost_per_keyword * hits
                cand.meta.setdefault("keyword_hits", []).extend(local_hits)
                boosted_candidates += 1

        if added_terms:
            terms = self._dedup(terms + added_terms)[: self.max_terms]
        state.query.meta["search_terms"] = terms
        state.log(
            self.name,
            added=len(added_terms),
            boosted=boosted_candidates,
            terms=len(terms),
        )
        return state

    # ---------- helpers ----------
    def _extract_keywords(self, text: str | None) -> list[str]:
        if not text:
            return []
        found: list[str] = []
        seen = set()
        for m in self._pattern.finditer(text):
            tok = m.group(0)
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(tok)
        return found

    def _dedup(self, terms: list[str]) -> list[str]:
        seen = set()
        uniq = []
        for t in terms:
            key = (t or "").strip()
            low = key.lower()
            if not key or low in seen:
                continue
            seen.add(low)
            uniq.append(key)
        return uniq


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
