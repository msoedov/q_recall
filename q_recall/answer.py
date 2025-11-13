import re

from .core import State


class ComposeAnswer:
    def __init__(self, prompt="Summarize precisely:"):
        self.prompt = prompt
        self.name = "Answer"

    def __call__(self, state: State) -> State:
        text = "\n".join(e.text for e in state.evidence if e.text)
        if len(text) > 50_000:
            text = text[:50_000]
        state.answer = f"{self.prompt}\n\n{text}" if text else "No evidence found."
        state.log("answer", chars=len(text))
        return state


class LLMSearchTermExtractor:
    """Stub: rule-based term extractor. Swap with an LLM call if desired."""

    def __init__(self, extra_terms=None):
        self.extra = extra_terms or []
        self.name = "LLMExtractor"

    def __call__(self, state: State) -> State:
        q = state.query.text
        terms = list({t.lower() for t in re.findall(r"[A-Za-zА-Яа-я0-9\-]{4,}", q)})
        terms += self.extra
        terms = sorted(terms, key=len, reverse=True)[:6]
        state.query.meta["search_terms"] = terms
        state.log("term_extract", terms=terms)
        return state


class MultilingualNormalizer:
    def __call__(self, state: State) -> State:
        txt = state.query.text
        is_cyr = bool(re.search(r"[А-Яа-я]", txt))
        state.query.lang = "ru" if is_cyr else "en"
        state.log("lang_norm", lang=state.query.lang)
        return state
