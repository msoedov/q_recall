import re
from collections.abc import Iterable

from .core import Candidate, State, dedup_candidates
from .ops_search import Grep


class ReferenceFollower:
    """Follow textual cross-references (e.g., 'See Note 12', 'Item 7', '§ 2.3').

    Strategy:
      1) Parse existing evidence (or candidate snippets if no evidence yet) for reference patterns.
      2) For each normalized target token, run a local Grep to discover matching files/contexts.
      3) Merge new candidates with dedup + light scoring boost.
    """

    def __init__(
        self,
        dir: str,
        patterns: list[str] = None,
        max_targets: int = 64,
        per_target_boost: float = 0.4,
    ):
        self.dir = dir
        self.max_targets = max_targets
        self.per_target_boost = per_target_boost
        # Default patterns cover common long-doc structures
        self.patterns = patterns or [
            r"See\s+Note\s+(\d+[A-Za-z]?)",
            r"Note\s+(\d+[A-Za-z]?)",
            r"Item\s+(\d+[A-Za-z]?)",
            r"(?:Section|Sec\.|§)\s+(\d+(?:\.\d+)*)",
            r"Appendix\s+([A-Z])",
            r"Exhibit\s+(\d+(?:\.\d+)*)",
            r"Table\s+(\d+(?:\.\d+)*)",
        ]

    def __call__(self, state: State) -> State:
        text_source = self._collect_text(state)
        targets = self._extract_targets(text_source)
        queries = self._normalize_targets(targets)[: self.max_targets]

        new_cands: list[Candidate] = []
        for q in queries:
            # Run a focused grep per target
            s2 = Grep(dir=self.dir)(State(query=state.query.__class__(text=q)))
            # Boost candidates that match explicit references
            for c in s2.candidates:
                c.score += self.per_target_boost
                c.meta.setdefault("ref_hit", []).append(q)
            new_cands.extend(s2.candidates)

        if new_cands:
            state.candidates = dedup_candidates(state.candidates + new_cands)

        state.log(
            "follow_refs",
            targets=len(targets),
            queries=len(queries),
            added=len(new_cands),
        )
        return state

    # --------------------------- helpers ---------------------------
    def _collect_text(self, state: State) -> str:
        if state.evidence:
            return "\n\n".join(e.text for e in state.evidence if e.text)
        # fallback: scan candidate snippets
        parts = []
        for c in state.candidates:
            if c.snippet:
                parts.append(c.snippet)
        return "\n\n".join(parts)

    def _extract_targets(self, text: str) -> list[str]:
        found: set[str] = set()
        for pat in self.patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                token = m.group(0).strip()
                found.add(self._normalize_token(token))
        return sorted(found)

    def _normalize_targets(self, tokens: Iterable[str]) -> list[str]:
        out: list[str] = []
        for t in tokens:
            out.append(t)
            m = re.match(r"(?i)(?:section|sec\.|§)\s+(\d+(?:\.\d+)*)", t)
            if m:
                num = m.group(1)
                out.extend([f"Section {num}", f"§ {num}", f"Sec. {num}"])
            m = re.match(r"(?i)note\s+(\d+[A-Za-z]?)", t)
            if m:
                num = m.group(1)
                out.extend([f"Note {num}", f"See Note {num}"])
            m = re.match(r"(?i)item\s+(\d+[A-Za-z]?)", t)
            if m:
                num = m.group(1)
                out.extend([f"Item {num}"])
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for q in out:
            if q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        return uniq

    def _normalize_token(self, token: str) -> str:
        token = re.sub(r"\s+", " ", token).strip()
        return token
