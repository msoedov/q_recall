import os
import re
from collections.abc import Iterable
from pathlib import Path

from .core import Candidate, State, dedup_candidates
from .ops_search import Grep


class ReferenceFollower:
    """Follow textual cross-references (e.g., 'See Note 12', 'Item 7', '§ 2.3').

    Strategy:
      1) Parse existing evidence (or candidate snippets if no evidence yet) for reference patterns.
      2) Add heuristic path hints (anchors, filenames) when in balanced/aggressive mode.
      3) For each normalized target token, run a local Grep to discover matching files/contexts.
      4) Merge new candidates with dedup + light scoring boost, repeat up to `max_hops`.
    """

    def __init__(
        self,
        dir: str | None = None,
        patterns: list[str] | None = None,
        max_targets: int | None = None,
        per_target_boost: float | None = None,
        mode: str = "balanced",
        max_hops: int | None = None,
        prune: bool | None = None,
    ):
        presets = {
            "light": {
                "max_targets": 32,
                "per_target_boost": 0.25,
                "max_hops": 1,
                "prune": True,
                "heuristics": False,
            },
            "balanced": {
                "max_targets": 64,
                "per_target_boost": 0.4,
                "max_hops": 2,
                "prune": True,
                "heuristics": True,
            },
            "aggressive": {
                "max_targets": 96,
                "per_target_boost": 0.5,
                "max_hops": 4,
                "prune": True,
                "heuristics": True,
            },
        }
        cfg = presets.get(mode, presets["balanced"])

        self.root = Path(dir).expanduser().resolve() if dir else None
        self.mode = mode
        self.max_targets = (
            max_targets if max_targets is not None else cfg["max_targets"]
        )
        self.per_target_boost = (
            per_target_boost
            if per_target_boost is not None
            else cfg["per_target_boost"]
        )
        self.max_hops = max_hops if max_hops is not None else cfg["max_hops"]
        self.prune = cfg["prune"] if prune is None else prune
        self.heuristics_enabled = cfg["heuristics"]
        # Default patterns cover common long-doc structures and anchors
        self.patterns = patterns or [
            r"See\s+Note\s+(\d+[A-Za-z]?)",
            r"Note\s+(\d+[A-Za-z]?)",
            r"Item\s+(\d+[A-Za-z]?)",
            r"(?:Section|Sec\.|§)\s+(\d+(?:\.\d+)*)",
            r"Appendix\s+([A-Z])",
            r"Exhibit\s+(\d+(?:\.\d+)*)",
            r"Table\s+(\d+(?:\.\d+)*)",
            r"Figure\s+(\d+(?:\.\d+)*)",
            r"Schedule\s+([A-Z0-9\.]+)",
            r"\[\s*(\d+[A-Za-z]?)\s*\]",  # footnotes [12]
            r"#([A-Za-z][\w\-]{2,})",  # markdown anchors
        ]

    def __call__(self, state: State) -> State:
        root = self._resolve_root(state)
        seen_queries: set[str] = set()
        seen_uris = {c.uri for c in state.candidates}

        total_targets = 0
        total_queries = 0
        total_added = 0
        hop = 0

        while hop < max(1, self.max_hops):
            text_source = self._collect_text(state)
            targets = self._extract_targets(text_source)
            if self.heuristics_enabled:
                targets |= self._heuristic_targets(state)
            normalized = self._normalize_targets(targets)
            queries = []
            for q in normalized:
                key = q.lower()
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                queries.append(q)
                if len(queries) >= self.max_targets:
                    break

            total_targets += len(targets)
            total_queries += len(queries)

            if not queries:
                break

            new_cands = self._search_queries(queries, state, root, hop)
            new_cands = self._dedup_new(new_cands, seen_uris)

            if new_cands:
                total_added += len(new_cands)
                state.candidates = dedup_candidates(state.candidates + new_cands)
                seen_uris.update(c.uri for c in new_cands)
            elif self.prune:
                break

            hop += 1

            if not new_cands:
                continue
            if self.prune and hop >= self.max_hops:
                break

        state.log(
            "follow_refs",
            targets=total_targets,
            queries=total_queries,
            added=total_added,
            hops=hop,
            mode=self.mode,
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

    def _extract_targets(self, text: str) -> set[str]:
        found: set[str] = set()
        for pat in self.patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                token = m.group(0).strip()
                found.add(self._normalize_token(token))
        return found

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

    def _resolve_root(self, state: State) -> Path:
        if self.root:
            return self.root

        paths: list[Path] = []
        for c in state.candidates:
            p = self._uri_to_path(c.uri)
            if p:
                paths.append(p.parent)
        for e in state.evidence:
            p = self._uri_to_path(getattr(e, "uri", None))
            if p:
                paths.append(p.parent)
        if not paths:
            return Path(".").resolve()
        try:
            common = os.path.commonpath([str(p) for p in paths])
            return Path(common)
        except Exception:
            return paths[0]

    def _search_queries(
        self, queries: list[str], state: State, root: Path, hop: int
    ) -> list[Candidate]:
        if not root.exists():
            return []
        grep = Grep(dir=str(root))
        results: list[Candidate] = []
        for q in queries:
            orig_q = state.query
            q_state = State(
                query=orig_q.__class__(
                    text=q,
                    lang=getattr(orig_q, "lang", "auto"),
                    meta=dict(orig_q.meta),
                )
            )
            s2 = grep(q_state)
            for c in s2.candidates:
                c.score += self.per_target_boost
                c.meta.setdefault("ref_hit", []).append(q)
                c.meta["ref_hop"] = hop
            results.extend(s2.candidates)
        return results

    def _dedup_new(
        self, cands: list[Candidate], seen_uris: set[str]
    ) -> list[Candidate]:
        fresh: list[Candidate] = []
        seen = set(seen_uris)
        for c in cands:
            if c.uri in seen:
                continue
            seen.add(c.uri)
            fresh.append(c)
        return fresh

    def _heuristic_targets(self, state: State) -> set[str]:
        targets: set[str] = set()
        for c in state.candidates:
            if c.snippet:
                targets.update(self._pathish_tokens(c.snippet))
            path = self._uri_to_path(c.uri)
            if path:
                tokens = {path.stem, path.name, path.parent.name}
                for t in tokens:
                    norm = self._normalize_token(t)
                    if len(norm) > 3:
                        targets.add(norm)
        return targets

    def _pathish_tokens(self, text: str) -> set[str]:
        tokens = set()
        for m in re.finditer(
            r"(?:(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(?:py|js|ts|md|txt|rst|ini|ya?ml|json))",
            text,
        ):
            token = self._normalize_token(m.group(0))
            if len(token) > 3:
                tokens.add(token)
        return tokens

    def _uri_to_path(self, uri: str | None) -> Path | None:
        if not uri:
            return None
        prefix = "file://"
        if uri.startswith(prefix):
            return Path(uri[len(prefix) :]).resolve()
        return Path(uri).resolve()
