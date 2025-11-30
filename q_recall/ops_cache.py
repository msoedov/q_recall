import hashlib
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .core import Candidate, Evidence, State
from .ops_agent import Op


@dataclass
class _Entry:
    signature: tuple[int, ...]
    text: str
    uri: str | None
    ts: float


class FingerprintCache(Op):
    """
    Lightweight local cache that remembers fragments via content fingerprints.

    - No embeddings/vector DB: uses rolling shingles + min-hash style signature.
    - If a fragment (candidate/evidence) is similar to cached content, reuse it
      instead of re-reading from disk or sending it downstream again.
    """

    def __init__(
        self,
        ttl: float | str | None = "session",
        match: str = "similar_blocks",
        shingle_size: int = 5,
        signature_size: int = 32,
        min_similarity: float | None = None,
        sample_chars: int = 4000,
        max_entries: int = 4096,
    ):
        self.ttl_seconds = None if ttl in (None, "session") else float(ttl)
        self.match = match
        self.shingle_size = max(2, shingle_size)
        self.signature_size = max(4, signature_size)
        self.min_similarity = (
            0.72
            if (min_similarity is None and match == "similar_blocks")
            else min_similarity
        )
        if self.min_similarity is None:
            self.min_similarity = 0.95
        self.sample_chars = max(200, sample_chars)
        self.max_entries = max(64, max_entries)
        self._entries: list[_Entry] = []
        self._by_uri: dict[str, list[_Entry]] = {}
        self.name = "FingerprintCache"

    def forward(self, state: State) -> State:
        now = time.time()
        pruned = self._prune(now)

        hits = 0
        misses = 0
        reused = 0
        evidence_dropped = 0

        new_candidates: list[Candidate] = []
        for cand in state.candidates:
            snippet, _ = self._resolve_candidate_text(cand)
            if snippet is None:
                new_candidates.append(cand)
                continue

            signature = self._fingerprint(snippet)
            if not signature:
                new_candidates.append(cand)
                continue

            match, sim = self._best_match(signature)
            if match and sim >= self.min_similarity:
                hits += 1
                if cand.snippet is None:
                    cand.snippet = match.text
                    reused += 1
                cand.meta.setdefault("fingerprint_cache", {}).update(
                    {"hit": True, "similarity": sim, "source": match.uri}
                )
                self._refresh(match, now)
            else:
                misses += 1
                cand.meta.setdefault("fingerprint_cache", {})["hit"] = False
                self._remember(signature, snippet, cand.uri, now)

            new_candidates.append(cand)

        state.candidates = new_candidates

        new_evidence: list[Evidence] = []
        for ev in state.evidence:
            text = (ev.text or "")[: self.sample_chars]
            if not text:
                new_evidence.append(ev)
                continue

            signature = self._fingerprint(text)
            if not signature:
                new_evidence.append(ev)
                continue

            match, sim = self._best_match(signature)
            if match and sim >= self.min_similarity:
                evidence_dropped += 1
                ev.meta.setdefault("fingerprint_cache", {}).update(
                    {"hit": True, "similarity": sim, "source": match.uri}
                )
                self._refresh(match, now)
                continue

            self._remember(signature, text, ev.uri, now)
            new_evidence.append(ev)

        state.evidence = new_evidence

        state.log(
            self.name,
            hits=hits,
            misses=misses,
            reused=reused,
            evidence_dropped=evidence_dropped,
            cache=len(self._entries),
            pruned=pruned,
            mode=self.match,
        )
        return state

    # ---------- helpers ----------
    def _resolve_candidate_text(self, cand: Candidate) -> tuple[str | None, bool]:
        if cand.snippet:
            return cand.snippet[: self.sample_chars], False

        cached = self._by_uri.get(cand.uri or "")
        if cached:
            return cached[-1].text, True

        text = self._read_candidate(cand)
        return (text[: self.sample_chars] if text else None), False

    def _read_candidate(self, cand: Candidate) -> str | None:
        if not cand.uri or not cand.uri.startswith("file://"):
            return None
        try:
            path = Path(cand.uri.replace("file://", ""))
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        line = cand.meta.get("line") if isinstance(cand.meta, dict) else None
        if line:
            lines = content.splitlines()
            idx = max(0, int(line) - 1)
            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)
            return "\n".join(lines[start:end])
        return content

    def _hash_shingles(self, shingles: Iterable[str]) -> list[int]:
        vals: list[int] = []
        for sh in shingles:
            digest = hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest()
            vals.append(int.from_bytes(digest, "big", signed=False))
        vals.sort()
        return vals[: self.signature_size]

    def _fingerprint(self, text: str) -> tuple[int, ...]:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return ()
        shingles: list[str] = []
        for i in range(len(tokens) - self.shingle_size + 1):
            window = tokens[i : i + self.shingle_size]
            shingles.append(" ".join(window))
        if not shingles:
            shingles = [" ".join(tokens)]
        hashed = self._hash_shingles(shingles)
        return tuple(hashed)

    def _best_match(self, signature: tuple[int, ...]) -> tuple[_Entry | None, float]:
        best: _Entry | None = None
        best_sim = 0.0
        sig_set = set(signature)
        for entry in self._entries:
            sim = self._jaccard(sig_set, entry.signature)
            if sim > best_sim:
                best_sim = sim
                best = entry
        return best, best_sim

    def _remember(
        self, signature: tuple[int, ...], text: str, uri: str | None, ts: float
    ):
        entry = _Entry(signature=signature, text=text, uri=uri, ts=ts)
        self._entries.append(entry)
        if uri:
            self._by_uri.setdefault(uri, []).append(entry)
        if len(self._entries) > self.max_entries:
            overflow = len(self._entries) - self.max_entries
            for _ in range(overflow):
                oldest = self._entries.pop(0)
                self._drop_uri_link(oldest)

    def _refresh(self, entry: _Entry, ts: float):
        entry.ts = ts

    def _prune(self, now: float) -> int:
        if self.ttl_seconds is None:
            return 0
        cutoff = now - self.ttl_seconds
        keep: list[_Entry] = []
        pruned = 0
        for e in self._entries:
            if e.ts >= cutoff:
                keep.append(e)
            else:
                pruned += 1
                self._drop_uri_link(e)
        self._entries = keep
        return pruned

    def _drop_uri_link(self, entry: _Entry):
        if not entry.uri:
            return
        bucket = self._by_uri.get(entry.uri, [])
        try:
            bucket.remove(entry)
        except ValueError:
            pass
        if not bucket:
            self._by_uri.pop(entry.uri, None)

    def _jaccard(self, a: set[int], b_sig: tuple[int, ...]) -> float:
        b = set(b_sig)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union
