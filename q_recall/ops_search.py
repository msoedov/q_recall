import re
from pathlib import Path

from .core import Candidate, State


class Grep:
    def __init__(
        self,
        dir: str,
        file_glob="**/*",
        ignore=r"\.(png|jpg|gif|pdf|bin|exe|zip)$",
        context=2,
    ):
        self.root = Path(dir)
        self.file_glob = file_glob
        self.ignore = re.compile(ignore)
        self.context = context
        self.name = "Grep"

    def __call__(self, state: State) -> State:
        terms = state.query.meta.get("search_terms") or [state.query.text]
        hits = []
        for p in self.root.glob(self.file_glob):
            if not p.is_file() or self.ignore.search(str(p)):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for term in terms:
                if not term:
                    continue
                q_re = re.escape(term)
                for m in re.finditer(q_re, txt, flags=re.IGNORECASE):
                    start = max(0, m.start() - 400)
                    end = min(len(txt), m.end() + 400)
                    snippet = txt[start:end]
                    line_no = txt.count("\n", 0, m.start()) + 1
                    hits.append(
                        Candidate(
                            uri=f"file://{p}",
                            score=1.0,
                            snippet=snippet,
                            meta={"line": line_no, "term": term},
                        )
                    )
        state.candidates += hits
        state.log("grep", matches=len(hits), terms=len(terms))
        return state


class Glob:
    def __init__(self, dir: str, pattern="**/*"):
        self.root = Path(dir)
        self.pattern = pattern
        self.name = "Glob"

    def __call__(self, state: State) -> State:
        files = [f for f in self.root.glob(self.pattern) if f.is_file()]
        cands = []
        for f in files:
            try:
                mtime = f.stat().st_mtime
            except Exception:
                mtime = 0.0
            cands.append(
                Candidate(
                    uri=f"file://{f}", score=0.2 + (mtime / 1e12), meta={"mtime": mtime}
                )
            )
        state.candidates += cands
        state.log("glob", files=len(cands))
        return state
