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
        case_insensitive=True,
        respect_path_hints=True,
    ):
        self.root = Path(dir)
        self.file_glob = file_glob
        self.ignore = re.compile(ignore)
        self.context = context
        self.case_insensitive = case_insensitive
        self.respect_path_hints = respect_path_hints
        self.engine = self._detect_engine()
        self.name = "Grep"

    def __call__(self, state: State) -> State:
        terms = state.query.meta.get("search_terms") or [state.query.text]
        hits = []
        bases = self._resolve_hint_paths(state) or [self.root]
        unique_bases: list[Path] = []
        seen_paths = set()
        for b in bases:
            resolved = b.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            unique_bases.append(resolved)

        for term in terms:
            term = (term or "").strip()
            if not term:
                continue
            for base in unique_bases:
                try:
                    hits.extend(self._search_term(term, base))
                except Exception as e:
                    state.log(
                        "grep_error",
                        term=term,
                        base=str(base),
                        error=str(e),
                    )
        state.candidates += hits
        state.log(
            "grep",
            matches=len(hits),
            terms=len(terms),
            engine=self.engine,
            bases=len(unique_bases),
        )
        return state

    # ---------- internal helpers ----------
    def _detect_engine(self) -> str:
        import shutil

        if shutil.which("rg"):
            return "rg"
        if shutil.which("grep"):
            return "grep"
        raise RuntimeError("Neither ripgrep (rg) nor grep is available on PATH.")

    def _resolve_hint_paths(self, state: State) -> list[Path]:
        if not self.respect_path_hints:
            return []
        hints = state.query.meta.get("path_hints") or []
        paths: list[Path] = []
        seen = set()
        for item in hints:
            raw_path = ""
            if isinstance(item, dict):
                raw_path = str(item.get("path") or "").strip()
            elif isinstance(item, str):
                raw_path = item.strip()
            if not raw_path:
                continue
            rel = Path(raw_path)
            base = (self.root / rel).resolve()
            key = str(base)
            if key in seen:
                continue
            seen.add(key)
            if base.exists():
                paths.append(base)
        return paths

    def _search_term(self, term: str, base: Path) -> list[Candidate]:
        match self.engine:
            case "rg":
                lines = self._run_cmd(self._rg_command(term, base), cwd=base)
            case "grep":
                lines = self._run_cmd(self._grep_command(term, base), cwd=base)
            case _:
                lines = []

        cache: dict[str, list[str]] = {}
        hits: list[Candidate] = []

        for line in lines:
            path_str, line_no = self._parse_hit(line)
            if path_str is None or line_no is None:
                continue

            candidate_path = Path(path_str)
            full_path = (
                candidate_path
                if candidate_path.is_absolute()
                else base.joinpath(candidate_path).resolve()
            )
            if self.ignore.search(str(full_path)):
                continue

            text_lines = cache.get(str(full_path))
            if text_lines is None:
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    text_lines = content.splitlines()
                except Exception:
                    text_lines = []
                cache[str(full_path)] = text_lines

            snippet = self._make_snippet(text_lines, line_no - 1)
            hits.append(
                Candidate(
                    uri=f"file://{full_path}",
                    score=1.0,
                    snippet=snippet,
                    meta={"line": line_no, "term": term},
                )
            )

        return hits

    def _rg_command(self, term: str, root: Path) -> list[str]:
        cmd = [
            "rg",
            "--no-heading",
            "--line-number",
            "--color",
            "never",
            "--fixed-strings",
        ]
        if self.case_insensitive:
            cmd.append("-i")
        if self.file_glob and self.file_glob != "**/*":
            cmd += ["--glob", self.file_glob]
        cmd += [term, str(root)]
        return cmd

    def _grep_command(self, term: str, root: Path) -> list[str]:
        cmd = [
            "grep",
            "-R",
            "-n",
            "-H",
            "-F",
            "-I",
            "--binary-files=without-match",
        ]
        if self.case_insensitive:
            cmd.append("-i")
        if self.file_glob and self.file_glob != "**/*":
            cmd += ["--include", self.file_glob]
        cmd += [term, str(root)]
        return cmd

    def _run_cmd(self, cmd: list[str], cwd: Path | None = None) -> list[str]:
        import subprocess

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd or self.root),
        )
        # grep returns 1 when no matches are found
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        return proc.stdout.splitlines()

    def _parse_hit(self, line: str) -> tuple[str | None, int | None]:
        parts = line.split(":", 2)
        if len(parts) < 2:
            return None, None
        path_str, line_no = parts[0], parts[1]
        try:
            return path_str, int(line_no)
        except ValueError:
            return None, None

    def _make_snippet(self, lines: list[str], idx: int) -> str:
        if not lines:
            return ""
        start = max(0, idx - self.context)
        end = min(len(lines), idx + self.context + 1)
        return "\n".join(lines[start:end])


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
