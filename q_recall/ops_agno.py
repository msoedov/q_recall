import json
import os
import re
from pathlib import Path
from typing import Any

from .core import State
from .ops_agent import Op


class LLMFileNavigator(Op):
    """
    Use an LLM (via the Agno framework) to propose which repo paths to scan first.

    The operator captures a shallow tree of the repository and asks the LLM to
    pick the most relevant files/directories for the incoming query. Results are
    stored under `state.query.meta["path_hints"]` as a list of `{path, reason, exists}`.

    If Agno or a model is unavailable, the op falls back to a lightweight
    heuristic matcher that surfaces paths containing query tokens.
    """

    def __init__(
        self,
        dir: str,
        policy: str = "reasoned-selection",
        agent=None,
        agent_factory=None,
        model_id: str | None = None,
        max_paths: int = 8,
        tree_depth: int = 3,
        max_dirs: int = 120,
        max_files_per_dir: int = 6,
    ):
        self.root = Path(dir).expanduser().resolve()
        self.policy = policy
        self.agent = agent
        self.agent_factory = agent_factory
        self.model_id = model_id or os.environ.get("Q_RECALL_AGNO_MODEL", "gpt-4o-mini")
        self.max_paths = max_paths
        self.tree_depth = tree_depth
        self.max_dirs = max_dirs
        self.max_files_per_dir = max_files_per_dir
        self.name = "LLMFileNavigator"
        self._last_error: str | None = None

    def forward(self, state: State) -> State:
        tree = self._snapshot_tree()
        prompt = self._build_prompt(state.query.text, tree)

        picks, source = self._llm_select(prompt)
        if not picks:
            picks = self._heuristic_select(state.query.text, tree)
            source = source or "heuristic"

        state.query.meta["path_hints"] = picks
        state.log(
            self.name,
            policy=self.policy,
            paths=len(picks),
            source=source,
            error=self._last_error,
        )
        return state

    # ---------- LLM path selection ----------
    def _llm_select(self, prompt: str) -> tuple[list[dict[str, Any]], str | None]:
        agent = self._ensure_agent()
        if agent is None:
            return [], "missing_agent"

        try:
            raw = self._run_agent(agent, prompt)
            picks = self._parse_selection(raw)
            if picks:
                picks = self._normalize_paths(picks)
                return picks[: self.max_paths], "agno"
        except Exception as e:  # pragma: no cover - defensive
            self._last_error = f"{type(e).__name__}: {e}"
        return [], "agno_error"

    def _ensure_agent(self):
        if self.agent is not None:
            return self.agent
        if self.agent_factory:
            try:
                self.agent = self.agent_factory()
                return self.agent
            except Exception as e:  # pragma: no cover - defensive
                self._last_error = f"agent_factory_failed: {e}"
                return None

        try:
            from agno.agent import Agent  # type: ignore
            from agno.models.openai import OpenAIChatModel  # type: ignore
        except Exception as e:
            self._last_error = f"import_error: {e}"
            return None

        try:
            self.agent = Agent(model=OpenAIChatModel(id=self.model_id))
            return self.agent
        except Exception as e:  # pragma: no cover - defensive
            self._last_error = f"agent_init_error: {e}"
            return None

    def _run_agent(self, agent, prompt: str) -> str:
        if hasattr(agent, "run"):
            out = agent.run(prompt)
        elif callable(agent):
            out = agent(prompt)
        else:  # pragma: no cover - defensive
            raise TypeError("Agno Agent must be callable or expose .run()")

        if isinstance(out, str):
            return out
        if hasattr(out, "content"):
            return str(getattr(out, "content"))
        return str(out)

    def _parse_selection(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []

        cleaned = text.strip()
        json_blob = self._extract_json(cleaned)
        if json_blob:
            try:
                data = json.loads(json_blob)
                if isinstance(data, dict):
                    data = data.get("paths") or data.get("items") or []
                if isinstance(data, list):
                    return self._normalize_candidates(data)
            except Exception:
                pass

        # Fallback: parse bullet list lines like "- path :: reason"
        picks: list[dict[str, Any]] = []
        for line in cleaned.splitlines():
            if not line.strip().startswith("-"):
                continue
            line = line.lstrip("-").strip()
            path, reason = self._split_line(line)
            picks.append({"path": path, "reason": reason})
        return picks

    def _normalize_candidates(self, data: list[Any]) -> list[dict[str, Any]]:
        picks: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, str):
                picks.append({"path": item, "reason": "llm"})
            elif isinstance(item, dict):
                path = item.get("path") or item.get("file") or item.get("location")
                reason = item.get("reason") or item.get("why") or item.get("note") or ""
                if path:
                    picks.append({"path": path, "reason": str(reason)})
        return picks

    def _split_line(self, line: str) -> tuple[str, str]:
        if "::" in line:
            path, reason = line.split("::", 1)
        elif " - " in line:
            path, reason = line.split(" - ", 1)
        else:
            return line.strip(), ""
        return path.strip(), reason.strip()

    def _normalize_paths(self, picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen = set()
        for item in picks:
            raw = str(item.get("path") or "").strip()
            if not raw:
                continue
            rel = str(Path(raw).as_posix()).lstrip("./")
            abs_path = (self.root / rel).resolve()
            key = rel.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "path": rel,
                    "reason": item.get("reason") or "",
                    "exists": abs_path.exists(),
                }
            )
            if len(normalized) >= self.max_paths:
                break
        return normalized

    # ---------- Fallback heuristic ----------
    def _heuristic_select(self, query: str, tree: list[dict[str, Any]]):
        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_\-]{4,}", query)}
        picks: list[dict[str, Any]] = []
        for entry in tree:
            haystack = " ".join([entry["path"]] + entry["files"]).lower()
            for tok in tokens:
                if tok in haystack:
                    picks.append(
                        {
                            "path": entry["path"],
                            "reason": f"matched token '{tok}'",
                            "exists": True,
                        }
                    )
                    break
            if len(picks) >= self.max_paths:
                break
        if not picks and tree:
            picks.append(
                {"path": tree[0]["path"], "reason": "fallback root", "exists": True}
            )
        return picks

    # ---------- Repo snapshot & prompt ----------
    def _snapshot_tree(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self.root.exists():
            return entries

        for root, dirs, files in os.walk(self.root):
            rel_dir = Path(root).resolve().relative_to(self.root).as_posix() or "."
            depth = len(Path(rel_dir).parts)
            if depth > self.tree_depth:
                dirs[:] = []  # prune deeper traversal
                continue

            entries.append(
                {
                    "path": rel_dir,
                    "files": sorted(files)[: self.max_files_per_dir],
                }
            )
            if len(entries) >= self.max_dirs:
                break
        return entries

    def _build_prompt(self, query: str, tree: list[dict[str, Any]]) -> str:
        listing = "\n".join(
            f"- {entry['path']}: {', '.join(entry['files']) if entry['files'] else '(dir)'}"
            for entry in tree
        )
        policy_note = (
            "Prioritize folders that align semantically with the question; "
            "prefer auth/security flows, middleware, config, and user models when queries mention login or permissions."
        )
        return (
            f"You are an Agno agent helping to navigate a repository.\n"
            f"Policy: {self.policy} ({policy_note})\n"
            f"Question: {query}\n"
            "Repo outline (truncated):\n"
            f"{listing}\n"
            "Return a JSON array of objects with keys `path` and `reason`. "
            "Keep paths relative, prefer existing entries from the outline, "
            f"and limit to {self.max_paths} items."
        )

    def _extract_json(self, text: str) -> str | None:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return None
