import json
from datetime import datetime, timezone
from pathlib import Path

from .core import State
from .ops_agent import Op


class PersistHistory(Op):
    """
    Append a compact record of the current State to a JSONL log.

    Keeps queries/answers, candidates, evidence, trace, and budget so runs can be
    inspected later. Large text fields are clipped to avoid runaway files.
    """

    def __init__(
        self,
        path: str | Path = ".q_recall/history.jsonl",
        include_trace: bool = True,
        include_candidates: bool = True,
        include_evidence: bool = True,
        max_text: int = 20_000,
    ):
        self.path = Path(path)
        self.include_trace = include_trace
        self.include_candidates = include_candidates
        self.include_evidence = include_evidence
        self.max_text = max_text
        self.name = "PersistHistory"

    def forward(self, state: State) -> State:
        record = self._serialize(state)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=repr)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            state.log(self.name, ok=True, path=str(self.path))
        except Exception as e:
            state.log(self.name, ok=False, path=str(self.path), error=str(e))
        return state

    # ---------- helpers ----------
    def _serialize(self, state: State) -> dict:
        record: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": {
                "text": state.query.text,
                "lang": state.query.lang,
                "meta": self._safe(state.query.meta),
            },
            "answer": state.answer,
            "budget": self._safe(state.budget),
        }

        if self.include_candidates:
            record["candidates"] = [
                {
                    "uri": c.uri,
                    "score": c.score,
                    "snippet": self._clip(c.snippet),
                    "meta": self._safe(c.meta),
                }
                for c in state.candidates
            ]

        if self.include_evidence:
            record["evidence"] = [
                {"uri": e.uri, "text": self._clip(e.text), "meta": self._safe(e.meta)}
                for e in state.evidence
            ]

        if self.include_trace:
            record["trace"] = [
                {"op": ev.op, "payload": self._safe(ev.payload), "t": ev.t}
                for ev in state.trace
            ]

        return record

    def _clip(self, text: str | None) -> str | None:
        if text is None or self.max_text is None:
            return text
        if len(text) <= self.max_text:
            return text
        return text[: self.max_text]

    def _safe(self, obj):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._safe(v) for v in obj]
        return repr(obj)
