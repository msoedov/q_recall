import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Query:
    text: str
    lang: str = "auto"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    uri: str
    score: float = 0.0
    snippet: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    text: str
    uri: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    op: str
    payload: dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class State:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    answer: str | None = None
    budget: dict[str, float] = field(
        default_factory=lambda: {
            "tokens": 2_000_000,
            "tokens_spent": 0,
            "seconds": None,
        }
    )
    trace: list[TraceEvent] = field(default_factory=list)

    def log(self, op, **payload):
        self.trace.append(TraceEvent(op, payload))

    def explain_trace(self):
        for ev in self.trace:
            print(ev.op, ev.payload)

    def visualize(self, path: str | Path = "trace.html") -> Path:
        """Render the current trace into a standalone HTML file."""
        from .flow_inspector import render_trace_html

        path = Path(path)
        html = render_trace_html(self.trace)
        path.write_text(html, encoding="utf-8")
        return path


def dedup_candidates(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in sorted(cands, key=lambda x: (-x.score, x.uri)):
        key = (c.uri, c.snippet)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
