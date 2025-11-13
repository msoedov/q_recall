import time
from dataclasses import dataclass, field
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
    budget: dict[str, float] = field(default_factory=lambda: {"tokens": 2_000_000})
    trace: list[TraceEvent] = field(default_factory=list)

    def log(self, op, **payload):
        self.trace.append(TraceEvent(op, payload))

    def explain_trace(self):
        for ev in self.trace:
            print(ev.op, ev.payload)


def dedup_candidates(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in sorted(cands, key=lambda x: (-x.score, x.uri)):
        key = (c.uri, c.snippet)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
