from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time


@dataclass
class Query:
    text: str
    lang: str = "auto"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    uri: str
    score: float = 0.0
    snippet: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    text: str
    uri: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    op: str
    payload: Dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class State:
    query: Query
    candidates: List[Candidate] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    answer: Optional[str] = None
    budget: Dict[str, float] = field(default_factory=lambda: {"tokens": 2_000_000})
    trace: List[TraceEvent] = field(default_factory=list)

    def log(self, op, **payload):
        self.trace.append(TraceEvent(op, payload))


def dedup_candidates(cands: List[Candidate]) -> List[Candidate]:
    seen, out = set(), []
    for c in sorted(cands, key=lambda x: (-x.score, x.uri)):
        key = (c.uri, c.snippet)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
