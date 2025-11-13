# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Callable
import time, re
from .core import State, Evidence
from .ops_agent import Op, Stack, Branch
from .ops_search import Grep, Glob
from .ops_rank import Concat


# ---------- Health model ----------
@dataclass
class Health:
    ok: bool = True
    reason: Optional[str] = None
    recovered: bool = False
    attempts: int = 0
    last_error: Optional[str] = None


# ---------- Predicates & refiners ----------
def has_candidates(state: State, min_n=1) -> bool:
    return len(state.candidates) >= min_n


def has_evidence(state: State, min_chars=500) -> bool:
    return any(len(e.text) >= min_chars for e in state.evidence if e.text)


def widen_search_terms(state: State, extra=None) -> State:
    extras = list(extra or ["overview", "summary", "introduction", "appendix"])
    state.query.meta.setdefault("search_terms", [])
    merged = list({*state.query.meta["search_terms"], *extras})
    state.query.meta["search_terms"] = merged
    state.log("auto_refine", added=extras)
    return state


# ---------- Healing wrappers ----------
class SelfHeal(Op):
    """
    Wrap any Op/Stack with:
      - retries + exponential backoff
      - fallback op
      - circuit breaker
      - post_condition and on_weak refinement hook
    """

    def __init__(
        self,
        op: Op,
        name: str = "SelfHeal",
        retries: int = 2,
        backoff: float = 0.0,
        fallback: Optional[Op] = None,
        post_condition: Optional[Callable[[State], bool]] = None,
        on_weak: Optional[Callable[[State], State]] = None,
        breaker_threshold: int = 4,
        breaker_cooldown: float = 0.0,
    ):
        self.op = op
        self.name = name
        self.retries = retries
        self.backoff = backoff
        self.fallback = fallback
        self.post_condition = post_condition
        self.on_weak = on_weak
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self._fail_count = 0
        self._breaker_until = 0.0

    def forward(self, state: State) -> State:
        now = time.time()
        if now < self._breaker_until:
            state.log(self.name, breaker="open", until=self._breaker_until)
            return state

        health = Health()
        last_exc = None

        # try main op with retries
        for attempt in range(self.retries + 1):
            health.attempts = attempt + 1
            try:
                # shallow copy for isolation
                out = self.op(State(**vars(state)))
                if self.post_condition and not self.post_condition(out):
                    raise ValueError("PostConditionFailed")
                state = out
                health.ok = True
                self._fail_count = 0
                break
            except Exception as e:
                health.ok = False
                last_exc = e
                health.last_error = f"{type(e).__name__}: {e}"
                state.log(self.name, error=health.last_error, attempt=attempt + 1)
                time.sleep(self.backoff * (2**attempt))

        # fallback if failed
        if not health.ok and self.fallback is not None:
            try:
                state.log(self.name, fallback=type(self.fallback).__name__)
                out = self.fallback(State(**vars(state)))
                if self.post_condition and not self.post_condition(out):
                    raise ValueError("FallbackPostConditionFailed")
                state = out
                health.ok = True
                health.recovered = True
                self._fail_count = 0
            except Exception as e:
                self._fail_count += 1
                state.log(self.name, fallback_error=f"{type(e).__name__}: {e}")
        elif not health.ok:
            self._fail_count += 1

        # circuit breaker
        if self._fail_count >= self.breaker_threshold:
            self._breaker_until = time.time() + self.breaker_cooldown
            state.log(self.name, breaker="open_set", fail_count=self._fail_count)

        # refine weak results if allowed
        if (
            health.ok
            and self.on_weak
            and self.post_condition
            and not self.post_condition(state)
        ):
            state = self.on_weak(state)
            state.log(self.name, refined=True)

        state.log(
            self.name,
            ok=health.ok,
            attempts=health.attempts,
            recovered=health.recovered,
            fail_count=self._fail_count,
        )
        return state


# ---------- Robust ops ----------
class SafeGrep(Grep):
    def forward(self, state: State) -> State:
        try:
            return super().forward(state)
        except Exception as e:
            state.log("SafeGrep", error=str(e))
            return state


class AdaptiveConcat(Concat):
    """Concats with graceful degradation and a sliding window."""

    def forward(self, state: State) -> State:
        try:
            return super().forward(state)
        except Exception:
            text = ""
            for c in state.candidates:
                chunk = c.snippet
                if not chunk:
                    try:
                        from .db import db  # lazy import if you hold a global

                        chunk = db.read(c.uri)
                    except Exception:
                        continue
                piece = chunk[: min(10_000, len(chunk))]
                if len(text) + len(piece) > self.max:
                    break
                text += f"\n\n----- {c.uri}\n{piece}"
            state.evidence.append(Evidence(text=text))
            state.log("AdaptiveConcat", size=len(text), degraded=True)
            return state


class StagnationGuard(Op):
    """If nothing improves, trigger an exploration/refinement action."""

    def __init__(self, min_gain=1, on_stall: Optional[Callable[[State], State]] = None):
        self.min_gain = min_gain
        self.on_stall = on_stall or (lambda s: widen_search_terms(s))
        self.name = "StagnationGuard"

    def forward(self, state: State) -> State:
        before = (len(state.candidates), sum(len(e.text) for e in state.evidence))
        # This op is a checkpoint; improvement is measured externally in your loop/graph.
        after = (len(state.candidates), sum(len(e.text) for e in state.evidence))
        if (after[0] - before[0]) < self.min_gain and (
            after[1] - before[1]
        ) < self.min_gain:
            state = self.on_stall(state)
            state.log(self.name, action="refine")
        else:
            state.log(self.name, action="none")
        return state


class AutoHealPass(Op):
    """Trigger a compact recovery pipeline if the predicate says the result is weak."""

    def __init__(
        self, recovery: Op, predicate: Callable[[State], bool], label="AutoHealPass"
    ):
        self.recovery = recovery
        self.predicate = predicate
        self.name = label

    def forward(self, state: State) -> State:
        if self.predicate(state):
            state.log(self.name, trigger=True)
            return self.recovery(state)
        state.log(self.name, trigger=False)
        return state
