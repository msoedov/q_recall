import time

from .core import Query, State, dedup_candidates
from .utils import timer


class Op:
    name: str = "Op"

    def __call__(self, state: State) -> State:
        return self.forward(state)

    def forward(self, state: State) -> State:
        raise NotImplementedError


class Stack(Op):
    def __init__(self, *ops, name="Stack"):
        self.ops = list(ops)
        self.name = name

    def __call__(self, state: State | str) -> State:
        match state:
            case State():
                return self.forward(state)
            case str():
                return self.forward(State(query=Query(text=state)))
            case _:
                raise ValueError(f"Expected State or str, got {type(state)}")

    def forward(self, state: State) -> State:
        with timer() as t_all:
            for op in self.ops:
                with timer() as t:
                    state = op(state)
                state.log(
                    "time", name=getattr(op, "name", op.__class__.__name__), seconds=t()
                )
        state.log("time_overall", name=self.name, seconds=t_all())
        return state


class WithBudget(Op):
    """
    Guard an op/stack with simple budgets.

    - `seconds`: wall-clock budget for the wrapped op and everything before it.
    - `tokens`: rough token budget based on text we accumulate in State.

    Can be used in two ways:
    - As a wrapper: `WithBudget(op, tokens=..., seconds=...)`
    - As a pass-through guard in a Stack: `WithBudget()`
    """

    def __init__(
        self,
        op: Op | None = None,
        tokens: int | None = None,
        seconds: float | None = None,
    ):
        self.op = op
        self.tokens = tokens
        self.seconds = seconds
        self.name = "WithBudget"

    def __call__(self, state: State | str) -> State:
        match state:
            case State():
                return self.forward(state)
            case str():
                return self.forward(State(query=Query(text=state)))
            case _:
                raise ValueError(f"Expected State or str, got {type(state)}")

    def forward(self, state: State) -> State:
        started_at = state.budget.setdefault("_started_at", time.perf_counter())
        elapsed = time.perf_counter() - started_at

        limit_seconds = (
            self.seconds
            if self.seconds is not None
            else state.budget.get("seconds", None)
        )
        spent_tokens = state.budget.get("tokens_spent", 0)
        limit_tokens = (
            self.tokens if self.tokens is not None else state.budget.get("tokens", None)
        )

        # If a constraint is already exhausted, short-circuit
        if limit_seconds is not None and elapsed >= limit_seconds:
            state.log(
                "budget",
                exhausted=True,
                constraint="seconds",
                elapsed=elapsed,
                limit=limit_seconds,
            )
            return state

        if limit_tokens is not None and spent_tokens >= limit_tokens:
            state.log(
                "budget",
                exhausted=True,
                constraint="tokens",
                spent=spent_tokens,
                limit=limit_tokens,
            )
            return state

        before_tokens = _estimate_tokens(state)

        if self.op is not None:
            with timer() as t:
                state = self.op(state)
            step_seconds = t()
        else:
            # No wrapped op: act as a pass-through guard
            step_seconds = 0.0

        after_tokens = _estimate_tokens(state)
        spent_tokens += max(0, after_tokens - before_tokens)

        state.budget["tokens_spent"] = spent_tokens
        if limit_tokens is not None:
            state.budget["tokens"] = limit_tokens
        if limit_seconds is not None:
            state.budget["seconds"] = limit_seconds

        state.log(
            "budget",
            exhausted=False,
            constraint="ok",
            seconds=step_seconds,
            elapsed=time.perf_counter() - started_at,
            tokens_spent=spent_tokens,
            limit_tokens=limit_tokens,
            limit_seconds=limit_seconds,
        )
        return state


class Gate(Op):
    """
    Assertion/guardrail for pipelines.

    - `predicate`: callable that returns truthy to continue.
    - `on_fail`: optional callable/Op that runs when predicate is false.
    - `raise_on_fail`: raise to trigger outer SelfHeal/WithBudget guards.
    """

    def __init__(self, predicate, on_fail=None, raise_on_fail=True, name="Gate"):
        self.predicate = predicate
        self.on_fail = on_fail
        self.raise_on_fail = raise_on_fail
        self.name = name

    def __call__(self, state: State | str) -> State:
        match state:
            case State():
                return self.forward(state)
            case str():
                return self.forward(State(query=Query(text=state)))
            case _:
                raise ValueError(f"Expected State or str, got {type(state)}")

    def forward(self, state: State) -> State:
        try:
            ok = bool(self.predicate(state))
        except Exception as e:
            state.log("gate", ok=False, error=str(e))
            if self.raise_on_fail:
                raise
            return state

        if ok:
            state.log("gate", ok=True, action="pass")
            return state

        state.log("gate", ok=False, action="fail")
        if self.on_fail:
            try:
                recovered_state = self.on_fail(state)
                state.log(
                    "gate_recover",
                    action=getattr(
                        self.on_fail,
                        "name",
                        getattr(self.on_fail, "__name__", type(self.on_fail).__name__),
                    ),
                )
                return recovered_state
            except Exception as e:
                state.log("gate_recover_error", error=str(e))
                if self.raise_on_fail:
                    raise
                return state

        if self.raise_on_fail:
            raise ValueError("Gate predicate failed")
        return state


class Branch(Op):
    def __init__(self, *branches, merge="concat"):
        self.branches = branches
        self.merge = merge
        self.name = "Branch"

    def forward(self, state: State) -> State:
        outs = []
        for b in self.branches:
            s = State(**vars(state))
            with timer() as t:
                out_s = b(s)
            state.log(
                "time",
                name=f"Branch::{getattr(b, 'name', b.__class__.__name__)}",
                seconds=t(),
            )
            outs.append(out_s)
        if self.merge == "concat":
            state.candidates = dedup_candidates(sum([o.candidates for o in outs], []))
            state.evidence = sum([o.evidence for o in outs], [])
        elif self.merge == "best":
            state = max(outs, key=lambda s: sum(c.score for c in s.candidates))
        return state


class QueryRouter(Op):
    """
    Route a query/state to the first matching sub-pipeline.

    Each route is a tuple of (name, predicate, op), evaluated in order. The first
    predicate that returns truthy wins. If nothing matches, `default` is used
    when provided; otherwise, the original state is returned (or an error is
    raised if `require_match=True`).
    """

    def __init__(self, *routes, default=None, require_match=False, name="QueryRouter"):
        self.routes = [self._normalize_route(r) for r in routes]
        self.default = default
        self.require_match = require_match
        self.name = name

    def __call__(self, state: State | str) -> State:
        match state:
            case State():
                return self.forward(state)
            case str():
                return self.forward(State(query=Query(text=state)))
            case _:
                raise ValueError(f"Expected State or str, got {type(state)}")

    def forward(self, state: State) -> State:
        for label, predicate, op in self.routes:
            try:
                matched = bool(predicate(state))
            except Exception as e:
                state.log(
                    self.name,
                    route=label,
                    matched=False,
                    error=str(e),
                    stage="predicate",
                )
                continue

            if matched:
                state.log(self.name, route=label, matched=True, stage="match")
                return self._run(op, state, label)

        if self.default is not None:
            state.log(self.name, route="default", matched=True, stage="default")
            return self._run(self.default, state, "default")

        state.log(self.name, route=None, matched=False, stage="nomatch")
        if self.require_match:
            raise ValueError("No route matched query")
        return state

    def _run(self, op, state: State, label: str) -> State:
        result = self._invoke(op, state)
        result.log(self.name, route=label, matched=True, stage="done")
        return result

    def _invoke(self, op, state: State) -> State:
        if isinstance(op, Op):
            return op(state)
        if callable(op):
            out = op(state)
            if isinstance(out, State):
                return out
        raise ValueError("Route target must be an Op or callable returning a State")

    def _normalize_route(self, route):
        try:
            name, predicate, op = route
        except Exception as e:
            raise ValueError("Each route must be a (name, predicate, op) tuple") from e
        if not callable(predicate):
            raise TypeError(f"Route '{name}' predicate must be callable")
        if not callable(op):
            raise TypeError(f"Route '{name}' op must be callable")
        return name, predicate, op


class Loop(Op):
    def __init__(self, body: Op, until, max_iters=8):
        self.body = body
        self.until = until
        self.max_iters = max_iters
        self.name = "Loop"

    def forward(self, state: State) -> State:
        with timer() as t_all:
            for i in range(self.max_iters):
                prev = (len(state.evidence), len(state.candidates))
                with timer() as t:
                    state = self.body(state)
                state.log(
                    "time",
                    name=f"LoopBody::{getattr(self.body, 'name', self.body.__class__.__name__)}",
                    seconds=t(),
                )
                if self.until(state, prev):
                    break
        state.log("time_overall", name=self.name, seconds=t_all())
        return state


def _estimate_tokens(state: State) -> int:
    """Rough token estimate based on accumulated text fields."""
    chars = len(getattr(state.query, "text", "") or "")
    chars += sum(len(c.snippet or "") for c in getattr(state, "candidates", []))
    chars += sum(len(e.text or "") for e in getattr(state, "evidence", []))
    if getattr(state, "answer", None):
        chars += len(state.answer)
    return max(0, chars // 4)
