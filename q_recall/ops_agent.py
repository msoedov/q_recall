from .core import State, dedup_candidates, Query
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

    def __call__(self, state: State | str) -> State:
        match state:
            case State():
                return self.forward(state)
            case str():
                return self.forward(State(query=Query(text=state)))
            case _:
                raise ValueError(f"Expected State or str, got {type(state)}")


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
