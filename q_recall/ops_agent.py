from .core import State, dedup_candidates
from typing import List


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
        for op in self.ops:
            state = op(state)
        return state


class Branch(Op):
    def __init__(self, *branches, merge="concat"):
        self.branches = branches
        self.merge = merge
        self.name = "Branch"

    def forward(self, state: State) -> State:
        # shallow copy of State for each branch
        outs = []
        for b in self.branches:
            s = State(**vars(state))
            outs.append(b(s))
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
        for i in range(self.max_iters):
            prev = (len(state.evidence), len(state.candidates))
            state = self.body(state)
            if self.until(state, prev):
                break
        return state
