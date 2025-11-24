from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .core import Query, State
from .ops_agent import Op


@dataclass
class Case:
    query: str
    must_include: list[str] = field(default_factory=list)
    must_hit_files: list[str] = field(default_factory=list)
    name: str | None = None

    def evaluate(self, state: State) -> "CaseResult":
        failures: list[str] = []
        answer = (state.answer or "").lower()

        for term in self.must_include:
            if term.lower() not in answer:
                failures.append(f"answer missing '{term}'")

        if self.must_hit_files:
            uris = [c.uri for c in state.candidates if c.uri]
            for fname in self.must_hit_files:
                if not any(fname in uri for uri in uris):
                    failures.append(f"missing file hit '{fname}'")

        return CaseResult(
            case=self, passed=not failures, failures=failures, state=state
        )


@dataclass
class CaseResult:
    case: Case
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None
    state: State | None = None


class EvalSuite:
    def __init__(self, name: str, cases: Sequence[Case]):
        self.name = name
        self.cases = list(cases)

    def run(
        self,
        pipeline: Callable[[str], State] | Op,
        stop_on_fail: bool = False,
    ) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in self.cases:
            try:
                state = self._run_case(pipeline, case.query)
                result = case.evaluate(state)
            except Exception as e:
                result = CaseResult(case=case, passed=False, error=str(e))
            results.append(result)
            if stop_on_fail and not result.passed:
                break
        return results

    def report(self, results: Iterable[CaseResult]) -> dict:
        results = list(results)
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        print(f"{self.name}: {passed}/{total} passed")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            label = r.case.name or r.case.query
            print(f"- {status}: {label}")
            if r.error:
                print(f"    error: {r.error}")
            for fail in r.failures:
                print(f"    missing: {fail}")
        return {"passed": passed, "total": total}

    def _run_case(self, pipeline, query: str) -> State:
        if isinstance(pipeline, Op):
            return pipeline(query)

        if callable(pipeline):
            out = pipeline(query)
            if isinstance(out, State):
                return out

        # Allow users to pass in Ops or simple callables; fail loudly otherwise.
        raise ValueError("pipeline must be an Op or callable returning a State")
