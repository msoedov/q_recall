"""Minimal demo that produces FlowInspector trace HTML."""
from pathlib import Path

from q_recall import Query, State


def main():
    state = State(query=Query("demo trace"))
    state.log("grep", files=3, uri="docs/readme.md")
    state.log("rank", kept=2, tokens_spent=120)
    state.log("answer", chars=180, confidence=0.72)

    out_path = Path(__file__).parent / "trace.html"
    out = state.visualize(out_path)
    print(f"Trace written to {out}")
    print("Open it in a browser to explore filters, timeline, and payloads.")


if __name__ == "__main__":
    main()
