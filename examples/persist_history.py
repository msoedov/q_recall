import os
from pathlib import Path

import q_recall as qr

# Persist every run into a JSONL log so you can inspect what happened later.
# Override the path with Q_RECALL_HISTORY to keep test runs isolated.
history_path = Path(os.environ.get("Q_RECALL_HISTORY", ".q_recall/history.jsonl"))

mem_with_history = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.WidenSearchTerms(),
    qr.Grep(dir="data"),
    qr.Ranking(max_candidates=8),
    qr.ContextEnricher(max_tokens=800),
    qr.Concat(max_window_size=8_000),
    qr.ComposeAnswer(prompt="Answer using only gathered context:"),
    qr.PersistHistory(path=history_path),
)


if __name__ == "__main__":
    state = mem_with_history("what is spec-driven development?")
    print(state.answer)
    print(f"History appended to {history_path.resolve()}")
