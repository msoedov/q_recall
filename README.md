# q_recall
**Post-RAG era Search Framework — Context-Rich, Index-Free**

`q_recall` is a lightweight, Keras-like agentic search framework built for the **post-RAG era**.
It combines **LLM-driven reasoning** with, **Zero indexing**, **direct file search (grep + glob)**, **reference following**, and **context enrichment** — allowing agents to *read and reason end-to-end*, without embeddings or indexing.

> _"Retrieval was for context-poor models. Agentic search is for context-rich intelligence."_

---

## 🚀 Why q_recall?

In the new context-abundant world (2M+ tokens), we no longer need heavy RAG pipelines.
`q_recall` adopts the Claude Code philosophy — **no vector DB, no chunking, no reranking** — just smart agents that **navigate, reason, and follow references** across live files.

### Key Ideas
- **Zero-index, live filesystem search** — instant availability of new docs
- **Composable pipelines** — build agentic stacks like `keras.Sequential`
- **Branching, looping, planning** — for true multi-step, investigative agents
- **Traceable execution** — every step is logged
- **LLM-ready hooks** — plug in real LLMs for extraction or answering

---

## 🧩 Quick Start

### Install
```bash
git clone https://github.com/yourname/q_recall.git
cd q_recall
pip install -e .
```

### Run an Example
```bash
python examples/basic.py
```

---

## 🧠 Minimal Example

```python
import q_recall as qr

db = qr.ParadigmDB()
db.register_fs("data", "./data")

mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.Branch(
        qr.Stack(qr.Grep(dir="data"), qr.Ranking(max_candidates=10)),
        qr.Stack(qr.Glob(dir="data"), qr.Ranking(max_candidates=5)),
    ),
    qr.Deduplicate(),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer()
)

state = mem0(qr.State(query=qr.Query(text="Describe a counter-dependent personality")))
print(state.answer)
```

## 🛠 Self-Healing Search Agent

`q_recall` supports agentic recovery behaviors via `Planner` + `Loop`.
A self-healing pipeline detects when search results are missing, weak, or irrelevant — and automatically expands, reformulates, or redirects the query until useful evidence is found.


---

## 🧱 Core Concepts

### The Building Blocks

| Component | Purpose |
|------------|----------|
| `Grep` | Fast, regex-based content search through files |
| `Glob` | File discovery by name pattern |
| `Ranking` | Simple scoring and filtering |
| `ContextEnricher` | Expands snippets into readable context |
| `Concat` | Combines multiple candidates into one evidence block |
| `ComposeAnswer` | Generates final answer (LLM hook ready) |
| `Stack` | Sequential composition (like `keras.Sequential`) |
| `Branch` | Parallel paths (like `keras.Functional`) |
| `Loop` | Iterative refinement until convergence |
| `ReferenceFollower` | Detects and follows references (e.g., “See Note 12”) |
| `Planner` | Expands or rephrases queries if nothing found |

---

## 🧮 Architecture Overview

```
           ┌──────────────┐
           │   Query()    │
           └──────┬───────┘
                  │
           ┌──────▼───────┐
           │ Term Extract │ ← optional LLM
           └──────┬───────┘
        ┌─────────┼──────────┐
        ▼                    ▼
   ┌──────────┐         ┌──────────┐
   │  Grep()  │         │  Glob()  │
   └──────────┘         └──────────┘
        ▼                    ▼
        └───────► Merge ►────┘
                 │
             Ranking()
                 │
         ContextEnricher()
                 │
              Concat()
                 │
          ComposeAnswer()
```

---

## ⚙️ Example Pipelines

### 1. Basic Search (English example)
```python
mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.Grep(dir="data"),
    qr.Ranking(max_candidates=10),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer()
)
```

### 2. SEC Filings Agent (Reference Following)
```python
lease_agent = qr.Stack(
    qr.Grep(dir="sec"),
    qr.Ranking(max_candidates=20),
    qr.ContextEnricher(max_tokens=2000),
    qr.Concat(max_window_size=80_000),
    qr.ReferenceFollower(dir="sec"),
    qr.Ranking(max_candidates=30, keyword_boost=["lease", "Note", "Item 7"]),
    qr.Concat(max_window_size=160_000),
    qr.ComposeAnswer(prompt="Compute final lease obligations with adjustments:")
)
```

### 3. Branching Code Search
```python
code_search = qr.Stack(
    qr.Branch(
        qr.Stack(qr.Grep(dir="repo"), qr.Ranking(max_candidates=25)),
        qr.Stack(qr.Glob(dir="repo", pattern="src/**/*.*"), qr.Ranking(max_candidates=10)),
    ),
    qr.Deduplicate(),
    qr.ContextEnricher(max_tokens=1500),
    qr.Concat(max_window_size=50_000),
    qr.ComposeAnswer(prompt="Summarize implementation details with file paths.")
)
```

---

## 📦 Project Structure

```
q_recall/
├── q_recall/
│   ├── __init__.py
│   ├── core.py
│   ├── ops_search.py
│   ├── ops_rank.py
│   ├── ops_agent.py
│   ├── answer.py
│   ├── db.py
│   └── utils.py
├── examples/
│   ├── basic.py
│   └── sec_lease.py
├── README.md
└── pyproject.toml
```

---

## 🔬 Design Philosophy

- **Direct reasoning over raw data**
  Agents read entire files and follow references — no artificial fragmentation.

- **Composable like Keras**
  `Stack` pipelines mirror deep learning model assembly: simple, declarative, inspectable.

- **Transparent execution**
  Every op logs its behavior into the `State.trace`, so you can audit the reasoning chain.

- **LLM Optional**
  Everything runs offline. Plug in real LLM calls only for term extraction or final synthesis.

---

## 🧩 Extending the Framework

```python
from q_recall.core import State
from q_recall.ops_agent import Op

class MyFilter(Op):
    def forward(self, state: State) -> State:
        state.candidates = [c for c in state.candidates if "important" in c.snippet.lower()]
        state.log("myfilter", kept=len(state.candidates))
        return state
```

---

## 🧰 Planned Extensions

- Smarter `ReferenceFollower`
- Autonomous `Planner`
- Caching and budget control
- LLM-structured answering

---

## 🧑‍💻 License

MIT License © 2025
