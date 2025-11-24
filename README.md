# q_recall
**Post-RAG era Search Framework — Context-Rich, Index-Free**

`q_recall` is a lightweight ~~vitamin supplement~~, Keras-like agentic search framework built for the **post-RAG era**.
It combines **LLM-driven reasoning** with, **Zero indexing**, **direct file search (grep + glob)**, **reference following**, and **context enrichment** — allowing agents to *read and reason end-to-end*, without embeddings or indexing. This eliminates vector databases, chunkers, rerankers, and indexing entirely—replacing them with direct, filesystem-native search pipelines that read, traverse, and reason end-to-end.

> _"Retrieval was for context-poor models. Agentic search is for context-rich intelligence."_

---

## Bottom Line Up Front

It’s the first Keras-like agentic search stack where you assemble live search pipelines using Stack, Branch, Loop, SelfHeal, and StagnationGuard—letting the agent read your actual files, not an index of them.

### Strategic thesis
* Retrieval Augmented Generation (RAG) was built for context-poor LLMs that couldn’t read.
* Context-rich LLMs don’t need index gymnastics—they need real data, live traversal, and reasoning flow.

q_recall is that framework.

### For modern LLMs, RAG is failing in predictable ways

* Too Much Infrastructure
  - Chunking → embedding → indexing → filtering → reranking → deduping → summarizing → re-summarizing.
  - Six steps to answer a simple question.

* Indexes drift instantly
  - Code and docs change hourly.
  - Indexes get stale in minutes.
  - Syncing is brittle and expensive.

* Chunking destroys context
  - Chunkers cannot understand:
    - file boundaries
    - semantic sections
    - imports
    - references
    - loops
    - cross-links
    - diagrams
    - schema relationships

  - Your agent gets a “micro-view” of your repo.

* Agents must follow references, not search snippets
  - RAG can only return Top-K chunks.
  - Agents need to:
    - expand context dynamically
    - follow references (“See Note 12”)
    - detect when evidence is weak
    - retry with a wider plan
    - loop until convergence

* RAG’s entire stack is now overkill for GPT-4.1 / Claude 3.7+
  - New models can ingest 100K–200K tokens.
  - They don’t need chunking—they need curation and flow control.


In the new context-abundant world (2M+ tokens), we no longer need heavy RAG pipelines.
`q_recall` adopts the Claude Code philosophy — **no vector DB, no chunking, no reranking** — just smart agents that **navigate, reason, and follow references** across live files.

### Key Ideas
- **Zero-index, live filesystem search** — instant availability of new docs
- **Composable pipelines** — build agentic stacks like `keras.Sequential`
- **Branching, looping, planning** — for true multi-step, investigative agents
- **Traceable execution** — every step is logged into `State.trace`
- **LLM-ready hooks** — plug in real LLMs for extraction or answering

### The Real-World Problem It Solves

Traditional RAG stacks are overkill for many developer-, repo-, and folder-scale problems. Here are some key issues:

* **Indexing overhead**
  * Chunk → embed → index → filter → rerank → sync — just to answer simple questions.
  * Every schema/prompt/embedding change forces a re-ingest.
* **Stale knowledge**
  * Code and docs change fast; indexes drift immediately.
  * You want “I just saved this file — ask the agent about it now.”
* **Top-K tunnel vision**
  * Classic retrieval returns n snippets and stops.
  * Agents must follow references, jump files, widen search, and loop until they understand.
* **Opaque behavior**
  * It’s unclear why specific chunks were picked.
  * Debugging relevance feels like guesswork.

q_recall fixes this by:

* Searching the real filesystem directly (no index, no DB).
* Making control flow explicit: stacks, branches, loops, self-healing.
* Logging every step into `State.trace`, so you can see exactly what happened.


## The Real-World Problem It Solves
You want your agent or assistant to answer questions like:
- “What is our authentication flow in this monorepo?”
- “Find every place we handle GDPR deletion requests across backend + frontend + infra”
- “Explain how leasing accounting works in these 2024 SEC filings and follow all Note references”
- “Why is the test failing in CI only on Python 3.12?”

Current solutions force you to:
- Pre-index everything → stale the moment a file changes
- Chunk + embed → lose context, references, structure
- Pay for Pinecone/Qdrant → can’t run offline, costs explode
- Use naive grep → no reasoning, no ranking, no reference following

**q_recall gives you all four**: instant, reasoning-aware, reference-following, self-correcting search over raw files — entirely local, entirely free.

## Technology Comparison Table

| Feature                        | q_recall          | Traditional RAG (Chroma/LanceDB) | Hybrid (LangChain + Pinecone) | Naive grep/ripgrep | Tree-sitter + AST search |
|--------------------------------|-------------------|----------------------------------|-------------------------------|--------------------|--------------------------|
| Indexing required              | Never             | Yes (minutes–hours)              | Yes                           | No                 | No                       |
| Works instantly on new files    | Immediate         | Stale until re-index             | Stale                         | Yes                | Yes                      |
| Handles references (“See Note 7”)| Built-in          | No                               | Rarely                        | No                 | No                       |
| Self-healing / query reformulation | Yes            | No                               | Partial                       | No                 | No                       |
| Cost                           | Free              | Free–$$$                         | $50–$1000+/mo                 | Free               | Free                     |
| Latency (100k files)           | ~200–600 ms       | 800 ms–3 s                       | 1–4 s                         | ~80 ms (no answer) | Varies                   |
| Context size limit             | Unlimited (adaptive concat) | 10–100 chunks                | 10–100 chunks                 | Unlimited (dumb)   | Unlimited (dumb)         |
| LLM reasoning integration      | First-class       | Bolt-on                          | Bolt-on                       | None               | None                     |
| Offline / private              | Fully             | (embedded)                       | Cloud                         | Yes                | Yes                      |

Closest competitors are just tools. **q_recall is the full agentic loop.**

## Closest Market Equivalents
1. **Ripgrep + manual LLM prompting** – Fast but dumb
2. **MemGPT / Infinite Context hacks** – Still page from vector DBs
3. **Aider / Cursor / Continue.dev** – Great but closed, no composability
4. **Sourcegraph Cody local** – Still builds indexes, slow on huge repos

**q_recall position**: The open, composable, self-healing brain everyone should be using.

## What Makes q_recall Unique – The “Triple Unlock”
1. **Zero-index + live filesystem** = always correct, instant availability
2. **Agentic control flow** (`SelfHeal`, `Loop`, `StagnationGuard`, `Branch`) = robust even with weak queries
3. **Full reference following + context enrichment** = reads like a human, not like a retriever

**No existing solution has all three.**

### Use Cases

| Use Case                         | Current Pain                              | q_recall Impact                                      |
|----------------------------------|-------------------------------------------|------------------------------------------------------|
| Private codebase assistant        | Stale indexes, chunk blindness            | Always-up-to-date answers with full file context     |
| SEC/financial document analysis  | References break chunking, notes missed   | Automatically follows “See Note 12”, builds 160k+ token context |
| Legal contract review            | Cross-document references, definitions    | ReferenceFollower traverses entire folder hierarchy |
| Autonomous research agent        | Gets stuck on bad queries                 | SelfHeal + Planner rewrites until evidence found     |
| Local RAG over 10 GB+ of PDFs    | Embedding cost & time prohibitive         | Zero cost, streams relevant sections directly        |

## 🔧 Mental Model (read this first)

`q_recall` is built around two core abstractions:

### **1. `State`**

Each operation receives and returns a `State` object, which holds:

``` python
class State:
    query: str                # user query
    candidates: list          # found matches (file paths + snippets)
    evidence: str | None      # concatenated readable context
    answer: str | None        # final synthesized answer
    trace: list[TraceEvent]   # detailed execution log
```

Ops mutate fields intentionally and log their steps in `trace`.

### **2. `Op` (operation)**

Every operator is a function:

    State → State

Examples: `Grep`, `Glob`, `Ranking`, `ContextEnricher`, `ComposeAnswer`.

### **3. `Stack`, `Branch`, and control flow**

-   `Stack` --- sequential pipeline (like `keras.Sequential`)
-   `Branch` --- parallel search paths that merge
-   `SelfHeal` --- retries, fallback ops, post-conditions
-   `StagnationGuard` --- detects stalls and widens search
-   `Loop` --- repetition until convergence

------------------------------------------------------------------------

---

## Quick Start

### Install
```bash
git clone https://github.com/msoedov/q_recall.git
cd q_recall
pip install -e .
```

### Run an Example
```bash
python examples/basic.py
```

---

### Minimal Pipeline

A true minimal example: search → build context → answer.

``` python
import q_recall as qr

mem = qr.Stack(
    qr.Grep(dir="data"),
    qr.ContextEnricher(max_tokens=1000),
    qr.Concat(max_window_size=10_000),
    qr.ComposeAnswer(),
)

state = mem("Describe post rag pipeline")
print(state.answer)
```

Tip: `Grep` respects `state.query.meta["search_terms"]`, so prepend `WidenSearchTerms()` (or `LLMSearchTermExtractor`) to seed synonyms and section markers automatically.

### Budget guard (time + tokens)
Keep long-running examples in check with `WithBudget`. It short-circuits when either wall-clock or a rough token estimate is exhausted and logs usage into `state.budget`.

```python
budgeted = qr.Stack(
        qr.MultilingualNormalizer(),
        qr.WidenSearchTerms(),
        qr.WithBudget(seconds=2.5, tokens=80_000),
        qr.Grep(dir="data"),
        qr.Concat(max_window_size=8_000),
        qr.ComposeAnswer(),
    )

state = budgeted("Find the onboarding steps")
print(state.budget)  # {'tokens': 80000, 'tokens_spent': 1462, 'seconds': 2.5, ...}
```

## Self-Healing Search Pipelines

`SelfHeal` wraps any op with retries, fallback, and post-conditions.

``` python
safe_grep = qr.SelfHeal(
    op=qr.Grep(dir="data"),
    fallback=qr.Grep(dir="data", case_insensitive=True),
    post_condition=qr.has_candidates,
)

pipeline = qr.Stack(
    safe_grep,
    qr.Ranking(max_candidates=10),
    qr.ContextEnricher(max_tokens=2000),
    qr.Concat(max_window_size=20_000),
    qr.ComposeAnswer(prompt="Provide a concise, evidence-based answer."),
)

state = pipeline("Explain spec-driven development")
print(state.answer)
```

This pattern is the backbone for building robust agents without
overengineering.


## Self-Healing Search Agent

`q_recall` supports agentic recovery behaviors via `Planner` + `Loop`.
A self-healing pipeline detects when search results are missing, weak, or irrelevant — and automatically expands, reformulates, or redirects the query until useful evidence is found.

```python
mem0 = qr.Stack(
    qr.MultilingualNormalizer(),
    qr.WidenSearchTerms(),
    qr.SelfHeal(
        op=qr.Stack(qr.Grep(dir="../data"), qr.Ranking(max_candidates=10)),
        fallback=PlanB,
        post_condition=lambda s: qr.has_candidates(s, 1),
        on_weak=qr.WidenSearchTerms(
            extra=[
                "summary",
                "abstract",
                "description",
                "characteristic",
                "pattern",
                "example",
                "illustration",
            ]
        ),
        retries=1,
        backoff=0.25,
    ),
    qr.Deduplicate(),
    qr.SelfHeal(
        op=qr.ContextEnricher(max_tokens=1000),
        fallback=qr.ContextEnricher(max_tokens=3000),
        post_condition=lambda s: qr.has_candidates(s, 1),
    ),
    qr.SelfHeal(
        op=qr.AdaptiveConcat(max_window_size=10_000),
        fallback=qr.AdaptiveConcat(max_window_size=6_000),
        post_condition=lambda s: qr.has_evidence(s, min_chars=600),
    ),
    qr.StagnationGuard(
        min_gain=1, on_stall=qr.WidenSearchTerms(extra=["summary", "abstract"])
    ),
    qr.SelfHeal(
        op=qr.ComposeAnswer(prompt="provide a concise answer based on the context"),
        fallback=qr.ComposeAnswer(prompt="provide a concise answer with direct quotes"),
        post_condition=lambda s: s.answer is not None and len(s.answer) > 200,
    ),
)


if __name__ == "__main__":
    state = mem0("what is spec-driven development?")
    print(state.answer)
    for ev in state.trace:
        print(ev.op, ev.payload)
```


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
| `WithBudget` | Stops when wall-clock or token budget is exhausted |
| `WidenSearchTerms` | Expands query metadata with related terms before searching |
| `ReferenceFollower` | Detects and follows references (e.g., “See Note 12”) |
| `Planner` | Expands or rephrases queries if nothing found |

---

## Architecture Overview
```
User Query
   ↓
MultilingualNormalizer → TermExtractor (optional LLM call)
   ↓
┌───────────────────────┐
   │   Branch / Parallel     │
   ▼                       ▼
Grep(dir="repo")    Glob(pattern="*.py")
   ↓                       ↓
   └─────────Merge──────────┘
         ↓
      Ranking → Deduplicate
         ↓
   ContextEnricher (snippets → full sections/files)
         ↓
   ReferenceFollower (recursive, budget-aware)
         ↓
    AdaptiveConcat (max_tokens=2_000_000)
         ↓
    ComposeAnswer (or route to tool-calling agent)
         ↓
      Final Answer + Trace
```
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

##  Self-healing path

```

                 ┌───────────────────────────────────────────────────────┐
                 │   SelfHeal( … ) wraps any op: retries, fallback,      │
                 │   circuit breaker, post_condition + on_weak hooks     │
                 └───────────────────────────────────────────────────────┘

           ┌──────────────┐      ┌─────────────────┐
           │  SafeGrep()  │◄────►│ StagnationGuard │──► WidenSearchTerms()
           └──────────────┘      └─────────────────┘
                    │
              AdaptiveConcat()
                    │
             AutoHealPass(recovery=Stack(Grep→Concat), predicate=weak)

```


---

## Example Pipelines

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


------------------------------------------------------------------------

## When *not* to use q_recall

- You need <50ms latency over millions of documents.
- You need language-agnostic semantic similarity across huge corpora.
- Your data is already cleanly embedded in a vector database and you’re happy there.
`q_recall` is for **developer-scale agentic search over local files**.

-----------------------
---

## Design Philosophy

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


## Bottom Line
### What is q_recall, practically?
The search engine your 2025+ context-rich LLM actually deserves — zero infrastructure, infinitely fresh, self-healing, reference-aware, and fast enough to feel like magic.

### Is there anything like it?
Pieces exist (ripgrep = fast, MemGPT = long context, AgenticDB = cognitive), but nothing combines zero-index live search + agentic control flow + reference following in a composable Python framework.

### Should you switch?
Yes — if you ever felt RAG pain, if your data is private or changes frequently, if you want your agent to actually read instead of guess.

### The opportunity
Be the default post-RAG retrieval layer for the entire local LLM ecosystem.
The same way Keras democratized deep learning, q_recall democratizes agentic reasoning over private data.

## Planned Extensions

- Smarter `ReferenceFollower`
- Autonomous `Planner`
- Caching and budget control
- LLM-structured answering

---

## License

MIT License © 2025
