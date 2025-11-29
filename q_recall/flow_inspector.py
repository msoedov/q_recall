"""Utilities for turning a recorded trace into an interactive HTML viewer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import TraceEvent


def render_trace_html(trace: Iterable["TraceEvent"]) -> str:
    """Return standalone HTML containing the FlowInspector UI with embedded trace data."""
    trace_json = _trace_to_json(trace)
    return TRACE_HTML_TEMPLATE.replace("__TRACE_DATA__", trace_json)


def _trace_to_json(trace: Iterable["TraceEvent"]) -> str:
    events = []
    t0 = None
    for idx, ev in enumerate(trace):
        t0 = t0 if t0 is not None else ev.t
        events.append(
            {
                "id": idx,
                "op": ev.op,
                "t": ev.t,
                "delta": round(ev.t - t0, 3),
                "payload": {k: _jsonify(v) for k, v in ev.payload.items()},
            }
        )
    return json.dumps(events)


def _jsonify(value: Any) -> Any:
    """Convert common trace payload types into JSON-serializable values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {k: _jsonify(v) for k, v in value.__dict__.items()}
    return str(value)


TRACE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FlowInspector - Trace</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0b1223;
      --panel: #0f172a;
      --card: #111c32;
      --accent: #7ef29d;
      --text: #e4ecff;
      --muted: #92a3bf;
      --border: #1e2a44;
      --danger: #ff8a8a;
      --mono: "SFMono-Regular", "JetBrains Mono", "Menlo", monospace;
      --sans: "IBM Plex Sans", "Inter", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 10% 20%, rgba(126,242,157,0.08), transparent 30%),
                  radial-gradient(circle at 80% 0%, rgba(255,188,117,0.08), transparent 30%),
                  var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      padding: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(17,28,50,0.95), rgba(17,28,50,0.75));
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0.02em; }
    .pill {
      padding: 6px 10px;
      border-radius: 10px;
      background: rgba(126,242,157,0.14);
      color: var(--accent);
      border: 1px solid rgba(126,242,157,0.3);
      font-size: 12px;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-top: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    .filters {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .filters input, .filters select {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
    }
    .filters button {
      background: var(--accent);
      color: #052412;
      border: none;
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 12px 0;
    }
    .summary .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }
    .summary .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
    .summary .value { font-size: 18px; font-weight: 600; margin-top: 4px; }
    #timeline {
      margin-top: 10px;
      position: relative;
      height: 160px;
    }
    .timeline-track {
      position: absolute;
      inset: 0;
      border-radius: 10px;
      background: linear-gradient(90deg, rgba(126,242,157,0.08), rgba(126,242,157,0.02));
      border: 1px dashed rgba(146,163,191,0.35);
    }
    .event-dot {
      position: absolute;
      bottom: 10px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(126,242,157,0.1);
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.2s ease;
    }
    .event-dot:hover { transform: scale(1.2); box-shadow: 0 0 0 10px rgba(126,242,157,0.12); }
    .event-dot.selected { background: #fff; }
    .event-line {
      position: absolute;
      bottom: 20px;
      width: 2px;
      background: rgba(126,242,157,0.3);
      border-radius: 2px;
    }
    #event-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }
    .event-card {
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--card);
      cursor: pointer;
      transition: border 0.15s ease, transform 0.1s ease;
    }
    .event-card:hover { border-color: rgba(126,242,157,0.4); transform: translateY(-1px); }
    .event-card.active { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(126,242,157,0.12); }
    .event-card .meta { display: flex; gap: 8px; color: var(--muted); font-size: 12px; }
    .event-card .title { font-weight: 700; margin-bottom: 6px; }
    #details pre {
      background: #0c1427;
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 12px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 13px;
      color: #d7e3ff;
    }
    .hint { color: var(--muted); font-size: 13px; }
    @media (max-width: 960px) {
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="pill">FlowInspector</div>
      <h1>Trace playback</h1>
    </div>
    <div class="pill" id="range-pill"></div>
  </header>

  <div class="panel">
    <div class="filters">
      <select id="op-filter">
        <option value="">All ops</option>
      </select>
      <input id="search" type="search" placeholder="Search payload, uri, route, file..." />
      <button id="reset">Reset view</button>
    </div>
    <div class="summary">
      <div class="card"><div class="label">Events</div><div class="value" id="stat-events">0</div></div>
      <div class="card"><div class="label">Unique ops</div><div class="value" id="stat-ops">0</div></div>
      <div class="card"><div class="label">Duration</div><div class="value" id="stat-duration">-</div></div>
      <div class="card"><div class="label">Tokens spent</div><div class="value" id="stat-tokens">-</div></div>
    </div>
    <div id="timeline" class="panel">
      <div class="timeline-track"></div>
    </div>
  </div>

  <div class="layout">
    <div class="panel">
      <div class="hint">Events (click to inspect)</div>
      <div id="event-list"></div>
    </div>
    <div class="panel" id="details">
      <div class="hint">Payload</div>
      <pre id="payload">Select an event to inspect its payload.</pre>
    </div>
  </div>

  <script>
    const data = __TRACE_DATA__.map(ev => ({
      ...ev,
      iso: new Date(ev.t * 1000).toISOString(),
      payloadText: JSON.stringify(ev.payload, null, 2)
    }));
    const ops = Array.from(new Set(data.map(d => d.op)));

    const opFilter = document.getElementById("op-filter");
    const search = document.getElementById("search");
    const resetBtn = document.getElementById("reset");
    const listEl = document.getElementById("event-list");
    const payloadEl = document.getElementById("payload");
    const timelineEl = document.getElementById("timeline");

    ops.forEach(op => {
      const opt = document.createElement("option");
      opt.value = op;
      opt.textContent = op;
      opFilter.appendChild(opt);
    });

    let selectedId = data.length ? data[0].id : null;

    resetBtn.onclick = () => {
      opFilter.value = "";
      search.value = "";
      render();
    };

    search.oninput = debounce(render, 150);
    opFilter.onchange = render;

    function render() {
      const q = search.value.toLowerCase();
      const op = opFilter.value;
      const filtered = data.filter(ev => {
        const matchesOp = !op || ev.op === op;
        const matchesText = !q || ev.payloadText.toLowerCase().includes(q) || ev.op.toLowerCase().includes(q);
        return matchesOp && matchesText;
      });
      renderStats(filtered);
      renderTimeline(filtered);
      renderList(filtered);
    }

    function renderStats(events) {
      document.getElementById("stat-events").textContent = events.length;
      document.getElementById("stat-ops").textContent = new Set(events.map(e => e.op)).size;
      const duration = (events.at(-1)?.delta ?? 0).toFixed(2);
      document.getElementById("stat-duration").textContent = events.length ? duration + "s" : "-";
      const tokens = events.reduce((acc, ev) => acc + (ev.payload.tokens_spent || ev.payload.tokens || 0), 0);
      document.getElementById("stat-tokens").textContent = tokens ? tokens.toLocaleString() : "-";
      const start = data[0] ? new Date(data[0].t * 1000).toLocaleString() : "";
      const end = data.at(-1) ? new Date(data.at(-1).t * 1000).toLocaleString() : "";
      document.getElementById("range-pill").textContent = start && end ? `${start} -> ${end}` : "No trace";
    }

    function renderTimeline(events) {
      timelineEl.querySelectorAll(".event-dot, .event-line").forEach(el => el.remove());
      if (!events.length) return;
      const maxDelta = Math.max(...events.map(e => e.delta || 0), 0.001);
      events.forEach(ev => {
        const x = (ev.delta / maxDelta) * 100;
        const line = document.createElement("div");
        line.className = "event-line";
        line.style.height = `${14 + (ev.id % 5) * 6}px`;
        line.style.left = `calc(${x}% - 1px)`;
        timelineEl.appendChild(line);
        const dot = document.createElement("div");
        dot.className = "event-dot" + (ev.id === selectedId ? " selected" : "");
        dot.style.left = `calc(${x}% - 5px)`;
        dot.title = `${ev.op} (#${ev.id})`;
        dot.onclick = () => select(ev.id);
        timelineEl.appendChild(dot);
      });
    }

    function renderList(events) {
      listEl.innerHTML = "";
      if (!events.length) {
        listEl.innerHTML = '<div class="hint">No events match the current filters.</div>';
        payloadEl.textContent = "Select an event to inspect its payload.";
        return;
      }
      events.forEach(ev => {
        const card = document.createElement("div");
        card.className = "event-card" + (ev.id === selectedId ? " active" : "");
        card.onclick = () => select(ev.id);
        const meta = `<div class="meta"><span>#${ev.id}</span><span>${ev.delta.toFixed(3)}s</span><span>${ev.iso}</span></div>`;
        const title = `<div class="title">${ev.op}</div>`;
        const payload = highlight(ev.payload);
        card.innerHTML = title + meta + payload;
        listEl.appendChild(card);
      });
      if (selectedId === null && events.length) {
        select(events[0].id);
      } else if (!events.some(e => e.id === selectedId)) {
        select(events[0].id);
      } else {
        select(selectedId, false);
      }
    }

    function select(id, scroll = true) {
      selectedId = id;
      document.querySelectorAll(".event-card").forEach(card => {
        card.classList.toggle("active", card.textContent.includes(`#${id}`));
      });
      document.querySelectorAll(".event-dot").forEach(dot => {
        dot.classList.toggle("selected", dot.title.includes(`#${id}`));
      });
      const ev = data.find(e => e.id === id);
      if (ev) {
        payloadEl.textContent = ev.payloadText;
      }
      if (scroll) {
        const card = Array.from(document.querySelectorAll(".event-card")).find(c => c.textContent.includes(`#${id}`));
        card?.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }

    function highlight(payload) {
      const text = JSON.stringify(payload, null, 2);
      const escaped = text.replace(/[&<>]/g, ch => ({ "&":"&amp;","<":"&lt;",">":"&gt;" }[ch]));
      return `<pre>${escaped}</pre>`;
    }

    function debounce(fn, delay) {
      let t;
      return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
      };
    }

    render();
  </script>
</body>
</html>
"""
