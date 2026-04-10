"""
FastAPI backend for the Knowledge Base Agent demo UI.

Usage:
    python app.py
Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILES = [
    ("python_intro_and_variables.txt",          "data/python_intro_and_variables.txt",          "intro_variables"),
    ("python_conditionals_loops_functions.txt",  "data/python_conditionals_loops_functions.txt",  "control_flow"),
    ("python_dictionaries_sets_list_tuples.txt", "data/python_dictionaries_sets_list_tuples.txt", "collections"),
    ("python_input_output.txt",                  "data/python_input_output.txt",                  "input_output"),
    ("python_error_exception.txt",               "data/python_error_exception.txt",               "exceptions"),
    ("python_module.txt",                        "data/python_module.txt",                        "modules"),
]

SYSTEM_PROMPT = """You are a precise Python documentation assistant.
You have access to a `search_knowledge_base` tool that retrieves relevant chunks \
from a curated set of Python tutorial documents covering: variables, control flow, \
data structures, input/output, exceptions, and modules.

Guidelines:
- Always search before answering factual questions about Python.
- You may call the tool multiple times with different queries if the first result is insufficient.
- Cite the source file for every claim (e.g. "According to python_module.txt, ...").
- If the retrieved context does not contain enough information, say so clearly.
- Keep answers concise and structured. Use code examples from the retrieved chunks when helpful."""

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the Python documentation knowledge base and return the most relevant text chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    },
}

# ── Startup: build embedder + index ──────────────────────────────────────────

from openai import OpenAI
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of texts — one API call for the whole batch."""
    resp = _openai.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

def embed_one(text: str) -> list[float]:
    return _openai.embeddings.create(model=_EMBED_MODEL, input=text).data[0].embedding

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

print(f"Building index with {_EMBED_MODEL} (FixedSizeChunker 500/50)...", flush=True)
from src.chunking import FixedSizeChunker
_chunker = FixedSizeChunker(chunk_size=500, overlap=50)

# Collect all chunks first, then batch-embed (minimize API round-trips)
_all_chunks: list[tuple[str, str]] = []  # (filename, chunk_text)
for fname, fpath, _topic in DATA_FILES:
    text = Path(fpath).read_text(encoding="utf-8")
    for chunk in _chunker.chunk(text):
        _all_chunks.append((fname, chunk))

BATCH = 100  # text-embedding-3-small accepts up to 2048 inputs per call
INDEX: list[tuple[str, str, list[float]]] = []
for i in range(0, len(_all_chunks), BATCH):
    batch = _all_chunks[i : i + BATCH]
    embeddings = embed([c for _, c in batch])
    for (fname, chunk), emb in zip(batch, embeddings):
        INDEX.append((fname, chunk, emb))
    print(f"  Indexed {min(i + BATCH, len(_all_chunks))}/{len(_all_chunks)} chunks...", flush=True)

print(f"Index ready: {len(INDEX)} chunks\n", flush=True)

# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_knowledge_base(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Returns (text_for_llm, chunks_for_ui)."""
    q_emb = embed_one(query)
    ranked = sorted(INDEX, key=lambda x: _cosine(q_emb, x[2]), reverse=True)[:top_k]
    chunks_ui = []
    lines = [f'Retrieved {len(ranked)} chunks for query: "{query}"\n']
    for i, (fname, chunk, cemb) in enumerate(ranked, 1):
        score = round(_cosine(q_emb, cemb), 4)
        preview = chunk.replace("\n", " ")
        lines.append(f"[{i}] source={fname}  score={score}\n{preview}\n")
        chunks_ui.append({"rank": i, "source": fname, "score": score, "text": chunk})
    return "\n".join(lines), chunks_ui

# ── SSE Agent loop ────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"

async def run_agent_stream(question: str, history: list[dict]) -> AsyncGenerator[str, None]:
    history.append({"role": "user", "content": question})

    for iteration in range(6):
        # ── Non-streaming round: detect tool calls ──────────────────────────
        response = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=[TOOL_DEFINITION],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )
        msg = response.choices[0].message

        # Has tool calls → execute them, push results, loop again
        if msg.tool_calls:
            msg_dict: dict = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            }
            history.append(msg_dict)

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                query = args["query"]
                top_k = int(args.get("top_k", 5))

                yield sse("searching", {"query": query, "top_k": top_k})
                result_text, chunks_ui = search_knowledge_base(query, top_k)
                yield sse("chunks", {"query": query, "chunks": chunks_ui})
                history.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            continue  # next iteration → LLM sees tool results

        # ── No tool calls → stream the final answer token by token ──────────
        full_text = ""
        yield sse("answer_start", {})

        with _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_text += delta
                    yield sse("token", {"text": delta})

        history.append({"role": "assistant", "content": full_text})
        yield sse("answer_done", {})
        return

    yield sse("token", {"text": "(agent reached max iterations)"})
    yield sse("answer_done", {})

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI()

# Per-session conversation histories keyed by session_id
_histories: dict[str, list[dict]] = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ResetRequest(BaseModel):
    session_id: str

@app.post("/reset")
def reset(req: ResetRequest):
    _histories.pop(req.session_id, None)
    return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
    history = _histories.setdefault(req.session_id, [])
    return StreamingResponse(
        run_agent_stream(req.question, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Python KB Agent</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    padding: 14px 24px;
    background: #1a1d27;
    border-bottom: 1px solid #2d3148;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .logo { font-size: 1.3rem; font-weight: 700; color: #7c83ff; }
  header .sub  { font-size: 0.8rem; color: #64748b; }
  header .badge {
    margin-left: auto;
    background: #1e2235;
    border: 1px solid #2d3148;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #94a3b8;
  }

  .layout {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Chat panel ── */
  .chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2d3148;
  }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .msg {
    max-width: 82%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
    font-size: 0.92rem;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: #3b3f8c;
    border-bottom-right-radius: 4px;
  }
  .msg.agent {
    align-self: flex-start;
    background: #1e2235;
    border: 1px solid #2d3148;
    border-bottom-left-radius: 4px;
  }
  .msg.agent code {
    background: #0f1117;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.85em;
  }
  .msg.agent pre {
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 12px 14px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .msg.agent pre code { background: none; padding: 0; }

  .status-line {
    align-self: flex-start;
    font-size: 0.8rem;
    color: #7c83ff;
    padding: 4px 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .spinner {
    width: 12px; height: 12px;
    border: 2px solid #3b3f8c;
    border-top-color: #7c83ff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .input-row {
    padding: 16px 20px;
    border-top: 1px solid #2d3148;
    display: flex;
    gap: 10px;
    background: #1a1d27;
  }
  #query {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 8px;
    color: #e2e8f0;
    padding: 10px 14px;
    font-size: 0.92rem;
    resize: none;
    height: 44px;
    outline: none;
    transition: border-color 0.2s;
  }
  #query:focus { border-color: #7c83ff; }
  button {
    background: #7c83ff;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0 20px;
    font-size: 0.9rem;
    cursor: pointer;
    font-weight: 600;
    transition: background 0.2s;
  }
  button:hover { background: #6366f1; }
  button:disabled { background: #2d3148; color: #475569; cursor: not-allowed; }
  #reset-btn {
    background: transparent;
    border: 1px solid #2d3148;
    color: #64748b;
    padding: 0 14px;
    font-weight: 400;
  }
  #reset-btn:hover { border-color: #7c83ff; color: #7c83ff; background: transparent; }

  /* ── Retrieval panel ── */
  .retrieval-panel {
    width: 380px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .retrieval-panel h2 {
    padding: 14px 18px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #2d3148;
    background: #1a1d27;
  }
  #chunks-container {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .search-group { display: flex; flex-direction: column; gap: 8px; }
  .search-label {
    font-size: 0.75rem;
    color: #7c83ff;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .search-label span { color: #475569; font-weight: 400; }

  .chunk-card {
    background: #1e2235;
    border: 1px solid #2d3148;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .chunk-card:hover { border-color: #4f548c; }
  .chunk-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #151827;
    border-bottom: 1px solid #2d3148;
  }
  .chunk-rank {
    width: 22px; height: 22px;
    background: #2d3148;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: #94a3b8;
    flex-shrink: 0;
  }
  .chunk-source {
    font-size: 0.72rem;
    color: #94a3b8;
    font-family: monospace;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .score-bar-wrap { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .score-val {
    font-size: 0.72rem;
    font-weight: 700;
    font-family: monospace;
    min-width: 40px;
    text-align: right;
  }
  .score-bar {
    width: 48px; height: 6px;
    background: #2d3148;
    border-radius: 3px;
    overflow: hidden;
  }
  .score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #4f548c, #7c83ff);
    transition: width 0.3s;
  }
  .chunk-body {
    padding: 8px 12px;
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.5;
    display: none;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .chunk-card.expanded .chunk-body { display: block; }

  .empty-state {
    color: #475569;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 40px;
  }

  .cursor {
    display: inline-block;
    color: #7c83ff;
    animation: blink 0.8s step-end infinite;
  }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

  /* scrollbars */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div class="logo">🐍 Python KB Agent</div>
  <div class="sub">RAG · tool-use · text-embedding-3-small + gpt-4o-mini</div>
  <div class="badge" id="chunk-count">306 chunks</div>
</header>

<div class="layout">

  <!-- Chat -->
  <div class="chat-panel">
    <div id="messages">
      <div class="msg agent">Hi! Ask me anything about Python basics — variables, loops, data structures, exceptions, modules, and more.</div>
    </div>
    <div class="input-row">
      <textarea id="query" placeholder="Ask a Python question…" rows="1"></textarea>
      <button id="send-btn" onclick="sendMessage()">Send</button>
      <button id="reset-btn" onclick="resetSession()">Reset</button>
    </div>
  </div>

  <!-- Retrieval sidebar -->
  <div class="retrieval-panel">
    <h2>Retrieved Chunks</h2>
    <div id="chunks-container">
      <div class="empty-state">Chunks retrieved by the agent<br>will appear here with scores.</div>
    </div>
  </div>

</div>

<script>
const SESSION_ID = Math.random().toString(36).slice(2);
let busy = false;

const messagesEl = document.getElementById('messages');
const chunksEl   = document.getElementById('chunks-container');
const queryEl    = document.getElementById('query');
const sendBtn    = document.getElementById('send-btn');

// Auto-resize textarea
queryEl.addEventListener('input', () => {
  queryEl.style.height = '44px';
  queryEl.style.height = Math.min(queryEl.scrollHeight, 120) + 'px';
});
queryEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function scoreColor(s) {
  if (s >= 0.7) return '#4ade80';
  if (s >= 0.5) return '#facc15';
  return '#f87171';
}

function appendMsg(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (role === 'agent') {
    div.innerHTML = renderMarkdown(text);
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function renderMarkdown(text) {
  // minimal markdown: code blocks, inline code, bold
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_,lang,code) =>
      `<pre><code>${code.trimEnd()}</code></pre>`)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function addSearchGroup(query) {
  if (chunksEl.querySelector('.empty-state')) chunksEl.innerHTML = '';
  const group = document.createElement('div');
  group.className = 'search-group';
  group.dataset.query = query;
  group.innerHTML = `<div class="search-label">🔍 Search <span>"${query}"</span></div>`;
  chunksEl.appendChild(group);
  chunksEl.scrollTop = chunksEl.scrollHeight;
  return group;
}

function addChunkCard(group, chunk) {
  const pct = Math.round(Math.max(0, chunk.score) * 100);
  const color = scoreColor(chunk.score);
  const card = document.createElement('div');
  card.className = 'chunk-card';
  card.innerHTML = `
    <div class="chunk-header">
      <div class="chunk-rank">${chunk.rank}</div>
      <div class="chunk-source">${chunk.source}</div>
      <div class="score-bar-wrap">
        <div class="score-val" style="color:${color}">${chunk.score.toFixed(4)}</div>
        <div class="score-bar"><div class="score-fill" style="width:${pct}%;background:${color}88"></div></div>
      </div>
    </div>
    <div class="chunk-body">${chunk.text.replace(/</g,'&lt;')}</div>`;
  card.addEventListener('click', () => card.classList.toggle('expanded'));
  group.appendChild(card);
  chunksEl.scrollTop = chunksEl.scrollHeight;
}

function setStatus(text) {
  let el = document.getElementById('status-line');
  if (!el) {
    el = document.createElement('div');
    el.id = 'status-line';
    el.className = 'status-line';
    messagesEl.appendChild(el);
  }
  el.innerHTML = text
    ? `<div class="spinner"></div>${text}`
    : '';
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function clearStatus() {
  const el = document.getElementById('status-line');
  if (el) el.remove();
}

async function sendMessage() {
  const q = queryEl.value.trim();
  if (!q || busy) return;
  busy = true;
  sendBtn.disabled = true;
  queryEl.value = '';
  queryEl.style.height = '44px';

  appendMsg('user', q);
  setStatus('Thinking…');

  let agentDiv = null;
  let agentText = '';

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: SESSION_ID, question: q})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\n\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));

        if (ev.event === 'searching') {
          setStatus(`Searching: "${ev.query}"`);
          addSearchGroup(ev.query);
        }
        else if (ev.event === 'chunks') {
          const group = [...chunksEl.querySelectorAll('.search-group')]
            .find(g => g.dataset.query === ev.query) || addSearchGroup(ev.query);
          for (const chunk of ev.chunks) addChunkCard(group, chunk);
          setStatus('Generating answer…');
        }
        else if (ev.event === 'answer_start') {
          clearStatus();
          agentText = '';
          agentDiv = appendMsg('agent', '');
          agentDiv.innerHTML = '<span class="cursor">▋</span>';
        }
        else if (ev.event === 'token') {
          agentText += ev.text;
          agentDiv.innerHTML = renderMarkdown(agentText) + '<span class="cursor">▋</span>';
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        else if (ev.event === 'answer_done') {
          if (agentDiv) agentDiv.innerHTML = renderMarkdown(agentText);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
    }
  } catch(e) {
    clearStatus();
    appendMsg('agent', `Error: ${e.message}`);
  }

  busy = false;
  sendBtn.disabled = false;
  queryEl.focus();
}

async function resetSession() {
  if (busy) return;
  await fetch('/reset', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: SESSION_ID})
  });
  messagesEl.innerHTML = '<div class="msg agent">Conversation reset. Ask me anything about Python!</div>';
  chunksEl.innerHTML = '<div class="empty-state">Chunks retrieved by the agent<br>will appear here with scores.</div>';
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
