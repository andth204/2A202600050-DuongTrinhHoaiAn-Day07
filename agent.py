"""
Real Knowledge Base Agent — tool-use loop với OpenAI function calling.

Khác với run_agent.py (hardcode retrieve → prompt → answer):
  - LLM TỰ quyết định khi nào search, search bao nhiêu lần, với query gì.
  - Dùng OpenAI function calling: `search_knowledge_base` là tool.
  - Có system prompt rõ ràng, conversation memory multi-turn.
  - Agent loop: LLM → [tool call?] → execute tool → trả kết quả → LLM tiếp → ...

Usage:
    python agent.py                    # FixedSizeChunker, top_k=5
    python agent.py --strategy recursive --k 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILES = [
    ("python_intro_and_variables.txt",         "data/python_intro_and_variables.txt",         "intro_variables"),
    ("python_conditionals_loops_functions.txt", "data/python_conditionals_loops_functions.txt", "control_flow"),
    ("python_dictionaries_sets_list_tuples.txt","data/python_dictionaries_sets_list_tuples.txt","collections"),
    ("python_input_output.txt",                "data/python_input_output.txt",                "input_output"),
    ("python_error_exception.txt",             "data/python_error_exception.txt",             "exceptions"),
    ("python_module.txt",                      "data/python_module.txt",                      "modules"),
]

SYSTEM_PROMPT = """You are a precise Python documentation assistant.
You have access to a `search_knowledge_base` tool that retrieves relevant chunks \
from a curated set of Python tutorial documents covering: variables, control flow, \
data structures, input/output, exceptions, and modules.

Guidelines:
- Always search before answering factual questions about Python.
- You may call the tool multiple times with different queries if the first result is insufficient.
- Cite the source file for every claim (e.g. "According to python_module.txt, ...").
- If the retrieved context does not contain enough information, say so clearly — \
  do NOT fabricate answers.
- Keep answers concise and structured. Use code examples from the retrieved chunks when helpful."""

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the Python documentation knowledge base and return the most relevant text chunks. "
            "Call this tool whenever you need factual information to answer the user's question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, e.g. 'how does range() work in for loops'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (1–10). Default 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# ── Embedding & Index ─────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def build_index(chunker, embed_fn) -> list[tuple[str, str, list[float]]]:
    """Returns list of (source_file, chunk_text, embedding)."""
    index: list[tuple[str, str, list[float]]] = []
    for fname, fpath, _topic in DATA_FILES:
        text = Path(fpath).read_text(encoding="utf-8")
        for chunk in chunker.chunk(text):
            index.append((fname, chunk, embed_fn(chunk)))
    return index


def search(query: str, index, embed_fn, top_k: int = 5) -> str:
    """Execute retrieval and format results as a string for the LLM."""
    q_emb = embed_fn(query)
    ranked = sorted(index, key=lambda x: _cosine(q_emb, x[2]), reverse=True)[:top_k]

    lines = [f"Retrieved {len(ranked)} chunks for query: \"{query}\"\n"]
    for i, (fname, chunk, emb) in enumerate(ranked, 1):
        score = round(_cosine(q_emb, emb), 4)
        preview = chunk.replace("\n", " ")
        lines.append(f"[{i}] source={fname}  score={score}\n{preview}\n")
    return "\n".join(lines)


# ── Agent Loop ────────────────────────────────────────────────────────────────

def run_agent_turn(
    question: str,
    history: list[dict],
    index,
    embed_fn,
    openai_client,
    default_top_k: int,
    model: str = "gpt-4o-mini",
) -> str:
    """
    One user turn: append question → agentic loop → return final answer.
    History is modified in-place so multi-turn conversation is preserved.
    """
    history.append({"role": "user", "content": question})

    max_iterations = 6  # safety cap on tool calls per turn
    for iteration in range(max_iterations):
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=[TOOL_DEFINITION],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Convert to plain dict for history
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        history.append(msg_dict)

        # No tool calls → final answer
        if not msg.tool_calls or finish_reason == "stop":
            return msg.content or "(no response)"

        # Execute each tool call and push results back
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            top_k = int(args.get("top_k", default_top_k))
            query = args["query"]

            print(f"\n  [tool] search_knowledge_base(query={query!r}, top_k={top_k})")
            result_text = search(query, index, embed_fn, top_k=top_k)

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    return "(agent reached max iterations without a final answer)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Real Knowledge Base Agent (tool-use loop)")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "recursive"], default="fixed")
    parser.add_argument("--k", type=int, default=5, help="Default top_k for retrieval")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    # Embedder (local)
    print("Loading embedder (all-MiniLM-L6-v2)...", flush=True)
    sys.path.insert(0, ".")
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=".sentence-transformers")
    embed_fn = lambda text: _st_model.encode(text, normalize_embeddings=True).tolist()

    # Chunker
    from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
    chunker = {"fixed": FixedSizeChunker(chunk_size=500, overlap=50), "sentence": SentenceChunker(max_sentences_per_chunk=3), "recursive": RecursiveChunker(chunk_size=500)}[args.strategy]
    strategy_name = type(chunker).__name__

    # Index
    print(f"Indexing with {strategy_name}...", flush=True)
    index = build_index(chunker, embed_fn)
    print(f"Index ready: {len(index)} chunks\n")

    # OpenAI client
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    # Conversation history (shared across turns for multi-turn memory)
    history: list[dict] = []

    print("=" * 64)
    print("  Python Knowledge Base Agent  (tool-use / multi-turn)")
    print(f"  LLM      : {args.model}")
    print(f"  Embedder : all-MiniLM-L6-v2 (local)")
    print(f"  Strategy : {strategy_name}  |  chunks={len(index)}")
    print("  Type 'exit' to quit, 'reset' to clear conversation history.")
    print("=" * 64)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break
        if question.lower() == "reset":
            history.clear()
            print("[conversation history cleared]")
            continue

        answer = run_agent_turn(question, history, index, embed_fn, client, args.k, args.model)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
