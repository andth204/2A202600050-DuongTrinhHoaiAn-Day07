"""
Interactive retrieval: nhập query --> top-5 chunks liên quan nhất (cosine similarity).

Usage:
    python query.py                          # nhập query thủ công, dùng FixedSizeChunker
    python query.py --strategy sentence      # dùng SentenceChunker
    python query.py --strategy recursive     # dùng RecursiveChunker
    python query.py --k 3                    # trả về top-3 thay vì top-5
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

DATA_FILES = [
    ("python_intro_and_variables.txt",         "data/python_intro_and_variables.txt",         "intro_variables"),
    ("python_conditionals_loops_functions.txt", "data/python_conditionals_loops_functions.txt", "control_flow"),
    ("python_dictionaries_sets_list_tuples.txt","data/python_dictionaries_sets_list_tuples.txt","collections"),
    ("python_input_output.txt",                "data/python_input_output.txt",                "input_output"),
    ("python_error_exception.txt",             "data/python_error_exception.txt",             "exceptions"),
    ("python_module.txt",                      "data/python_module.txt",                      "modules"),
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def build_index(chunker, embed_fn) -> list[tuple[str, str, str, list[float]]]:
    """Returns list of (filename, topic, chunk_text, embedding)."""
    index = []
    for fname, fpath, topic in DATA_FILES:
        text = Path(fpath).read_text(encoding="utf-8")
        for chunk in chunker.chunk(text):
            index.append((fname, topic, chunk, embed_fn(chunk)))
    return index


def retrieve(query: str, index, embed_fn, top_k: int = 5):
    q_emb = embed_fn(query)
    ranked = sorted(index, key=lambda x: cosine(q_emb, x[3]), reverse=True)
    return [(fname, topic, chunk, round(cosine(q_emb, emb), 4)) for fname, topic, chunk, emb in ranked[:top_k]]


def print_results(results: list, query: str) -> None:
    print(f'\nQuery: "{query}"')
    print("=" * 72)
    for i, (fname, topic, chunk, score) in enumerate(results, 1):
        preview = chunk[:300].replace("\n", " ")
        if len(chunk) > 300:
            preview += "..."
        print(f"[{i}] score={score:.4f} | {fname} (topic: {topic})")
        print(f"    {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chunk retrieval")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "recursive"], default="fixed")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Load embedder
    print("Loading embedder (all-MiniLM-L6-v2)...", flush=True)
    sys.path.insert(0, ".")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=".sentence-transformers")

    def embed(text: str) -> list[float]:
        return _model.encode(text, normalize_embeddings=True).tolist()

    # Build chunker
    from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker

    chunker_map = {
        "fixed":     FixedSizeChunker(chunk_size=500, overlap=50),
        "sentence":  SentenceChunker(max_sentences_per_chunk=3),
        "recursive": RecursiveChunker(chunk_size=500),
    }
    chunker = chunker_map[args.strategy]
    strategy_name = type(chunker).__name__

    # Index documents
    print(f"Indexing documents with {strategy_name}...", flush=True)
    index = build_index(chunker, embed)
    print(f"Index built: {len(index)} chunks total\n")

    # Interactive loop
    print(f"=== Retrieval ready | strategy={strategy_name} | top_k={args.k} ===")
    print("Type your query and press Enter. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        results = retrieve(query, index, embed, top_k=args.k)
        print_results(results, query)


if __name__ == "__main__":
    main()
