"""
Group Task — Benchmark Runner (FixedSizeChunker + all-MiniLM-L6-v2)
Runs the 5 group benchmark queries against the ChromaDB index and prints results
for filling in REPORT.md Section 2 / Section 6.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME = "python_docs"

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What does range(5) generate in Python?",
        "gold": "range(5) generates 0, 1, 2, 3, 4, and the end point is not included.",
        "expected_file": "python_conditionals_loops_functions.txt",
    },
    {
        "id": 2,
        "query": "How can a list be used as a stack in Python?",
        "gold": "Use append() to push items and pop() without an index to remove the last item in LIFO order.",
        "expected_file": "python_dictionaries_sets_list_tuples.txt",
    },
    {
        "id": 3,
        "query": "What is the difference between str() and repr()?",
        "gold": "str() returns a human-readable representation, while repr() returns an interpreter-readable or debugging representation.",
        "expected_file": "python_input_output.txt",
    },
    {
        "id": 4,
        "query": "What happens if an exception type matches the except clause?",
        "gold": "The except clause executes, and execution then continues after the try/except block.",
        "expected_file": "python_error_exception.txt",
    },
    {
        "id": 5,
        "query": "What does import fibo do and how can you access fib() after importing?",
        "gold": "import fibo binds the module name fibo in the current namespace, and the function is called as fibo.fib(...).",
        "expected_file": "python_module.txt",
    },
]


def main() -> None:
    import chromadb
    from openai import OpenAI

    print(f"Using embedder: {OPENAI_EMBEDDING_MODEL} (OpenAI)")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} chunks\n")
    print("=" * 80)

    results_table = []

    for bq in BENCHMARK_QUERIES:
        query = bq["query"]
        resp = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=query)
        q_emb = resp.data[0].embedding

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # cosine distance → cosine similarity
        scores = [round(1 - d, 4) for d in distances]

        top_doc = docs[0]
        top_meta = metas[0]
        top_score = scores[0]
        top_file = top_meta.get("doc_id", "?")
        relevant = top_file == bq["expected_file"]

        preview = top_doc[:120].replace("\n", " ") + "..."

        print(f"Q{bq['id']}: {query}")
        print(f"  Top-1 file : {top_file}")
        print(f"  Score      : {top_score}")
        print(f"  Preview    : {preview}")
        print(f"  Relevant?  : {'Yes' if relevant else 'No'}")
        print()

        results_table.append(
            {
                "id": bq["id"],
                "query": query,
                "top1_preview": preview,
                "score": top_score,
                "relevant": "Yes" if relevant else "No",
            }
        )

    print("=" * 80)
    print("\n--- Markdown Table (for REPORT.md) ---\n")
    print("| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? |")
    print("|---|---|---|---|---|")
    for row in results_table:
        print(f"| {row['id']} | {row['query']} | {row['top1_preview']} | {row['score']} | {row['relevant']} |")


if __name__ == "__main__":
    main()
