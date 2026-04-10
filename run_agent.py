"""
Agent Runner -- uses text-embedding-3-small (OpenAI) for embeddings + OpenAI chat for generation
Interactive mode: type questions, agent retrieves chunks and answers.

Usage:
    python run_agent.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "python_docs_openai"

FILE_METADATA = {
    "python_intro_and_variables.txt": {"topic": "intro_variables", "source": "python_docs", "level": "basic"},
    "python_conditionals_loops_functions.txt": {"topic": "control_flow", "source": "python_docs", "level": "basic"},
    "python_dictionaries_sets_list_tuples.txt": {"topic": "collections", "source": "python_docs", "level": "basic"},
    "python_input_output.txt": {"topic": "input_output", "source": "python_docs", "level": "basic"},
    "python_error_exception.txt": {"topic": "exceptions", "source": "python_docs", "level": "basic"},
    "python_module.txt": {"topic": "modules", "source": "python_docs", "level": "basic"},
}

CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 3


def build_openai_embedder():
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
        return [float(v) for v in response.data[0].embedding]

    return embed, client


def ensure_index(collection, embed_fn) -> None:
    """Index documents if collection is empty."""
    from pathlib import Path

    from src.chunking import FixedSizeChunker

    if collection.count() > 0:
        print(f"[index] Collection '{COLLECTION_NAME}' already has {collection.count()} chunks -- skipping re-index.")
        return

    print(f"[index] Indexing documents with {OPENAI_EMBEDDING_MODEL} ...")
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    data_dir = Path("data")

    total = 0
    for filename, meta in FILE_METADATA.items():
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        text = filepath.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        print(f"  {filename}: {len(chunks)} chunks")

        ids = [f"openai:{filename}:{i}" for i in range(len(chunks))]
        embeddings = [embed_fn(c) for c in chunks]
        metadatas = [{**meta, "doc_id": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        total += len(chunks)

    print(f"[index] Done. {total} chunks indexed.\n")


def build_llm_fn(openai_client):
    def llm(prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return llm


def search_and_log(collection, embed_fn, question: str, top_k: int) -> list[dict]:
    """Embed question, search ChromaDB, print chunk log, return results."""
    q_emb = embed_fn(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    print(f"\n[retrieval] Top-{top_k} chunks retrieved:")
    print("-" * 60)
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        score = round(1 - dist, 4)
        doc_id = meta.get("doc_id", "?")
        topic = meta.get("topic", "?")
        chunk_idx = meta.get("chunk_index", "?")
        preview = doc[:200].replace("\n", " ")
        print(f"  [{i}] file={doc_id}  topic={topic}  chunk={chunk_idx}  score={score}")
        print(f"       {preview}...")
        print()
        chunks.append({"content": doc, "metadata": meta, "score": score})

    return chunks


def answer_question(question: str, chunks: list[dict], llm_fn) -> str:
    context_sections = [f"[{i}] {c['content']}" for i, c in enumerate(chunks, start=1)]
    context = "\n\n".join(context_sections) if context_sections else "No relevant context retrieved."
    prompt = (
        "You are a helpful assistant answering questions from a knowledge base.\n"
        "Use only the retrieved context below when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return llm_fn(prompt)


def main() -> None:
    import chromadb

    embed_fn, openai_client = build_openai_embedder()
    llm_fn = build_llm_fn(openai_client)

    print(f"[setup] Connecting to ChromaDB at {CHROMA_PERSIST_DIR} ...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ensure_index(collection, embed_fn)

    print("=" * 60)
    print("  Python Knowledge Base Agent")
    print(f"  Embedder : {OPENAI_EMBEDDING_MODEL}")
    print(f"  LLM      : gpt-4o-mini")
    print(f"  Chunks   : {collection.count()} indexed")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        chunks = search_and_log(collection, embed_fn, question, top_k=TOP_K)

        print("[answer]")
        print("-" * 60)
        answer = answer_question(question, chunks, llm_fn)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
