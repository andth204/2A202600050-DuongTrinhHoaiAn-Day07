"""
Group Task — Document Indexing Script
Indexes all Python tutorial files using FixedSizeChunker + all-MiniLM-L6-v2
into ChromaDB persistent store at CHROMA_PERSIST_DIR=./chroma_data
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

# File metadata mapping
FILE_METADATA = {
    "python_intro_and_variables.txt": {
        "topic": "intro_variables",
        "source": "python_docs",
        "level": "basic",
    },
    "python_conditionals_loops_functions.txt": {
        "topic": "control_flow",
        "source": "python_docs",
        "level": "basic",
    },
    "python_dictionaries_sets_list_tuples.txt": {
        "topic": "collections",
        "source": "python_docs",
        "level": "basic",
    },
    "python_input_output.txt": {
        "topic": "input_output",
        "source": "python_docs",
        "level": "basic",
    },
    "python_error_exception.txt": {
        "topic": "exceptions",
        "source": "python_docs",
        "level": "basic",
    },
    "python_module.txt": {
        "topic": "modules",
        "source": "python_docs",
        "level": "basic",
    },
}

CHUNK_SIZE = 500
OVERLAP = 50


def _batch_embed(openai_client, texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts using OpenAI API (batched)."""
    response = openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    import chromadb
    from openai import OpenAI

    from src.chunking import FixedSizeChunker

    print(f"Using embedder: {OPENAI_EMBEDDING_MODEL} (OpenAI)")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Connecting to ChromaDB at {CHROMA_PERSIST_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Drop and recreate collection for clean re-index
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    data_dir = Path("data")

    total_chunks = 0
    for filename, meta in FILE_METADATA.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        text = filepath.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        print(f"  {filename}: {len(text):,} chars -> {len(chunks)} chunks")

        ids = [f"{filename}:{i}" for i in range(len(chunks))]
        embeddings = _batch_embed(openai_client, chunks, OPENAI_EMBEDDING_MODEL)
        metadatas = [{**meta, "doc_id": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\nDone! Indexed {total_chunks} chunks into collection '{COLLECTION_NAME}'")
    print(f"Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"ChromaDB path: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
