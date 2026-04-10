from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_sections = []

        for index, result in enumerate(results, start=1):
            context_sections.append(f"[{index}] {result['content']}")

        context = "\n\n".join(context_sections) if context_sections else "No relevant context retrieved."
        prompt = (
            "You are a helpful assistant answering questions from a knowledge base.\n"
            "Use only the retrieved context below when possible.\n\n"
            "There is no need to provide the source information for the document."
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return str(self.llm_fn(prompt))
