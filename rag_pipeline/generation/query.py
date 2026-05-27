"""High-level orchestrator that wires the full RAG query flow."""

import logging

from rag_pipeline.vectorstore import embed_text
from rag_pipeline.vectorstore.pg_store import PgVectorStore

from .llm_client import generate
from .prompt_builder import build_prompt

logger = logging.getLogger(__name__)


def rag_query(question: str, config) -> dict:
    """Run a full RAG query: embed → retrieve → prompt → generate.

    Args:
        question: The user's question.
        config: Pipeline configuration.

    Returns:
        A dict with keys: question, answer, sources, model, provider.
    """
    logger.info("Embedding query: %s", question)
    query_vec = embed_text(question, config)

    store = PgVectorStore(config)
    try:
        chunks = store.retrieve(query_vec, top_k=config.top_k)
    finally:
        store.close()

    logger.info("Retrieved %d chunks", len(chunks))

    prompt = build_prompt(question, chunks)
    answer = generate(prompt, config)

    sources = [
        {
            "text": c["text"],
            "source": c["source"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model": config.llm_model,
        "provider": config.llm_provider,
    }
