"""High-level orchestrator that wires the full RAG query flow."""

import logging
from typing import Generator

from rag_pipeline.config import Config
from rag_pipeline.vectorstore import embed_text
from rag_pipeline.vectorstore.pg_store import PgVectorStore

from .llm_client import generate, generate_stream
from .prompt_builder import build_prompt

logger = logging.getLogger(__name__)


def rag_query(
    question: str,
    config: Config,
    *,
    top_k: int | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """Run a full RAG query: embed → retrieve → prompt → generate.

    Args:
        question: The user's question.
        config: Pipeline configuration.
        top_k: Override for config.top_k.
        model: Override for config.llm_model.
        provider: Override for config.llm_provider.

    Returns:
        A dict with keys: question, answer, sources, model, provider.
    """
    logger.info("Embedding query: %s", question)
    query_vec = embed_text(question, config)

    # Use overrides if provided
    actual_top_k = top_k if top_k is not None else config.top_k
    
    # We create a temporary config to pass down overridden model/provider
    run_config = config
    if model is not None or provider is not None:
        run_config = Config(**config.__dict__)
        if model is not None:
            run_config.llm_model = model
        if provider is not None:
            run_config.llm_provider = provider

    store = PgVectorStore(run_config)
    try:
        chunks = store.retrieve(query_vec, top_k=actual_top_k)
    finally:
        store.close()

    logger.info("Retrieved %d chunks", len(chunks))

    prompt = build_prompt(question, chunks)
    answer = generate(prompt, run_config)

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
        "model": run_config.llm_model,
        "provider": run_config.llm_provider,
    }


def rag_query_stream(
    question: str,
    config: Config,
    *,
    top_k: int | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[list[dict], Generator[str, None, None]]:
    """Run RAG query and return (sources, token_generator).

    Returns sources eagerly so the API can include metadata
    while the LLM streams tokens.
    """
    logger.info("Embedding query for stream: %s", question)
    query_vec = embed_text(question, config)

    # Use overrides if provided
    actual_top_k = top_k if top_k is not None else config.top_k
    
    # We create a temporary config to pass down overridden model/provider
    run_config = config
    if model is not None or provider is not None:
        run_config = Config(**config.__dict__)
        if model is not None:
            run_config.llm_model = model
        if provider is not None:
            run_config.llm_provider = provider

    store = PgVectorStore(run_config)
    try:
        chunks = store.retrieve(query_vec, top_k=actual_top_k)
    finally:
        store.close()

    logger.info("Retrieved %d chunks for stream", len(chunks))

    prompt = build_prompt(question, chunks)
    
    sources = [
        {
            "text": c["text"],
            "source": c["source"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]
    
    token_gen = generate_stream(prompt, run_config)
    
    return sources, token_gen
