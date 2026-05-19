"""Embedding client for Ollama's local inference server.

Calls the /api/embed endpoint using the requests library.
All data stays local — zero egress from the home server.
"""

import logging
import time

import requests

from rag_pipeline.config import Config
from rag_pipeline.models import EmbeddingError

from ..ingestion.models import Chunk
from .models import EmbeddedChunk

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 120  # Ollama can be slow on first call (model loading)


def embed_text(text: str, config: Config) -> list[float]:
    """Embed a single text string via Ollama.

    Args:
        text: The text to embed.
        config: Pipeline configuration with ollama_url, embedding_model,
            and embedding_dim.

    Returns:
        A list of floats with length == config.embedding_dim.

    Raises:
        EmbeddingError: If the request fails or dimensions don't match.
    """
    url = f"{config.ollama_url}/api/embed"
    payload = {"model": config.embedding_model, "input": text}

    try:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
    except requests.ConnectionError as e:
        raise EmbeddingError(
            f"Cannot connect to Ollama at {config.ollama_url}"
        ) from e
    except requests.Timeout as e:
        raise EmbeddingError(
            f"Ollama request timed out after {_REQUEST_TIMEOUT}s"
        ) from e
    except requests.RequestException as e:
        raise EmbeddingError(f"Ollama embedding request failed: {e}") from e

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise EmbeddingError(f"Empty embedding response from Ollama: {data}")

    vector = embeddings[0]

    if len(vector) != config.embedding_dim:
        raise EmbeddingError(
            f"Dimension mismatch: expected {config.embedding_dim}, "
            f"got {len(vector)}"
        )

    logger.debug(f"Embedded {len(text)} chars in {elapsed_ms:.0f}ms")
    return vector


def embed_chunks(chunks: list[Chunk], config: Config) -> list[EmbeddedChunk]:
    """Embed a list of Chunks, returning EmbeddedChunks with vectors attached.

    Args:
        chunks: List of Chunk objects from the ingestion layer.
        config: Pipeline configuration.

    Returns:
        List of EmbeddedChunk objects with embedding vectors.

    Raises:
        EmbeddingError: If any embedding call fails.
    """
    embedded: list[EmbeddedChunk] = []

    for i, ch in enumerate(chunks):
        logger.info(f"Embedding chunk {i + 1}/{len(chunks)} from {ch.source}")
        vector = embed_text(ch.text, config)
        embedded.append(
            EmbeddedChunk(
                text=ch.text,
                embedding=vector,
                token_count=ch.token_count,
                source=ch.source,
                chunk_index=ch.chunk_index,
                metadata=dict(ch.metadata),
            )
        )

    logger.info(f"Embedded {len(embedded)} chunks")
    return embedded
