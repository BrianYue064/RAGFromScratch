"""Document chunking module for splitting documents into semantic chunks."""

import logging
from typing import List

import nltk
import tiktoken

# Ensure required NLTK data is available; downloads silently if missing.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from .models import Chunk, ChunkingError, Document

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk(doc: Document, chunk_size: int = 512, overlap: int = 50) -> List[Chunk]:
    """Split a Document into chunks using sentence-boundary-aware chunking.

    Maintains token counts incrementally to avoid O(n^2) re-encoding of the
    entire running text on every sentence.

    Args:
        doc: The Document to chunk.
        chunk_size: Target token count per chunk.
        overlap: Number of tokens to carry over between chunks.

    Returns:
        List of Chunk objects.

    Raises:
        ChunkingError: If chunking fails.
    """
    try:
        sentences = nltk.sent_tokenize(doc.content)
    except Exception as e:
        logger.error(f"Failed to tokenize sentences for {doc.source}: {e}")
        raise ChunkingError(f"Failed to tokenize sentences: {doc.source}") from e

    if not sentences:
        logger.warning(f"No sentences found in document: {doc.source}")
        return []

    chunks: List[Chunk] = []
    current_chunk_tokens: List[int] = []

    for sentence in sentences:
        sentence_prefix = " " if current_chunk_tokens else ""
        sentence_tokens = ENCODING.encode(sentence_prefix + sentence)
        sentence_token_count = len(sentence_tokens)

        projected_count = len(current_chunk_tokens) + sentence_token_count

        if projected_count <= chunk_size:
            current_chunk_tokens.extend(sentence_tokens)
        else:
            if current_chunk_tokens:
                token_count = len(current_chunk_tokens)
                if token_count > chunk_size:
                    logger.warning(
                        f"Chunk {len(chunks)} exceeds chunk_size: {token_count} tokens"
                    )

                chunks.append(
                    Chunk(
                        text=ENCODING.decode(current_chunk_tokens).strip(),
                        token_count=token_count,
                        source=doc.source,
                        chunk_index=len(chunks),
                        metadata=dict(doc.metadata),
                    )
                )

                if overlap > 0:
                    current_chunk_tokens = list(current_chunk_tokens[-overlap:])
                else:
                    current_chunk_tokens = []
            else:
                current_chunk_tokens = []

            if sentence_token_count > chunk_size:
                logger.warning(
                    f"Sentence exceeds chunk_size: {sentence_token_count} tokens"
                )
                tokens_split = _split_at_token_level(
                    sentence_tokens, chunk_size, overlap
                )
                for token_batch in tokens_split:
                    chunks.append(
                        Chunk(
                            text=ENCODING.decode(token_batch).strip(),
                            token_count=len(token_batch),
                            source=doc.source,
                            chunk_index=len(chunks),
                            metadata=dict(doc.metadata),
                        )
                    )
                if overlap > 0 and tokens_split:
                    current_chunk_tokens = list(tokens_split[-1][-overlap:])
                else:
                    current_chunk_tokens = []
            else:
                current_chunk_tokens.extend(sentence_tokens)

    if current_chunk_tokens:
        token_count = len(current_chunk_tokens)
        if token_count > chunk_size:
            logger.warning(
                f"Final chunk {len(chunks)} exceeds chunk_size: {token_count} tokens"
            )

        chunks.append(
            Chunk(
                text=ENCODING.decode(current_chunk_tokens).strip(),
                token_count=token_count,
                source=doc.source,
                chunk_index=len(chunks),
                metadata=dict(doc.metadata),
            )
        )

    for i, ch in enumerate(chunks):
        ch.chunk_index = i

    return chunks


def _split_at_token_level(
    tokens: List[int], chunk_size: int, overlap: int
) -> List[List[int]]:
    """Split a token list at the token level when a sentence exceeds chunk_size."""
    result = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        result.append(tokens[start:end])
        start = end - overlap if overlap > 0 else end
    return result
