"""Document chunking module for splitting documents into semantic chunks."""

import logging
from typing import List

import nltk
import tiktoken

from .models import Chunk, ChunkingError, Document

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk(
    doc: Document, chunk_size: int = 512, overlap: int = 50
) -> List[Chunk]:
    """Split a Document into chunks using sentence-boundary-aware chunking.

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
    current_chunk_text = ""
    current_chunk_tokens = []

    for sentence in sentences:
        sentence_tokens = ENCODING.encode(sentence)
        sentence_token_count = len(sentence_tokens)

        tokens_with_sentence = ENCODING.encode(current_chunk_text + " " + sentence)

        if len(tokens_with_sentence) <= chunk_size:
            current_chunk_text = current_chunk_text + " " + sentence
            current_chunk_tokens = tokens_with_sentence
        else:
            if current_chunk_text:
                token_count = len(current_chunk_tokens)
                if token_count > chunk_size:
                    logger.warning(
                        f"Chunk {len(chunks)} exceeds chunk_size: {token_count} tokens"
                    )

                chunks.append(
                    Chunk(
                        text=current_chunk_text.strip(),
                        token_count=token_count,
                        source=doc.source,
                        chunk_index=len(chunks),
                        metadata=dict(doc.metadata),
                    )
                )

                if overlap > 0 and current_chunk_tokens:
                    overlap_tokens = current_chunk_tokens[-overlap:]
                    overlap_text = ENCODING.decode(overlap_tokens)
                    current_chunk_text = overlap_text + " " + sentence
                    current_chunk_tokens = ENCODING.encode(current_chunk_text)
                else:
                    current_chunk_text = sentence
                    current_chunk_tokens = sentence_tokens
            else:
                current_chunk_text = sentence
                current_chunk_tokens = sentence_tokens

                if sentence_token_count > chunk_size:
                    logger.warning(
                        f"Sentence exceeds chunk_size: {sentence_token_count} tokens"
                    )
                    tokens_split = _split_at_token_level(
                        sentence_tokens, chunk_size, overlap
                    )
                    for i, token_batch in enumerate(tokens_split):
                        chunks.append(
                            Chunk(
                                text=ENCODING.decode(token_batch).strip(),
                                token_count=len(token_batch),
                                source=doc.source,
                                chunk_index=len(chunks) + i,
                                metadata=dict(doc.metadata),
                            )
                        )
                    current_chunk_text = ""
                    current_chunk_tokens = []

    if current_chunk_text:
        token_count = len(current_chunk_tokens)
        if token_count > chunk_size:
            logger.warning(
                f"Final chunk {len(chunks)} exceeds chunk_size: {token_count} tokens"
            )

        chunks.append(
            Chunk(
                text=current_chunk_text.strip(),
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