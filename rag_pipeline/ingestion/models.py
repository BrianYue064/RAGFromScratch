"""Data models and exceptions for the RAG ingestion pipeline."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors."""

    pass


class UnsupportedFileTypeError(RAGPipelineError):
    """Raised when a file type is not supported for ingestion."""

    pass


class IngestionError(RAGPipelineError):
    """Raised when content ingestion fails."""

    pass


class ChunkingError(RAGPipelineError):
    """Raised when chunking fails."""

    pass


@dataclass
class Document:
    """Represents a loaded document ready for chunking."""

    content: str
    source: str
    file_type: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"Document(source={self.source!r}, file_type={self.file_type!r}, "
            f"content='{preview}...')"
        )


@dataclass
class Chunk:
    """Represents a chunk of text extracted from a Document."""

    text: str
    token_count: int
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(chunk_index={self.chunk_index}, token_count={self.token_count}, "
            f"text='{preview}...')"
        )
