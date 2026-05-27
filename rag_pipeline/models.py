"""Shared exceptions for the RAG pipeline.

All exception classes live here so that every subpackage (ingestion,
vectorstore, evaluation, …) can inherit from the same base without
circular imports.
"""


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


class EmbeddingError(RAGPipelineError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(RAGPipelineError):
    """Raised when vector store operations fail."""

    pass


class GenerationError(RAGPipelineError):
    """Raised when LLM generation fails."""

    pass
