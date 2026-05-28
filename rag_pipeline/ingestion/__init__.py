"""Ingestion package for loading and chunking documents."""

from .chunker import chunk
from .loader import load
from .models import Chunk, Document, RAGPipelineError
from .resolver import resolve_paths

__all__ = [
    "load",
    "chunk",
    "resolve_paths",
    "Document",
    "Chunk",
    "RAGPipelineError",
]