"""Ingestion package for loading and chunking documents."""

from .chunker import chunk
from .loader import load
from .models import Chunk, Document, RAGPipelineError

__all__ = ["load", "chunk", "Document", "Chunk", "RAGPipelineError"]