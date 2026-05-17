"""Configuration for the RAG pipeline.

Reads settings from environment variables.
All variables are prefixed with RAG_ for namespacing.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    chunk_size: int = 512
    overlap: int = 50
    encoding: str = "cl100k_base"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    db_url: str = "postgresql://postgres:postgres@localhost:5432/rag"


def load_config() -> Config:
    return Config(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        overlap=int(os.getenv("RAG_OVERLAP", "50")),
        encoding=os.getenv("RAG_ENCODING", "cl100k_base"),
        ollama_url=os.getenv("RAG_OLLAMA_URL", "http://localhost:11434"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
        db_url=os.getenv(
            "RAG_DB_URL", "postgresql://postgres:postgres@localhost:5432/rag"
        ),
    )
