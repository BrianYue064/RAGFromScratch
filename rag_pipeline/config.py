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
    embedding_dim: int = 768
    db_url: str = "postgresql://postgres:postgres@localhost:5432/rag"
    llm_provider: str = "lmstudio"
    llm_model: str = "qwen3.5-0.8b"
    lmstudio_url: str = "http://localhost:1234"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    top_k: int = 5


def load_config() -> Config:
    return Config(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
        overlap=int(os.getenv("RAG_OVERLAP", "50")),
        encoding=os.getenv("RAG_ENCODING", "cl100k_base"),
        ollama_url=os.getenv("RAG_OLLAMA_URL", "http://localhost:11434"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_dim=int(os.getenv("RAG_EMBEDDING_DIM", "768")),
        db_url=os.getenv(
            "RAG_DB_URL", "postgresql://postgres:postgres@localhost:5432/rag"
        ),
        llm_provider=os.getenv("RAG_LLM_PROVIDER", "lmstudio"),
        llm_model=os.getenv("RAG_LLM_MODEL", "qwen3.5-0.8b"),
        lmstudio_url=os.getenv("RAG_LMSTUDIO_URL", "http://localhost:1234"),
        llm_temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0.7")),
        llm_max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "1024")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
    )
