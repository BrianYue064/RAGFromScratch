"""Vector store package for embedding and retrieval."""

from .embedder import embed_chunks, embed_text
from .models import EmbeddedChunk

__all__ = ["embed_text", "embed_chunks", "EmbeddedChunk", "PgVectorStore"]


def __getattr__(name: str):
    """Lazy-import PgVectorStore so psycopg is only required at use time."""
    if name == "PgVectorStore":
        from .pg_store import PgVectorStore

        return PgVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

