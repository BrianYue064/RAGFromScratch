"""Data models for the vector store."""

from dataclasses import dataclass, field


@dataclass
class EmbeddedChunk:
    """A chunk of text with its embedding vector attached.

    This is an in-flight object between the embedding step and storage.
    The database assigns chunk_id (UUID) and created_at on insert.
    """

    text: str
    embedding: list[float]
    token_count: int
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        dim = len(self.embedding)
        return (
            f"EmbeddedChunk(chunk_index={self.chunk_index}, "
            f"token_count={self.token_count}, dim={dim}, "
            f"text='{preview}...')"
        )
