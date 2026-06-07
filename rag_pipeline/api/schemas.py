"""Pydantic schemas for the RAG API."""

from pydantic import BaseModel


# ── Query ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    model: str | None = None
    provider: str | None = None
    stream: bool = True


class SourceChunk(BaseModel):
    text: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    model: str
    provider: str


# ── Embed ──────────────────────────────────────────────
class EmbedRequest(BaseModel):
    path: str
    chunk_size: int | None = None
    overlap: int | None = None


class EmbedResponse(BaseModel):
    source: str
    chunks_stored: int
    embedding_model: str
    embedding_dim: int


# ── Chunks ─────────────────────────────────────────────
class ChunkInfo(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    token_count: int
    similarity: float | None = None


class ChunksListResponse(BaseModel):
    chunks: list[ChunkInfo]
    total: int


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int


# ── Health ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str


# ── Errors ─────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
