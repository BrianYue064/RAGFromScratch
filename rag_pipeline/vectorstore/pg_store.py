"""PostgreSQL + pgvector backed vector store.

Stores embedded chunks and performs cosine-similarity retrieval
using the <=> operator. All data stays in the local PostgreSQL instance.
"""

import json
import logging
import uuid

import psycopg
from pgvector.psycopg import register_vector

from rag_pipeline.config import Config
from rag_pipeline.models import VectorStoreError

from .models import EmbeddedChunk

logger = logging.getLogger(__name__)


class PgVectorStore:
    """PostgreSQL + pgvector backed vector store."""

    def __init__(self, config: Config):
        """Connect to PostgreSQL and register pgvector types.

        Args:
            config: Pipeline configuration with db_url and embedding_dim.

        Raises:
            VectorStoreError: If connection fails.
        """
        self._config = config
        try:
            self._conn = psycopg.connect(config.db_url, autocommit=False)
            logger.info("Connected to PostgreSQL")
        except psycopg.Error as e:
            raise VectorStoreError(
                f"Failed to connect to database: {e}"
            ) from e

    def initialize(self) -> None:
        """Create pgvector extension, chunks table, and HNSW index.

        Safe to call multiple times (all statements use IF NOT EXISTS).

        Raises:
            VectorStoreError: If schema creation fails.
        """
        dim = self._config.embedding_dim
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self._conn.commit()
            register_vector(self._conn)
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        text        TEXT NOT NULL,
                        embedding   VECTOR({dim}) NOT NULL,
                        token_count INTEGER NOT NULL,
                        source      TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        metadata    JSONB DEFAULT '{{}}',
                        created_at  TIMESTAMPTZ DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON chunks USING hnsw (embedding vector_cosine_ops);
                    """
                )
            self._conn.commit()
            logger.info(
                f"Vector store initialized (embedding dim={dim})"
            )
        except psycopg.Error as e:
            self._conn.rollback()
            raise VectorStoreError(
                f"Failed to initialize vector store: {e}"
            ) from e

    def insert(self, chunks: list[EmbeddedChunk]) -> int:
        """Batch insert embedded chunks.

        Args:
            chunks: List of EmbeddedChunk objects to store.

        Returns:
            Number of chunks inserted.

        Raises:
            VectorStoreError: If insertion fails.
        """
        if not chunks:
            return 0

        sql = """
            INSERT INTO chunks
                (chunk_id, text, embedding, token_count, source,
                 chunk_index, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        rows = [
            (
                str(uuid.uuid4()),
                ch.text,
                ch.embedding,
                ch.token_count,
                ch.source,
                ch.chunk_index,
                json.dumps(ch.metadata),
            )
            for ch in chunks
        ]
        try:
            with self._conn.cursor() as cur:
                cur.executemany(sql, rows)
            self._conn.commit()
            logger.info(f"Inserted {len(rows)} chunks")
            return len(rows)
        except psycopg.Error as e:
            self._conn.rollback()
            raise VectorStoreError(
                f"Failed to insert chunks: {e}"
            ) from e

    def retrieve(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Cosine-similarity search using the <=> operator.

        Args:
            query_embedding: The query vector (must match embedding_dim).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, text, source, chunk_index,
            metadata, similarity (float, 0–1, higher = more similar).

        Raises:
            VectorStoreError: If retrieval fails.
        """
        sql = """
            SELECT chunk_id, text, source, chunk_index, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (query_embedding, query_embedding, top_k))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        except psycopg.Error as e:
            raise VectorStoreError(
                f"Failed to retrieve chunks: {e}"
            ) from e

    def count(self) -> int:
        """Return total number of stored chunks.

        Raises:
            VectorStoreError: If the query fails.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks;")
                return cur.fetchone()[0]
        except psycopg.Error as e:
            raise VectorStoreError(
                f"Failed to count chunks: {e}"
            ) from e

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source.

        Args:
            source: The source identifier to delete by.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If deletion fails.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE source = %s;", (source,)
                )
                count = cur.rowcount
            self._conn.commit()
            logger.info(f"Deleted {count} chunks from source: {source}")
            return count
        except psycopg.Error as e:
            self._conn.rollback()
            raise VectorStoreError(
                f"Failed to delete chunks: {e}"
            ) from e

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("Database connection closed")
