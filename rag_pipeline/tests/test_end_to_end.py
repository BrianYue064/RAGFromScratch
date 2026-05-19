"""End-to-end integration tests for the full RAG pipeline.

Tests the flow: load → chunk → embed (mocked) → store → retrieve.
Requires PostgreSQL + pgvector via Docker (pytest-docker).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg required for integration tests")

from rag_pipeline.config import Config
from rag_pipeline.ingestion import load, chunk
from rag_pipeline.vectorstore.embedder import embed_chunks
from rag_pipeline.vectorstore.models import EmbeddedChunk
from rag_pipeline.vectorstore.pg_store import PgVectorStore


FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Use a small embedding dimension for test speed
TEST_DIM = 3


@pytest.fixture(scope="session")
def pg_config(docker_services, docker_ip):
    """Wait for PostgreSQL to be ready, return a Config for e2e tests."""
    port = docker_services.port_for("postgres", 5432)
    db_url = f"postgresql://postgres:postgres@{docker_ip}:{port}/rag_test"

    docker_services.wait_until_responsive(
        check=lambda: _pg_is_ready(docker_ip, port),
        timeout=30,
        pause=1,
    )
    return Config(
        db_url=db_url,
        embedding_dim=TEST_DIM,
    )


def _pg_is_ready(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def store(pg_config):
    """Provide a clean PgVectorStore for each test."""
    s = PgVectorStore(pg_config)
    s.initialize()

    yield s

    with s._conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")
    s._conn.commit()
    s.close()


def _mock_embed_response():
    """Create a mock requests response with a TEST_DIM-dim vector."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[0.5] * TEST_DIM]}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end tests for the full pipeline."""

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_ingest_embed_store_retrieve(self, mock_post, store, pg_config):
        """Full pipeline: load a file → chunk → embed → store → retrieve."""
        mock_post.return_value = _mock_embed_response()

        # 1. Load and chunk
        doc = load(FIXTURES_DIR / "sample.txt")
        chunks = chunk(doc, chunk_size=50, overlap=10)
        assert len(chunks) > 0

        # 2. Embed (mocked Ollama)
        embedded = embed_chunks(chunks, pg_config)
        assert len(embedded) == len(chunks)
        assert all(isinstance(ec, EmbeddedChunk) for ec in embedded)

        # 3. Store
        inserted = store.insert(embedded)
        assert inserted == len(chunks)
        assert store.count() == len(chunks)

        # 4. Retrieve
        query_vector = [0.5] * TEST_DIM
        results = store.retrieve(query_vector, top_k=3)
        assert len(results) > 0
        assert "similarity" in results[0]
        assert "text" in results[0]

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_round_trip_preserves_content(self, mock_post, store, pg_config):
        """Stored text should exactly match the original chunk text."""
        mock_post.return_value = _mock_embed_response()

        doc = load(FIXTURES_DIR / "sample.txt")
        chunks = chunk(doc, chunk_size=512, overlap=50)

        embedded = embed_chunks(chunks, pg_config)
        store.insert(embedded)

        query_vector = [0.5] * TEST_DIM
        results = store.retrieve(query_vector, top_k=len(chunks))

        stored_texts = {r["text"] for r in results}
        original_texts = {ch.text for ch in chunks}

        assert stored_texts == original_texts
