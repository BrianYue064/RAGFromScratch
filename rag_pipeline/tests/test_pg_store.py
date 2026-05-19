"""Integration tests for PgVectorStore (requires PostgreSQL + pgvector via Docker)."""

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg required for integration tests")

from rag_pipeline.config import Config
from rag_pipeline.models import VectorStoreError
from rag_pipeline.vectorstore.models import EmbeddedChunk
from rag_pipeline.vectorstore.pg_store import PgVectorStore


@pytest.fixture(scope="session")
def pg_config(docker_services, docker_ip):
    """Wait for PostgreSQL to be ready, return a Config pointing at it."""
    port = docker_services.port_for("postgres", 5432)
    db_url = f"postgresql://postgres:postgres@{docker_ip}:{port}/rag_test"

    docker_services.wait_until_responsive(
        check=lambda: _pg_is_ready(docker_ip, port),
        timeout=30,
        pause=1,
    )
    return Config(
        db_url=db_url,
        embedding_dim=3,  # small dims for test speed
    )


def _pg_is_ready(host: str, port: int) -> bool:
    """Check if PostgreSQL is accepting connections."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def store(pg_config):
    """Create a fresh PgVectorStore and clean up after each test."""
    s = PgVectorStore(pg_config)
    s.initialize()

    yield s

    # Clean up: drop all rows between tests
    with s._conn.cursor() as cur:
        cur.execute("DELETE FROM chunks;")
    s._conn.commit()
    s.close()


def _make_chunk(
    text: str = "test chunk",
    embedding: list[float] | None = None,
    source: str = "test.txt",
    chunk_index: int = 0,
) -> EmbeddedChunk:
    """Helper to create an EmbeddedChunk with 3-dim embedding."""
    return EmbeddedChunk(
        text=text,
        embedding=embedding or [0.1, 0.2, 0.3],
        token_count=len(text.split()),
        source=source,
        chunk_index=chunk_index,
    )


@pytest.mark.integration
class TestPgVectorStore:
    """Integration tests for PgVectorStore."""

    def test_initialize_creates_table(self, store):
        """Initialize should create the chunks table."""
        with store._conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS ("
                "  SELECT FROM information_schema.tables"
                "  WHERE table_name = 'chunks'"
                ");"
            )
            assert cur.fetchone()[0] is True

    def test_initialize_idempotent(self, store):
        """Calling initialize twice should not raise an error."""
        store.initialize()  # second call (first was in fixture)

    def test_insert_and_count(self, store):
        """Inserting chunks should be reflected in count()."""
        chunks = [
            _make_chunk(text="first", chunk_index=0),
            _make_chunk(text="second", chunk_index=1),
            _make_chunk(text="third", chunk_index=2),
        ]
        inserted = store.insert(chunks)

        assert inserted == 3
        assert store.count() == 3

    def test_insert_empty_list(self, store):
        """Inserting an empty list should return 0 and not error."""
        assert store.insert([]) == 0
        assert store.count() == 0

    def test_retrieve_returns_sorted_results(self, store):
        """Results should be sorted by cosine similarity, highest first."""
        # Insert chunks with known embeddings
        store.insert([
            _make_chunk(text="similar", embedding=[1.0, 0.0, 0.0], chunk_index=0),
            _make_chunk(text="different", embedding=[0.0, 1.0, 0.0], chunk_index=1),
            _make_chunk(text="most similar", embedding=[0.9, 0.1, 0.0], chunk_index=2),
        ])

        # Query with a vector close to [1, 0, 0]
        results = store.retrieve([1.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        # First result should be the most similar
        assert results[0]["text"] == "similar"
        # Similarity scores should be descending
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_retrieve_top_k_limit(self, store):
        """Retrieve should respect the top_k limit."""
        chunks = [
            _make_chunk(text=f"chunk {i}", chunk_index=i)
            for i in range(10)
        ]
        store.insert(chunks)

        results = store.retrieve([0.1, 0.2, 0.3], top_k=3)

        assert len(results) == 3

    def test_retrieve_empty_store(self, store):
        """Retrieving from an empty store should return an empty list."""
        results = store.retrieve([0.1, 0.2, 0.3], top_k=5)

        assert results == []

    def test_delete_by_source(self, store):
        """Deleting by source should remove only matching chunks."""
        store.insert([
            _make_chunk(text="keep", source="keep.txt", chunk_index=0),
            _make_chunk(text="delete1", source="delete.txt", chunk_index=0),
            _make_chunk(text="delete2", source="delete.txt", chunk_index=1),
        ])

        deleted = store.delete_by_source("delete.txt")

        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent_source(self, store):
        """Deleting a source that doesn't exist should return 0."""
        deleted = store.delete_by_source("nonexistent.txt")

        assert deleted == 0

    def test_retrieve_result_fields(self, store):
        """Retrieved results should contain all expected fields."""
        store.insert([
            _make_chunk(
                text="test content",
                source="doc.pdf",
                chunk_index=5,
            )
        ])

        results = store.retrieve([0.1, 0.2, 0.3], top_k=1)

        assert len(results) == 1
        result = results[0]
        assert "chunk_id" in result
        assert result["text"] == "test content"
        assert result["source"] == "doc.pdf"
        assert result["chunk_index"] == 5
        assert "similarity" in result
        assert isinstance(result["similarity"], float)
