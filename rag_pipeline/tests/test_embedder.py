"""Tests for the embedding client (unit tests with mocked HTTP)."""

import pytest
from unittest.mock import patch, MagicMock

from rag_pipeline.config import Config
from rag_pipeline.models import EmbeddingError
from rag_pipeline.vectorstore.embedder import embed_text, embed_chunks
from rag_pipeline.vectorstore.models import EmbeddedChunk
from rag_pipeline.ingestion.models import Chunk


@pytest.fixture
def config():
    """Config with default embedding settings."""
    return Config(embedding_dim=768)


@pytest.fixture
def config_384():
    """Config with custom 384-dim embedding."""
    return Config(embedding_dim=384, embedding_model="custom-model")


def _make_mock_response(vector):
    """Create a mock requests.Response with the given embedding vector."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [vector]}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestEmbedText:
    """Tests for the embed_text function."""

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_returns_correct_dims(self, mock_post, config):
        """Embedding response with 768 dims should return a 768-length list."""
        fake_vector = [0.1] * 768
        mock_post.return_value = _make_mock_response(fake_vector)

        result = embed_text("hello world", config)

        assert isinstance(result, list)
        assert len(result) == 768
        mock_post.assert_called_once()

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_custom_dim(self, mock_post, config_384):
        """Config with 384 dims should accept a 384-length vector."""
        fake_vector = [0.2] * 384
        mock_post.return_value = _make_mock_response(fake_vector)

        result = embed_text("test", config_384)

        assert len(result) == 384

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_dimension_mismatch(self, mock_post, config):
        """Wrong dimensions in response should raise EmbeddingError."""
        fake_vector = [0.1] * 512  # config expects 768
        mock_post.return_value = _make_mock_response(fake_vector)

        with pytest.raises(EmbeddingError, match="Dimension mismatch"):
            embed_text("hello", config)

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_connection_error(self, mock_post, config):
        """ConnectionError should be wrapped in EmbeddingError."""
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")

        with pytest.raises(EmbeddingError, match="Cannot connect"):
            embed_text("hello", config)

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_timeout(self, mock_post, config):
        """Timeout should be wrapped in EmbeddingError."""
        import requests

        mock_post.side_effect = requests.Timeout("timed out")

        with pytest.raises(EmbeddingError, match="timed out"):
            embed_text("hello", config)

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_bad_response_body(self, mock_post, config):
        """Missing 'embeddings' key should raise EmbeddingError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with pytest.raises(EmbeddingError, match="Empty embedding"):
            embed_text("hello", config)

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_sends_correct_payload(self, mock_post, config):
        """Verify the request payload matches the Ollama /api/embed contract."""
        fake_vector = [0.1] * 768
        mock_post.return_value = _make_mock_response(fake_vector)

        embed_text("test input", config)

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {
            "model": "nomic-embed-text",
            "input": "test input",
        }
        assert config.ollama_url + "/api/embed" in call_kwargs[0][0]


class TestEmbedChunks:
    """Tests for the embed_chunks function."""

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_returns_embedded_chunks(self, mock_post, config):
        """embed_chunks should return a list of EmbeddedChunk objects."""
        fake_vector = [0.1] * 768
        mock_post.return_value = _make_mock_response(fake_vector)

        chunks = [
            Chunk(
                text="First chunk.",
                token_count=3,
                source="test.txt",
                chunk_index=0,
                metadata={"key": "val"},
            ),
            Chunk(
                text="Second chunk.",
                token_count=3,
                source="test.txt",
                chunk_index=1,
            ),
        ]

        result = embed_chunks(chunks, config)

        assert len(result) == 2
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert result[0].text == "First chunk."
        assert result[0].source == "test.txt"
        assert result[0].chunk_index == 0
        assert result[0].metadata == {"key": "val"}
        assert len(result[0].embedding) == 768
        assert result[1].chunk_index == 1

    @patch("rag_pipeline.vectorstore.embedder.requests.post")
    def test_empty_list(self, mock_post, config):
        """Empty chunk list should return empty result without HTTP calls."""
        result = embed_chunks([], config)

        assert result == []
        mock_post.assert_not_called()
