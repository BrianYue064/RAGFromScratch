"""Tests for the RAG query orchestrator (all external calls mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from rag_pipeline.config import Config
from rag_pipeline.models import GenerationError
from rag_pipeline.generation.query import rag_query


@pytest.fixture
def config():
    return Config(
        embedding_dim=768,
        top_k=5,
        llm_provider="ollama",
        llm_model="qwen3:8b",
    )


def _fake_vector(dim=768):
    return [0.1] * dim


def _fake_chunks(count=3):
    return [
        {
            "chunk_id": f"id-{i}",
            "text": f"Chunk {i} content.",
            "source": "doc.pdf",
            "chunk_index": i,
            "metadata": {},
            "similarity": round(0.9 - i * 0.05, 3),
        }
        for i in range(count)
    ]


class TestRagQuery:
    @patch("rag_pipeline.generation.query.embed_text")
    @patch("rag_pipeline.generation.query.PgVectorStore")
    @patch("rag_pipeline.generation.query.generate")
    def test_rag_query_returns_complete_result(
        self, mock_generate, mock_store_cls, mock_embed_text, config
    ):
        """All keys present in result dict."""
        mock_embed_text.return_value = _fake_vector()
        mock_store = MagicMock()
        mock_store.retrieve.return_value = _fake_chunks(2)
        mock_store_cls.return_value = mock_store
        mock_generate.return_value = "Generated answer."

        result = rag_query("What is RAG?", config)

        assert result["question"] == "What is RAG?"
        assert result["answer"] == "Generated answer."
        assert len(result["sources"]) == 2
        assert result["model"] == "qwen3:8b"
        assert result["provider"] == "ollama"
        mock_store.close.assert_called_once()

    @patch("rag_pipeline.generation.query.embed_text")
    @patch("rag_pipeline.generation.query.PgVectorStore")
    @patch("rag_pipeline.generation.query.generate")
    def test_rag_query_passes_top_k(
        self, mock_generate, mock_store_cls, mock_embed_text, config
    ):
        """top_k from config is forwarded to retrieve."""
        mock_embed_text.return_value = _fake_vector()
        mock_store = MagicMock()
        mock_store.retrieve.return_value = _fake_chunks(7)
        mock_store_cls.return_value = mock_store
        mock_generate.return_value = "Answer."

        config.top_k = 7
        rag_query("Test?", config)

        mock_store.retrieve.assert_called_once()
        _, kwargs = mock_store.retrieve.call_args
        assert kwargs["top_k"] == 7

    @patch("rag_pipeline.generation.query.embed_text")
    @patch("rag_pipeline.generation.query.PgVectorStore")
    @patch("rag_pipeline.generation.query.generate")
    def test_rag_query_generation_error_propagates(
        self, mock_generate, mock_store_cls, mock_embed_text, config
    ):
        """GenerationError from LLM client bubbles up."""
        mock_embed_text.return_value = _fake_vector()
        mock_store = MagicMock()
        mock_store.retrieve.return_value = _fake_chunks(1)
        mock_store_cls.return_value = mock_store
        mock_generate.side_effect = GenerationError("LLM down")

        with pytest.raises(GenerationError, match="LLM down"):
            rag_query("Test?", config)

        mock_store.close.assert_called_once()

    @patch("rag_pipeline.generation.query.embed_text")
    @patch("rag_pipeline.generation.query.PgVectorStore")
    @patch("rag_pipeline.generation.query.generate")
    def test_rag_query_empty_retrieval(
        self, mock_generate, mock_store_cls, mock_embed_text, config
    ):
        """Zero retrieved chunks still generates (with empty context)."""
        mock_embed_text.return_value = _fake_vector()
        mock_store = MagicMock()
        mock_store.retrieve.return_value = []
        mock_store_cls.return_value = mock_store
        mock_generate.return_value = "I don't know."

        result = rag_query("Test?", config)

        assert result["sources"] == []
        assert result["answer"] == "I don't know."
        mock_store.close.assert_called_once()
