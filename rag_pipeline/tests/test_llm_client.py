"""Tests for the LLM client (unit tests with mocked HTTP)."""

import pytest
from unittest.mock import patch, MagicMock

from rag_pipeline.config import Config
from rag_pipeline.models import GenerationError
from rag_pipeline.generation.llm_client import generate


@pytest.fixture
def config():
    return Config(
        llm_provider="ollama",
        llm_model="qwen3:8b",
        llm_temperature=0.7,
        llm_max_tokens=1024,
    )


@pytest.fixture
def lmstudio_config():
    return Config(
        llm_provider="lmstudio",
        llm_model="local-model",
        lmstudio_url="http://localhost:1234",
        llm_temperature=0.7,
        llm_max_tokens=1024,
    )


def _mock_ollama_response(text: str):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": text}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _mock_lmstudio_response(text: str):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestGenerateOllama:
    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_ollama_success(self, mock_post, config):
        """Mock Ollama response returns generated text."""
        mock_post.return_value = _mock_ollama_response("Paris is the capital.")
        result = generate("What is the capital?", config)
        assert result == "Paris is the capital."

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_ollama_connection_error(self, mock_post, config):
        """ConnectionError raises GenerationError."""
        import httpx

        mock_post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(GenerationError, match="Cannot connect to Ollama"):
            generate("test", config)

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_ollama_timeout(self, mock_post, config):
        """Timeout raises GenerationError."""
        import httpx

        mock_post.side_effect = httpx.TimeoutException("timed out")
        with pytest.raises(GenerationError, match="timed out"):
            generate("test", config)

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_ollama_empty_response(self, mock_post, config):
        """Missing 'response' key raises GenerationError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        with pytest.raises(GenerationError, match="missing 'response'"):
            generate("test", config)

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_sends_correct_ollama_payload(self, mock_post, config):
        """Verify request body matches Ollama API contract."""
        mock_post.return_value = _mock_ollama_response("answer")
        generate("my prompt", config)
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {
            "model": "qwen3:8b",
            "prompt": "my prompt",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            },
        }
        assert config.ollama_url + "/api/generate" in call_kwargs[0][0]


class TestGenerateLMStudio:
    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_lmstudio_success(self, mock_post, lmstudio_config):
        """Mock LM Studio response returns generated text."""
        mock_post.return_value = _mock_lmstudio_response("Paris is capital.")
        result = generate("What is capital?", lmstudio_config)
        assert result == "Paris is capital."

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_lmstudio_connection_error(
        self, mock_post, lmstudio_config
    ):
        """ConnectionError raises GenerationError."""
        import httpx

        mock_post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(
            GenerationError, match="Cannot connect to LM Studio"
        ):
            generate("test", lmstudio_config)

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_lmstudio_empty_choices(
        self, mock_post, lmstudio_config
    ):
        """Missing choices raises GenerationError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "invalid request"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        with pytest.raises(
            GenerationError, match="missing expected structure"
        ):
            generate("test", lmstudio_config)

    @patch("rag_pipeline.generation.llm_client.httpx.post")
    def test_generate_sends_correct_lmstudio_payload(
        self, mock_post, lmstudio_config
    ):
        """Verify request body matches OpenAI API contract."""
        mock_post.return_value = _mock_lmstudio_response("answer")
        generate("my prompt", lmstudio_config)
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {
            "model": "local-model",
            "messages": [{"role": "user", "content": "my prompt"}],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
        }
        assert "http://localhost:1234/v1/chat/completions" in call_kwargs[0][0]


class TestGenerateInvalidProvider:
    def test_generate_invalid_provider(self):
        """Unknown provider string raises GenerationError."""
        bad_config = Config(llm_provider="nonexistent")
        with pytest.raises(
            GenerationError, match="Unknown LLM provider"
        ):
            generate("test", bad_config)
