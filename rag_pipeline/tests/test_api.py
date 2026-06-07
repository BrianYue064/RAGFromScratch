"""Tests for the FastAPI application endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Create a test client with mocked configuration and database storage."""
    with patch("rag_pipeline.api.app.PgVectorStore") as mock_store_cls, \
         patch("rag_pipeline.api.app.load_config") as mock_load_config:
        
        mock_config = MagicMock()
        mock_config.ollama_url = "http://localhost:11434"
        mock_config.embedding_dim = 384
        mock_config.embedding_model = "nomic-embed-text"
        mock_load_config.return_value = mock_config

        mock_store = MagicMock()
        mock_store.count.return_value = 10
        mock_store_cls.return_value = mock_store

        from rag_pipeline.api.app import app
        with TestClient(app) as client:
            yield client


def test_root_redirect(client):
    """Test that visiting root (/) redirects to /docs."""
    # follow_redirects=False lets us assert the 307 Temporary Redirect status and headers
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


@patch("rag_pipeline.api.app.httpx.Client")
def test_health_check_healthy(mock_httpx_cls, client):
    """Test that the /health endpoint returns healthy when all services are responsive."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_client.get.return_value = mock_response
    mock_httpx_cls.return_value.__enter__.return_value = mock_client

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["ollama"] == "reachable"


@patch("rag_pipeline.api.app.httpx.Client")
def test_health_check_degraded(mock_httpx_cls, client):
    """Test that the /health endpoint returns degraded when Ollama is unreachable."""
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("Connection refused")
    mock_httpx_cls.return_value.__enter__.return_value = mock_client

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["database"] == "connected"
    assert "error" in data["ollama"]
