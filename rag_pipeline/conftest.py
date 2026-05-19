"""Pytest configuration: ensures rag_pipeline/ is on sys.path."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def docker_compose_file():
    """Point pytest-docker to the test Docker Compose file."""
    return str(Path(__file__).parent / "tests" / "docker-compose.yml")
