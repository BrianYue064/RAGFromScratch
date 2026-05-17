"""Tests for the document chunker."""

import logging
from pathlib import Path

import pytest
import tiktoken

from ingestion.chunker import _split_at_token_level, chunk
from ingestion.loader import load
from ingestion.models import Chunk

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ENCODING = tiktoken.get_encoding("cl100k_base")


def test_chunk_output_not_empty():
    """Test that chunking returns a non-empty list."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    chunks = chunk(doc, chunk_size=50, overlap=10)

    assert chunks
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_token_count_within_limit():
    """Test that no chunk exceeds chunk_size + tolerance.

    The +10 tolerance accounts for subword token boundary artifacts
    when a sentence boundary falls mid-token during overlap decoding.
    """
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    chunk_size = 50
    tolerance = 10
    chunks = chunk(doc, chunk_size=chunk_size, overlap=10)

    for i, ch in enumerate(chunks):
        tokens = ENCODING.encode(ch.text)
        token_count = len(tokens)
        assert token_count <= chunk_size + tolerance, (
            f"Chunk {i} exceeds limit: {token_count} > {chunk_size + tolerance}"
        )


def test_chunk_long_document():
    """Test chunking a longer document produces multiple well-sized chunks."""
    txt_path = FIXTURES_DIR / "sample_long.txt"
    doc = load(txt_path)

    chunk_size = 50
    chunks = chunk(doc, chunk_size=chunk_size, overlap=10)

    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    for i, ch in enumerate(chunks):
        tokens = ENCODING.encode(ch.text)
        token_count = len(tokens)
        tolerance = 10
        assert token_count <= chunk_size + tolerance, (
            f"Chunk {i} exceeds limit: {token_count} > {chunk_size + tolerance}"
        )


def test_chunk_index_sequential():
    """Test that chunk_index values are sequential starting from 0."""
    txt_path = FIXTURES_DIR / "sample_long.txt"
    doc = load(txt_path)

    chunks = chunk(doc, chunk_size=50, overlap=10)

    for i, ch in enumerate(chunks):
        assert ch.chunk_index == i, f"Expected chunk_index {i}, got {ch.chunk_index}"


def test_chunk_overlap():
    """Test that the last ~N tokens of chunk i appear at the start of chunk i+1.

    Verifies overlap at the text level (rather than raw token IDs) to account
    for the round-trip through decode().strip() which may shift subword token
    boundaries.
    """
    txt_path = FIXTURES_DIR / "sample_long.txt"
    doc = load(txt_path)

    overlap = 10
    chunks = chunk(doc, chunk_size=50, overlap=overlap)

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap")

    for i in range(len(chunks) - 1):
        tokens_a = ENCODING.encode(chunks[i].text)

        actual_overlap = min(overlap, len(tokens_a))
        if actual_overlap == 0:
            continue

        overlap_text = ENCODING.decode(tokens_a[-actual_overlap:]).strip()

        assert chunks[i + 1].text.startswith(overlap_text), (
            f"Overlap text from chunk {i} not found at start of chunk {i + 1}: "
            f"{overlap_text!r} vs {chunks[i + 1].text[:60]!r}"
        )


def test_chunk_metadata_inherited():
    """Test that Chunk inherits metadata from Document."""
    txt_path = FIXTURES_DIR / "sample_long.txt"
    doc = load(txt_path)

    chunks = chunk(doc, chunk_size=50, overlap=10)

    for ch in chunks:
        assert ch.source == doc.source
        assert "num_characters" in ch.metadata


def test_chunk_empty_document():
    """Test chunking a document with no content returns empty list."""
    from ingestion.models import Document

    doc = Document(content="", source="empty.txt", file_type="text")
    chunks = chunk(doc, chunk_size=50, overlap=10)

    assert chunks == []


def test_chunk_single_sentence():
    """Test chunking a short document produces one chunk."""
    from ingestion.models import Document

    doc = Document(
        content="This is a single short sentence.", source="short.txt", file_type="text"
    )
    chunks = chunk(doc, chunk_size=50, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "short.txt"


def test_split_at_token_level():
    """Test _split_at_token_level splits tokens correctly."""
    tokens = list(range(100))
    chunk_size = 30
    overlap = 5

    result = _split_at_token_level(tokens, chunk_size, overlap)

    assert len(result) == 4
    assert result[0] == list(range(0, 30))
    assert result[1] == list(range(25, 55))
    assert result[2] == list(range(50, 80))
    assert result[3] == list(range(75, 100))


def test_split_at_token_level_no_overlap():
    """Test _split_at_token_level with zero overlap."""
    tokens = list(range(100))
    chunk_size = 30
    overlap = 0

    result = _split_at_token_level(tokens, chunk_size, overlap)

    assert len(result) == 4
    assert result[0] == list(range(0, 30))
    assert result[1] == list(range(30, 60))
    assert result[2] == list(range(60, 90))
    assert result[3] == list(range(90, 100))
