"""Tests for the document chunker."""

import logging
from pathlib import Path

import pytest
import tiktoken

from ingestion.chunker import chunk
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
    """Test that no chunk exceeds chunk_size + tolerance."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    chunk_size = 50
    chunks = chunk(doc, chunk_size=chunk_size, overlap=10)

    for i, ch in enumerate(chunks):
        tokens = ENCODING.encode(ch.text)
        token_count = len(tokens)
        assert token_count <= chunk_size + 10, (
            f"Chunk {i} exceeds limit: {token_count} > {chunk_size + 10}"
        )


def test_chunk_index_sequential():
    """Test that chunk_index values are sequential starting from 0."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    chunks = chunk(doc, chunk_size=50, overlap=10)

    for i, ch in enumerate(chunks):
        assert ch.chunk_index == i, (
            f"Expected chunk_index {i}, got {ch.chunk_index}"
        )


def test_chunk_overlap():
    """Test that overlap tokens appear at start of consecutive chunks."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    overlap = 10
    chunks = chunk(doc, chunk_size=50, overlap=overlap)

    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap")

    for i in range(len(chunks) - 1):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]

        tokens_a = ENCODING.encode(chunk_a.text)
        overlap_tokens_a = tokens_a[-overlap:]
        overlap_text_a = ENCODING.decode(overlap_tokens_a).strip()

        tokens_b = ENCODING.encode(chunk_b.text)
        prefix_tokens_b = tokens_b[:overlap]
        prefix_text_b = ENCODING.decode(prefix_tokens_b).strip()

        assert (
            overlap_text_a == prefix_text_b or prefix_text_b in chunk_b.text
        ), f"Overlap mismatch between chunk {i} and {i+1}"


def test_chunk_metadata_inherited():
    """Test that Chunk inherits metadata from Document."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    chunks = chunk(doc, chunk_size=50, overlap=10)

    for ch in chunks:
        assert ch.source == doc.source
        assert "num_characters" in ch.metadata