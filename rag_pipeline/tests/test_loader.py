"""Tests for the document loader."""

import logging
from pathlib import Path

import pytest

from ingestion.loader import load
from ingestion.models import Document, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_txt_file():
    """Test loading a .txt file returns a Document with correct file_type."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    assert isinstance(doc, Document)
    assert doc.file_type in ("text", "md")
    assert doc.content
    assert len(doc.content) > 0


def test_load_pdf_file():
    """Test loading a .pdf file returns a Document with markdown content."""
    pdf_path = FIXTURES_DIR / "sample.pdf"
    doc = load(pdf_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "pdf"
    assert doc.content is not None
    assert len(doc.content) > 0


def test_load_unsupported_extension():
    """Test that an unsupported extension raises UnsupportedFileTypeError."""
    unsupported_path = FIXTURES_DIR / "sample.xyz"

    with pytest.raises(UnsupportedFileTypeError):
        load(unsupported_path)


def test_metadata_fields():
    """Test that required metadata keys are present."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    assert "file_size_bytes" in doc.metadata
    assert "last_modified" in doc.metadata
    assert "num_characters" in doc.metadata

    assert isinstance(doc.metadata["file_size_bytes"], int)
    assert isinstance(doc.metadata["last_modified"], str)
    assert isinstance(doc.metadata["num_characters"], int)


def test_load_html_file():
    """Test loading an .html file returns Document with content."""
    html_path = FIXTURES_DIR / "sample.html"
    doc = load(html_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "html"
    assert doc.content
    assert len(doc.content) > 0