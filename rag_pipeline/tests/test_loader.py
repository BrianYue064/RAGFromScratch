"""Tests for the document loader."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.loader import load
from ingestion.models import Document, IngestionError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_txt_file():
    """Test loading a .txt file returns a Document with correct file_type."""
    txt_path = FIXTURES_DIR / "sample.txt"
    doc = load(txt_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "text"
    assert doc.content
    assert len(doc.content) > 0


def test_load_markdown_file():
    """Test loading a .md file returns a Document with text file_type."""
    md_path = FIXTURES_DIR / "sample.md"
    doc = load(md_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "text"
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


def test_load_docx_file():
    """Test loading a .docx file returns a Document with docx file_type.

    Note: sample.docx fixture must be a valid DOCX file for this test to pass.
    """
    docx_path = FIXTURES_DIR / "sample.docx"
    if docx_path.stat().st_size == 0:
        pytest.skip(
            "sample.docx fixture is empty; provide a valid DOCX to enable this test"
        )

    doc = load(docx_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "docx"
    assert doc.content is not None
    assert len(doc.content) > 0


def test_load_html_file():
    """Test loading an .html file returns Document with content."""
    html_path = FIXTURES_DIR / "sample.html"
    doc = load(html_path)

    assert isinstance(doc, Document)
    assert doc.file_type == "html"
    assert doc.content
    assert len(doc.content) > 0


def test_load_unsupported_extension():
    """Test that an unsupported extension raises UnsupportedFileTypeError."""
    unsupported_path = FIXTURES_DIR / "sample.xyz"

    with pytest.raises(UnsupportedFileTypeError):
        load(unsupported_path)


def test_load_file_not_found():
    """Test that a nonexistent file raises IngestionError."""
    nonexistent = FIXTURES_DIR / "does_not_exist.txt"

    with pytest.raises(IngestionError, match="File not found"):
        load(nonexistent)


def test_load_url():
    """Test loading from a URL returns Document with html file_type."""
    fake_html = "<html><body><p>Hello from the web.</p></body></html>"

    with patch("ingestion.parsers.html_parser.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = fake_html
        mock_response.raise_for_status.return_value = None

        doc = load("https://example.com/test")

    assert isinstance(doc, Document)
    assert doc.file_type == "html"
    assert doc.content
    assert "Hello from the web" in doc.content


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
