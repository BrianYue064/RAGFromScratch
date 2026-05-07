"""Document loader for various file types."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from docling.document_converter import DocumentConverter

from .models import Document, IngestionError, UnsupportedFileTypeError
from .parsers import (
    parse_html_file,
    parse_html_url,
    parse_text_file,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".md": "text",
}


def load(path: Union[str, Path]) -> Document:
    """Load a document from a file path or URL.

    Automatically detects file type and uses the appropriate parser.

    Args:
        path: Absolute file path or HTTP URL.

    Returns:
        A Document instance with content and metadata.

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
        IngestionError: If loading fails.
    """
    path_str = str(path)

    if path_str.startswith("http://") or path_str.startswith("https://"):
        logger.info(f"Loading document from URL: {path}")
        return _load_from_url(path_str)

    file_path = Path(path).resolve()
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    file_type = _detect_file_type(file_path)
    logger.info(f"Loading document from file: {file_path} (type: {file_type})")

    content = _load_file(file_path, file_type)

    metadata = {
        "file_size_bytes": file_path.stat().st_size,
        "last_modified": datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "num_characters": len(content),
    }

    return Document(
        content=content,
        source=str(file_path),
        file_type=file_type,
        metadata=metadata,
    )


def _detect_file_type(path: Path) -> str:
    """Detect file type from path suffix."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix}. Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    return SUPPORTED_EXTENSIONS[suffix]


def _load_file(path: Path, file_type: str) -> str:
    """Load content from a file based on its type."""
    if file_type in ("pdf", "docx"):
        return _load_via_docling(path, file_type)
    elif file_type == "html":
        return parse_html_file(path)
    elif file_type == "text":
        return parse_text_file(path)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_type}")


def _load_via_docling(path: Path, file_type: str) -> str:
    """Load PDF or DOCX using Docling and convert to markdown."""
    converter = DocumentConverter()

    try:
        if file_type == "pdf":
            result = converter.convert(path)
        elif file_type == "docx":
            result = converter.convert(path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported docling type: {file_type}")

        return result.document.export_to_markdown()

    except Exception as e:
        logger.error(f"Docling conversion failed for {path}: {e}")
        raise IngestionError(f"Failed to convert {file_type} file: {path}") from e


def _load_from_url(url: str) -> Document:
    """Load content from a URL using HTTP GET."""
    content = parse_html_url(url)

    metadata = {
        "num_characters": len(content),
    }

    return Document(
        content=content,
        source=url,
        file_type="html",
        metadata=metadata,
    )