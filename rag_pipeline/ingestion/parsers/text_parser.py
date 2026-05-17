"""Parser implementations for plain text files."""

import logging
from pathlib import Path

from ..models import IngestionError

logger = logging.getLogger(__name__)


def parse_text_file(path: Path) -> str:
    """Parse a plain text file and return raw content.

    Reads with UTF-8 encoding, falls back to latin-1 on decode errors.
    Strips BOM characters if present.

    Args:
        path: Path to the text file.

    Returns:
        Raw string content.
    """
    bom = "\ufeff"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.startswith(bom):
                content = content[1:]
            return content
    except UnicodeDecodeError:
        logger.debug(f"UTF-8 decode failed, falling back to latin-1 for {path}")
        try:
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()
                if content.startswith(bom):
                    content = content[1:]
                return content
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Failed to read text file {path}: {e}")
            raise IngestionError(f"Failed to read text file: {path}") from e
