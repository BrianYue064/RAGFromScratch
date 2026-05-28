"""Path resolution for multi-file and folder ingestion.

Expands user-supplied paths into an ordered list of concrete file paths,
handling files, folders (one level deep), and URLs.
"""

import logging
from pathlib import Path

from .loader import SUPPORTED_EXTENSIONS
from .models import IngestionError

logger = logging.getLogger(__name__)


def resolve_paths(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []

    for raw in paths:
        if raw.startswith("http://") or raw.startswith("https://"):
            resolved.append(Path(raw))
            continue

        p = Path(raw)

        if p.is_dir():
            dir_resolved = _resolve_directory(p)
            resolved.extend(dir_resolved)
        elif p.exists():
            resolved.append(p.resolve())
        else:
            raise IngestionError(f"Path does not exist: {raw}")

    return _dedup(resolved)


def _resolve_directory(dir_path: Path) -> list[Path]:
    resolved: list[Path] = []

    for child in sorted(dir_path.iterdir()):
        if child.is_file():
            suffix = child.suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS:
                resolved.append(child.resolve())
            else:
                logger.warning(
                    "Skipping unsupported file: %s (supported: %s)",
                    child,
                    list(SUPPORTED_EXTENSIONS.keys()),
                )

    return resolved


def _dedup(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result
