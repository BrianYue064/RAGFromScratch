"""Parser implementations for HTML content."""

import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from ..models import IngestionError

logger = logging.getLogger(__name__)

HTML_REQUEST_TIMEOUT = 10


def parse_html_file(path: Path) -> str:
    """Parse a local HTML file and extract clean text content.

    Args:
        path: Path to the HTML file.

    Returns:
        Extracted text content as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return _extract_text_from_html(html_content)


def parse_html_url(url: str) -> str:
    """Fetch and parse a live HTML URL and extract clean text content.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted text content as a string.

    Raises:
        IngestionError: If the request fails or times out.
    """
    try:
        response = requests.get(url, timeout=HTML_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.Timeout as e:
        logger.error(f"Request timeout for URL {url}: {e}")
        raise IngestionError(f"Request timeout for {url}") from e
    except requests.RequestException as e:
        logger.error(f"Request failed for URL {url}: {e}")
        raise IngestionError(f"Request failed for {url}") from e

    return _extract_text_from_html(response.text)


def _extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content.

    Strips script, style, nav, footer, header, and aside tags.
    Prefers main or article tags, falls back to body.

    Args:
        html_content: Raw HTML string.

    Returns:
        Cleaned text content.
    """
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    content_container: Optional[BeautifulSoup] = None
    for selector in ["main", "article"]:
        container = soup.select_one(selector)
        if container:
            content_container = container
            break

    if not content_container:
        content_container = soup.body

    if not content_container:
        logger.warning("No main, article, or body tag found in HTML")
        return soup.get_text(separator="\n", strip=True)

    paragraphs = content_container.find_all("p")
    if paragraphs:
        text_parts = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    text = content_container.get_text(separator="\n", strip=True)
    return text