"""Parsers package for different file types."""

from .html_parser import parse_html_file, parse_html_url
from .text_parser import parse_text_file

__all__ = ["parse_html_file", "parse_html_url", "parse_text_file"]