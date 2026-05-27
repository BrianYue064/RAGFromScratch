"""Tests for the prompt builder (no external dependencies)."""

import pytest

from rag_pipeline.generation.prompt_builder import (
    build_prompt,
    DEFAULT_PROMPT_TEMPLATE,
)


def _make_chunk(text: str, source: str = "test.txt", similarity: float = 0.9):
    return {"text": text, "source": source, "similarity": similarity}


class TestBuildPrompt:
    def test_build_prompt_default_template(self):
        """Verifies context and question appear in output."""
        chunks = [_make_chunk("Paris is the capital of France.")]
        result = build_prompt("What is the capital?", chunks)
        assert "Paris is the capital of France." in result
        assert "What is the capital?" in result
        assert "Answer the question based only on the following context" in result

    def test_build_prompt_custom_template(self):
        """Custom template with {context} and {question} works."""
        template = "CONTEXT: {context}\nQ: {question}"
        chunks = [_make_chunk("Some context.")]
        result = build_prompt("A question?", chunks, template=template)
        assert result == "CONTEXT: Some context.\nQ: A question?"

    def test_build_prompt_empty_chunks(self):
        """Empty chunk list produces prompt with empty context section."""
        result = build_prompt("Question?", [])
        assert "Question?" in result
        assert result.startswith("Answer the question")

    def test_build_prompt_multiple_chunks(self):
        """Chunks are joined with --- separators."""
        chunks = [
            _make_chunk("First chunk."),
            _make_chunk("Second chunk."),
            _make_chunk("Third chunk."),
        ]
        result = build_prompt("Question?", chunks)
        assert "First chunk." in result
        assert "Second chunk." in result
        assert "Third chunk." in result
        assert "---" in result

    def test_context_ordering_preserved(self):
        """Chunks appear in the same order as input."""
        chunks = [
            _make_chunk("AAA"),
            _make_chunk("BBB"),
            _make_chunk("CCC"),
        ]
        result = build_prompt("Q?", chunks)
        ctx_start = result.index("AAA")
        ctx_mid = result.index("BBB")
        ctx_end = result.index("CCC")
        assert ctx_start < ctx_mid < ctx_end
