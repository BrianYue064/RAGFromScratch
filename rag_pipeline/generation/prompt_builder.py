"""Prompt builder for assembling LLM prompts from retrieved chunks."""

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question based only on the following context. "
    "If the context does not contain enough information, "
    'say "I don\'t have enough information to answer that."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_CHUNK_SEPARATOR = "\n\n---\n\n"


def build_prompt(
    question: str, chunks: list[dict], template: str | None = None
) -> str:
    """Assemble an LLM prompt from a question and retrieved chunks.

    Args:
        question: The user's question.
        chunks: List of retrieval result dicts (as returned by
            PgVectorStore.retrieve()). Each dict must have a 'text' key.
        template: Optional f-string template with {context} and {question}
            placeholders. Falls back to DEFAULT_PROMPT_TEMPLATE if None.

    Returns:
        The assembled prompt string.
    """
    context = _CHUNK_SEPARATOR.join(c["text"] for c in chunks)
    tpl = template if template is not None else DEFAULT_PROMPT_TEMPLATE
    return tpl.format(context=context, question=question)
