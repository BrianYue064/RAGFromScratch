"""HTTP client for local LLM inference.

Supports Ollama (/api/generate) and LM Studio (OpenAI-compatible
/v1/chat/completions) behind a unified generate() function.
"""

import logging
import time

import requests

from rag_pipeline.config import Config
from rag_pipeline.models import GenerationError

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 120


def generate(prompt: str, config: Config) -> str:
    """Send a prompt to the configured LLM provider and return the answer.

    Args:
        prompt: The full prompt string (already assembled by prompt_builder).
        config: Pipeline configuration with llm_provider, llm_model, etc.

    Returns:
        The generated text string.

    Raises:
        GenerationError: On connection failures, timeouts, or unexpected
            responses.
    """
    if config.llm_provider == "ollama":
        return _generate_ollama(prompt, config)
    elif config.llm_provider == "lmstudio":
        return _generate_lmstudio(prompt, config)
    else:
        raise GenerationError(
            f"Unknown LLM provider: {config.llm_provider!r}. "
            "Expected 'ollama' or 'lmstudio'."
        )


def _generate_ollama(prompt: str, config: Config) -> str:
    """Generate via Ollama /api/generate."""
    url = f"{config.ollama_url}/api/generate"
    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.llm_temperature,
            "num_predict": config.llm_max_tokens,
        },
    }

    try:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
    except requests.ConnectionError as e:
        raise GenerationError(
            f"Cannot connect to Ollama at {config.ollama_url}"
        ) from e
    except requests.Timeout as e:
        raise GenerationError(
            f"Ollama request timed out after {_REQUEST_TIMEOUT}s"
        ) from e
    except requests.RequestException as e:
        raise GenerationError(
            f"Ollama generation request failed: {e}"
        ) from e

    data = response.json()
    answer = data.get("response")
    if not answer:
        raise GenerationError(
            f"Ollama response missing 'response' key: {data}"
        )

    logger.info(
        "Ollama generate | model=%s | prompt_len=%d | answer_len=%d | "
        "latency=%.0fms",
        config.llm_model,
        len(prompt),
        len(answer),
        elapsed_ms,
    )
    return answer


def _generate_lmstudio(prompt: str, config: Config) -> str:
    """Generate via LM Studio (OpenAI-compatible) /v1/chat/completions."""
    url = f"{config.lmstudio_url}/v1/chat/completions"
    payload = {
        "model": config.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.llm_temperature,
        "max_tokens": config.llm_max_tokens,
        "stream": False,
    }

    try:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
    except requests.ConnectionError as e:
        raise GenerationError(
            f"Cannot connect to LM Studio at {config.lmstudio_url}"
        ) from e
    except requests.Timeout as e:
        raise GenerationError(
            f"LM Studio request timed out after {_REQUEST_TIMEOUT}s"
        ) from e
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else "no body"
        raise GenerationError(
            f"LM Studio generation request failed: {e}\n"
            f"Response body: {body}"
        ) from e
    except requests.RequestException as e:
        raise GenerationError(
            f"LM Studio generation request failed: {e}"
        ) from e

    data = response.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise GenerationError(
            f"LM Studio response missing expected structure: {data}"
        ) from e

    logger.info(
        "LM Studio generate | model=%s | prompt_len=%d | answer_len=%d | "
        "latency=%.0fms",
        config.llm_model,
        len(prompt),
        len(answer),
        elapsed_ms,
    )
    return answer
