"""OpenRouter API client for LLM interactions."""

import time
import logging
import sys
from typing import Optional
from openai import OpenAI

from .config import OPENROUTER_BASE_URL, load_api_key

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_client() -> OpenAI:
    """Create and return an OpenAI-compatible OpenRouter client."""
    api_key = load_api_key()
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 60,
) -> tuple[str, float]:
    """
    Call an LLM via OpenRouter and return (response_text, latency_seconds).

    Args:
        model: OpenRouter model ID (e.g. 'google/gemini-3.1-pro-preview')
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Tuple of (response_text, latency_in_seconds)
    """
    client = get_client()
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_headers={
                "HTTP-Referer": "https://llm-fitness-tool",
                "X-Title": "LLM Fitness Tool",
            },
        )
        latency = time.time() - start
        # Guard against None choices or content
        if not response.choices or response.choices[0].message is None:
            return "", latency
        content = response.choices[0].message.content or ""
        return content, latency
    except Exception as e:
        latency = time.time() - start
        logger.error(f"Error calling {model}: {e}")
        return f"ERROR: {str(e)}", latency


def call_judge(messages: list[dict], temperature: float = 0.3, max_tokens: int = 8192) -> str:
    """
    Call the Judge LLM (Gemini 3.1 Pro) and return response text.

    Args:
        messages: List of message dicts
        temperature: Low temperature for consistent judging
        max_tokens: Maximum tokens

    Returns:
        Response text from the judge
    """
    from .config import JUDGE_MODEL_ID
    response, _ = call_llm(
        model=JUDGE_MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )
    return response
