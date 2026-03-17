"""OpenRouter API client for LLM interactions."""

import time
import logging
from typing import Optional
from openai import OpenAI

from .config import OPENROUTER_BASE_URL, load_api_key

logger = logging.getLogger(__name__)


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
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = None,
) -> dict:
    """
    Call an LLM via OpenRouter and return a structured response dict.

    Args:
        model: OpenRouter model ID (e.g. 'google/gemini-3.1-pro-preview')
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        tools: Optional list of tool schemas for function calling
        tool_choice: Optional tool choice strategy ("auto", "none", etc.)

    Returns:
        Dict with content, tool_calls, latency, usage, finish_reason
    """
    client = get_client()
    start = time.time()
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_headers={
                "HTTP-Referer": "https://pm-llm-evaluator",
                "X-Title": "PM's LLM Evaluator",
            },
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = client.chat.completions.create(**kwargs)
        latency = time.time() - start

        # Extract usage data
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens or 0
            usage["completion_tokens"] = response.usage.completion_tokens or 0

        # Guard against None choices
        if not response.choices or response.choices[0].message is None:
            return {
                "content": "",
                "tool_calls": None,
                "latency": latency,
                "usage": usage,
                "finish_reason": "error",
            }

        message = response.choices[0].message
        content = message.content or None
        tool_calls = list(message.tool_calls) if message.tool_calls else None
        finish_reason = response.choices[0].finish_reason or "stop"

        return {
            "content": content,
            "tool_calls": tool_calls,
            "latency": latency,
            "usage": usage,
            "finish_reason": finish_reason,
        }
    except Exception as e:
        latency = time.time() - start
        logger.error(f"Error calling {model}: {e}")
        return {
            "content": f"ERROR: {str(e)}",
            "tool_calls": None,
            "latency": latency,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "finish_reason": "error",
        }


def call_generator(
    messages: list[dict],
    thinking_level: str = "minimal",
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """
    Call the Generator LLM (Gemini 3 Flash) with thinking enabled.

    Args:
        messages: List of message dicts
        thinking_level: "minimal" (budget_tokens=1024) or "medium" (budget_tokens=4096)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Response text from the generator
    """
    from .config import GENERATOR_MODEL_ID

    budget_tokens = 1024 if thinking_level == "minimal" else 4096

    client = get_client()
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=GENERATOR_MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            extra_headers={
                "HTTP-Referer": "https://pm-llm-evaluator",
                "X-Title": "PM's LLM Evaluator",
            },
            extra_body={
                "thinking": {"type": "enabled", "budget_tokens": budget_tokens}
            },
        )
        latency = time.time() - start
        logger.debug(
            f"Generator ({thinking_level} thinking) responded in {latency:.2f}s"
        )

        if not response.choices or response.choices[0].message is None:
            return ""
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling generator: {e}")
        return f"ERROR: {str(e)}"


def call_judge(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> str:
    """
    Call the Judge LLM (Gemini 3.1 Pro) with medium thinking and return response text.

    Args:
        messages: List of message dicts
        temperature: Low temperature for consistent judging
        max_tokens: Maximum tokens

    Returns:
        Response text from the judge
    """
    from .config import JUDGE_MODEL_ID

    client = get_client()
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
            extra_headers={
                "HTTP-Referer": "https://pm-llm-evaluator",
                "X-Title": "PM's LLM Evaluator",
            },
            extra_body={
                "thinking": {"type": "enabled", "budget_tokens": 4096}
            },
        )
        latency = time.time() - start
        logger.debug(f"Judge responded in {latency:.2f}s")

        if not response.choices or response.choices[0].message is None:
            return ""
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling judge: {e}")
        return f"ERROR: {str(e)}"
