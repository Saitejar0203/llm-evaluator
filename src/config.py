"""Configuration module for the PM's LLM Evaluator."""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL_ID = "google/gemini-3.1-pro-preview"
GENERATOR_MODEL_ID = "google/gemini-3-flash-preview"

# Credential path
OPENROUTER_CONFIG_PATH = Path.home() / ".config" / "openrouter" / "config"


def load_api_key() -> str:
    """Load OpenRouter API key from config file or environment variable."""
    # Try environment variable first
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        return api_key

    # Try config file
    if OPENROUTER_CONFIG_PATH.exists():
        try:
            with open(OPENROUTER_CONFIG_PATH) as f:
                config = json.load(f)
                api_key = config.get("api_key", "")
                if api_key:
                    return api_key
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Failed to parse OpenRouter config: {e}")

    raise ValueError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY env var or "
        f"create {OPENROUTER_CONFIG_PATH} with {{\"api_key\": \"your-key\"}}"
    )


# Our candidate models — selected for price parity (~$0.20-0.26/M input)
CANDIDATE_MODELS = [
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5-mini",
    "qwen/qwen3.5-122b-a10b",
    "minimax/minimax-m2.5",
    "inception/mercury-2",
    "mistralai/ministral-14b-2512",
]

# Per-model pricing ($/M tokens)
MODEL_PRICING = {
    "google/gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "openai/gpt-5-mini":                    {"input": 0.25, "output": 2.00},
    "qwen/qwen3.5-122b-a10b":              {"input": 0.26, "output": 2.08},
    "minimax/minimax-m2.5":                 {"input": 0.25, "output": 1.20},
    "inception/mercury-2":                  {"input": 0.25, "output": 0.75},
    "mistralai/ministral-14b-2512":         {"input": 0.20, "output": 0.20},
}

# PM-centric evaluation dimensions
EVAL_DIMENSIONS = [
    "accuracy", "hallucination_resistance", "faithfulness",
    "abstention", "tool_calling",
]

# Tool calling settings
MAX_TOOL_CALLS = 3  # cap per question to prevent runaway loops

# Benchmarking settings
MAX_CANDIDATES = 6
MAX_TEST_CASES = 10
REQUEST_TIMEOUT = 60  # seconds
