"""Model discovery via OpenRouter API — simplified for fixed candidate list."""

import logging

import requests

from .config import OPENROUTER_BASE_URL, CANDIDATE_MODELS, load_api_key

logger = logging.getLogger(__name__)


def fetch_available_models() -> list[dict]:
    """
    Fetch all available models from OpenRouter API.

    Returns:
        List of model dicts with id, name, pricing, context_length
    """
    try:
        api_key = load_api_key()
        resp = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        logger.error(f"Failed to fetch models from OpenRouter: {e}")
        return []


def discover_candidate_models() -> list[dict]:
    """
    Return configured candidate models enriched with OpenRouter metadata.

    Reads CANDIDATE_MODELS from config (no LLM call needed — models are fixed).
    Enriches with live metadata (name, context length) from OpenRouter API.

    Returns:
        List of candidate model dicts with id, name, context_length, pricing
    """
    logger.info(f"Loading {len(CANDIDATE_MODELS)} configured candidate models...")

    # Fetch live model list for metadata enrichment
    available_models = fetch_available_models()
    available_ids = {m["id"]: m for m in available_models}

    candidates = []
    for model_id in CANDIDATE_MODELS:
        if model_id in available_ids:
            model_info = available_ids[model_id]
            candidates.append({
                "id": model_id,
                "name": model_info.get("name", model_id),
                "context_length": model_info.get("context_length", 8192),
                "pricing": model_info.get("pricing", {}),
            })
        else:
            # Model not available live, still include with basic info
            logger.warning(f"Model {model_id} not found in live list, including anyway")
            candidates.append({
                "id": model_id,
                "name": model_id.split("/")[-1].replace("-", " ").title(),
                "context_length": 8192,
                "pricing": {},
            })

    logger.info(f"Loaded {len(candidates)} candidate models")
    return candidates
