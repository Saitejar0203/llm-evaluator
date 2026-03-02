"""Dynamic model discovery via OpenRouter API."""

import json
import logging
import sys
import re
from typing import Optional

import requests

from .config import OPENROUTER_BASE_URL, MODEL_CATEGORIES, MAX_CANDIDATES, load_api_key
from .openrouter_client import call_judge

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


CATEGORY_DETECTION_PROMPT = """You are an AI task classifier. Given a task description, identify the primary category.

Task description: {task_description}

Choose ONE category from this list:
- coding (software engineering, programming, debugging, code review)
- math (mathematics, statistics, calculations, proofs)
- reasoning (logic, analysis, planning, problem-solving)
- conversation (chatbots, customer service, dialogue, Q&A)
- writing (content creation, summarization, translation, editing)
- general (anything that doesn't fit the above)

Return ONLY the category name, nothing else."""


def detect_task_category(task_description: str) -> str:
    """
    Detect the task category using the Judge LLM.

    Args:
        task_description: Natural language task description

    Returns:
        Category string (e.g. 'coding', 'math', etc.)
    """
    messages = [
        {"role": "user", "content": CATEGORY_DETECTION_PROMPT.format(task_description=task_description)}
    ]
    response = call_judge(messages, temperature=0.1, max_tokens=20)
    category = response.strip().lower().split()[0] if response.strip() else "general"

    # Validate against known categories
    valid = set(MODEL_CATEGORIES.keys())
    if category not in valid:
        # Try partial match
        for cat in valid:
            if cat in response.lower():
                return cat
        return "general"
    return category


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


def discover_candidate_models(task_description: str, max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """
    Discover top candidate LLMs for the given task via OpenRouter.

    Strategy:
    1. Detect task category using Judge LLM
    2. Fetch available models from OpenRouter
    3. Filter to known top performers for the category
    4. Return enriched model metadata

    Args:
        task_description: Natural language task description
        max_candidates: Maximum number of candidate models to return

    Returns:
        List of candidate model dicts with id, name, category, context_length
    """
    logger.info("Detecting task category...")
    category = detect_task_category(task_description)
    logger.info(f"Detected category: {category}")

    # Get preferred models for this category
    preferred_ids = MODEL_CATEGORIES.get(category, MODEL_CATEGORIES["general"])

    # Fetch live model list from OpenRouter
    available_models = fetch_available_models()
    available_ids = {m["id"]: m for m in available_models}

    candidates = []
    for model_id in preferred_ids:
        if len(candidates) >= max_candidates:
            break
        if model_id in available_ids:
            model_info = available_ids[model_id]
            candidates.append({
                "id": model_id,
                "name": model_info.get("name", model_id),
                "category": category,
                "context_length": model_info.get("context_length", 8192),
                "pricing": model_info.get("pricing", {}),
            })
        else:
            # Model not available live, still include with basic info
            logger.warning(f"Model {model_id} not found in live list, including anyway")
            candidates.append({
                "id": model_id,
                "name": model_id.split("/")[-1].replace("-", " ").title(),
                "category": category,
                "context_length": 8192,
                "pricing": {},
            })

    logger.info(f"Discovered {len(candidates)} candidate models for category '{category}'")
    return candidates[:max_candidates]
