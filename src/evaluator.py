"""Multi-faceted evaluation of LLM responses using the Judge LLM."""

import json
import logging
import sys
import re

from .openrouter_client import call_judge


def _extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from text that may contain markdown fences,
    trailing commas, or truncated content.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find the first { ... } block
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try to fix trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Cannot extract JSON from: {text[:200]}")


def _extract_scores_regex(text: str) -> dict:
    """Extract numeric scores from text using regex as last resort."""
    scores = {}
    for key in ["accuracy", "hallucination", "grounding", "tool_calling", "clarity", "overall"]:
        pattern = rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        scores[key] = float(match.group(1)) if match else 5.0
    if "reasoning" not in scores:
        scores["reasoning"] = "Extracted via regex fallback."
    return scores

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


EVALUATION_PROMPT = """You are an expert AI evaluator. Your task is to evaluate an LLM's response to a benchmark test with strict fairness and precision.

## Task Context
{task_description}

## Test Case
- **Category**: {test_category}
- **Difficulty**: {test_difficulty}
- **Prompt**: {prompt}
- **Evaluation Criteria**: {evaluation_criteria}
- **Expected Elements**: {expected_elements}

## LLM Response to Evaluate
{response}

## Evaluation Instructions
Score the response on each dimension from 0 to 10:

1. **accuracy** (0-10): Is the response factually correct and does it address the prompt correctly?
2. **hallucination** (0-10): 10 = no hallucination, 0 = severe hallucination. Penalize fabricated facts, citations, or code.
3. **grounding** (0-10): Is the response well-grounded in the prompt context? Does it stay on topic?
4. **tool_calling** (0-10): If tool/function calling was needed, was it done correctly? If not applicable, score 7 (neutral).
5. **clarity** (0-10): Is the response clear, well-structured, and easy to understand?

Return ONLY a valid JSON object with this exact structure:
{{
  "accuracy": <0-10>,
  "hallucination": <0-10>,
  "grounding": <0-10>,
  "tool_calling": <0-10>,
  "clarity": <0-10>,
  "overall": <weighted average 0-10>,
  "reasoning": "<2-3 sentence justification for the scores>"
}}

Be strict and fair. Do not inflate scores. Return ONLY the JSON object."""


RANKING_PROMPT = """You are an expert AI systems evaluator. Based on the benchmark results below, provide a final ranking and analysis.

## Task Description
{task_description}

## Benchmark Results Summary
{results_summary}

Provide a final ranking of the top 3 models. Return ONLY a valid JSON object:
{{
  "ranking": [
    {{
      "rank": 1,
      "model_id": "<model_id>",
      "overall_score": <0-10>,
      "strengths": ["strength 1", "strength 2"],
      "weaknesses": ["weakness 1"],
      "recommendation": "<one sentence why this model is best for the task>"
    }},
    ...
  ],
  "summary": "<2-3 sentence overall analysis>"
}}"""


def evaluate_response(
    task_description: str,
    result: dict,
) -> dict:
    """
    Evaluate a single LLM response using the Judge LLM.

    Args:
        task_description: The original task description
        result: A benchmark result dict from benchmarker.run_single_test

    Returns:
        Evaluation dict with scores for each dimension
    """
    if result.get("error"):
        return {
            "accuracy": 0, "hallucination": 0, "grounding": 0,
            "tool_calling": 0, "clarity": 0, "overall": 0,
            "reasoning": "Response contained an API error.",
        }

    messages = [
        {
            "role": "user",
            "content": EVALUATION_PROMPT.format(
                task_description=task_description,
                test_category=result.get("test_category", "general"),
                test_difficulty=result.get("test_difficulty", "medium"),
                prompt=result["prompt"],
                evaluation_criteria=result.get("evaluation_criteria", "N/A"),
                expected_elements=", ".join(result.get("expected_elements", [])),
                response=result["response"][:3000],  # Truncate very long responses
            ),
        }
    ]

    raw = call_judge(messages, temperature=0.1, max_tokens=1024)

    try:
        scores = _extract_json_object(raw)
        # Validate required keys
        required = ["accuracy", "hallucination", "grounding", "tool_calling", "clarity", "overall"]
        for key in required:
            if key not in scores:
                scores[key] = 5.0
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse evaluation JSON: {e}\nRaw: {raw[:300]}")
        # Try to extract individual scores with regex as last resort
        return _extract_scores_regex(raw)


def evaluate_all_results(
    task_description: str,
    benchmark_results: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Evaluate all benchmark results for all models.

    Args:
        task_description: The original task description
        benchmark_results: Dict mapping model_id -> list of result dicts

    Returns:
        Dict mapping model_id -> aggregated evaluation scores
    """
    model_evaluations: dict[str, dict] = {}

    for model_id, results in benchmark_results.items():
        logger.info(f"Evaluating responses for {model_id}...")
        per_test_scores = []

        for result in results:
            scores = evaluate_response(task_description, result)
            scores["test_id"] = result["test_id"]
            scores["latency"] = result["latency"]
            per_test_scores.append(scores)

        # Aggregate scores across all tests
        if per_test_scores:
            dims = ["accuracy", "hallucination", "grounding", "tool_calling", "clarity", "overall"]
            aggregated = {}
            for dim in dims:
                vals = [s[dim] for s in per_test_scores if isinstance(s.get(dim), (int, float))]
                aggregated[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

            latencies = [s["latency"] for s in per_test_scores]
            aggregated["avg_latency"] = round(sum(latencies) / len(latencies), 3)
            aggregated["per_test"] = per_test_scores
        else:
            aggregated = {
                "accuracy": 0, "hallucination": 0, "grounding": 0,
                "tool_calling": 0, "clarity": 0, "overall": 0,
                "avg_latency": 0, "per_test": [],
            }

        model_evaluations[model_id] = aggregated
        logger.info(
            f"  {model_id}: overall={aggregated['overall']:.1f}, "
            f"avg_latency={aggregated['avg_latency']:.2f}s"
        )

    return model_evaluations


def rank_models(
    task_description: str,
    model_evaluations: dict[str, dict],
    candidates: list[dict],
) -> dict:
    """
    Use the Judge LLM to produce a final ranked list of top 3 models.

    Args:
        task_description: The original task description
        model_evaluations: Dict mapping model_id -> aggregated scores
        candidates: List of candidate model dicts with metadata

    Returns:
        Ranking dict with top 3 models and analysis
    """
    # Build summary for the judge
    candidate_map = {c["id"]: c for c in candidates}
    summary_lines = []
    for model_id, scores in model_evaluations.items():
        name = candidate_map.get(model_id, {}).get("name", model_id)
        summary_lines.append(
            f"Model: {model_id} ({name})\n"
            f"  Overall: {scores['overall']:.1f}/10 | "
            f"Accuracy: {scores['accuracy']:.1f} | "
            f"Hallucination: {scores['hallucination']:.1f} | "
            f"Grounding: {scores['grounding']:.1f} | "
            f"Tool-calling: {scores['tool_calling']:.1f} | "
            f"Clarity: {scores['clarity']:.1f} | "
            f"Avg Latency: {scores['avg_latency']:.2f}s"
        )

    results_summary = "\n\n".join(summary_lines)

    messages = [
        {
            "role": "user",
            "content": RANKING_PROMPT.format(
                task_description=task_description,
                results_summary=results_summary,
            ),
        }
    ]

    raw = call_judge(messages, temperature=0.2, max_tokens=2048)

    try:
        ranking = _extract_json_object(raw)
        return ranking
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse ranking JSON: {e}")
        # Build fallback ranking from scores
        sorted_models = sorted(
            model_evaluations.items(),
            key=lambda x: x[1].get("overall", 0),
            reverse=True,
        )
        return {
            "ranking": [
                {
                    "rank": i + 1,
                    "model_id": mid,
                    "overall_score": scores["overall"],
                    "strengths": ["Strong overall performance"],
                    "weaknesses": [],
                    "recommendation": f"Ranked #{i+1} by overall score.",
                }
                for i, (mid, scores) in enumerate(sorted_models[:3])
            ],
            "summary": "Models ranked by aggregated evaluation scores.",
        }
