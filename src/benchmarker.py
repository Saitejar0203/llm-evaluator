"""Benchmarking engine: runs test suite across candidate LLMs."""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .openrouter_client import call_llm
from .config import REQUEST_TIMEOUT

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run_single_test(model_id: str, test_case: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """
    Run a single test case against one model.

    Args:
        model_id: OpenRouter model ID
        test_case: Test case dict with 'prompt', 'id', 'category', etc.
        timeout: Request timeout in seconds

    Returns:
        Result dict with model_id, test_id, response, latency, error
    """
    messages = [{"role": "user", "content": test_case["prompt"]}]
    response, latency = call_llm(
        model=model_id,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        timeout=timeout,
    )
    is_error = response.startswith("ERROR:")
    return {
        "model_id": model_id,
        "test_id": test_case["id"],
        "test_category": test_case.get("category", "general"),
        "test_difficulty": test_case.get("difficulty", "medium"),
        "prompt": test_case["prompt"],
        "response": response,
        "latency": round(latency, 3),
        "error": is_error,
        "evaluation_criteria": test_case.get("evaluation_criteria", ""),
        "expected_elements": test_case.get("expected_elements", []),
    }


def run_benchmark(
    candidates: list[dict],
    test_cases: list[dict],
    max_workers: int = 3,
) -> dict[str, list[dict]]:
    """
    Run the full benchmark suite across all candidate models.

    Uses thread-based parallelism to run models concurrently while
    keeping per-model test ordering sequential.

    Args:
        candidates: List of candidate model dicts (must have 'id' key)
        test_cases: List of test case dicts
        max_workers: Max parallel model evaluations

    Returns:
        Dict mapping model_id -> list of result dicts
    """
    results: dict[str, list[dict]] = {c["id"]: [] for c in candidates}

    total_calls = len(candidates) * len(test_cases)
    completed = 0

    logger.info(f"Starting benchmark: {len(candidates)} models × {len(test_cases)} tests = {total_calls} calls")

    def run_model_tests(candidate: dict) -> tuple[str, list[dict]]:
        """Run all tests for a single model sequentially."""
        model_id = candidate["id"]
        model_results = []
        for tc in test_cases:
            result = run_single_test(model_id, tc)
            model_results.append(result)
            status = "✓" if not result["error"] else "✗"
            logger.info(
                f"  [{status}] {model_id} | Test {tc['id']} | "
                f"Latency: {result['latency']:.2f}s"
            )
        return model_id, model_results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_model_tests, c): c["id"] for c in candidates}
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                mid, model_results = future.result()
                results[mid] = model_results
                completed += len(model_results)
                logger.info(f"Completed {model_id}: {len(model_results)} tests done ({completed}/{total_calls} total)")
            except Exception as e:
                logger.error(f"Model {model_id} benchmark failed: {e}")

    return results


def compute_latency_stats(results: list[dict]) -> dict:
    """
    Compute latency statistics for a model's results.

    Args:
        results: List of result dicts for one model

    Returns:
        Dict with avg, min, max, p50, p95 latency values
    """
    latencies = [r["latency"] for r in results if not r["error"]]
    if not latencies:
        return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "error_rate": 1.0}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50_idx = int(n * 0.5)
    p95_idx = min(int(n * 0.95), n - 1)

    error_count = sum(1 for r in results if r["error"])

    return {
        "avg": round(sum(latencies) / n, 3),
        "min": round(sorted_lat[0], 3),
        "max": round(sorted_lat[-1], 3),
        "p50": round(sorted_lat[p50_idx], 3),
        "p95": round(sorted_lat[p95_idx], 3),
        "error_rate": round(error_count / len(results), 3),
    }
