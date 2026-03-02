"""Benchmarking engine: runs test suite across candidate LLMs in parallel."""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

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
    max_workers: int = 6,
) -> dict[str, list[dict]]:
    """
    Run the full benchmark suite across all candidate models fully in parallel.

    All (model, test) combinations are dispatched concurrently, maximising
    throughput. Results are collected as futures complete and assembled into
    per-model ordered lists.

    Args:
        candidates: List of candidate model dicts (must have 'id' key)
        test_cases: List of test case dicts
        max_workers: Max concurrent API calls (default 6 — one per model)

    Returns:
        Dict mapping model_id -> list of result dicts (ordered by test_id)
    """
    # Pre-initialise result buckets preserving test order
    results: dict[str, list[dict]] = {c["id"]: [None] * len(test_cases) for c in candidates}
    test_index = {tc["id"]: idx for idx, tc in enumerate(test_cases)}

    total_calls = len(candidates) * len(test_cases)
    completed = 0
    start_time = time.time()

    logger.info(
        f"Starting parallel benchmark: {len(candidates)} models × "
        f"{len(test_cases)} tests = {total_calls} concurrent API calls "
        f"(max_workers={max_workers})"
    )

    def _run_task(candidate: dict, tc: dict) -> tuple[str, int, dict]:
        """Execute a single (model, test) pair and return (model_id, test_idx, result)."""
        model_id = candidate["id"]
        result = run_single_test(model_id, tc)
        return model_id, test_index[tc["id"]], result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit ALL (model × test) combinations at once
        future_to_info = {
            executor.submit(_run_task, candidate, tc): (candidate["id"], tc["id"])
            for candidate in candidates
            for tc in test_cases
        }

        for future in as_completed(future_to_info):
            model_id_key, test_id_key = future_to_info[future]
            try:
                mid, tidx, result = future.result(timeout=REQUEST_TIMEOUT + 30)
                results[mid][tidx] = result
                completed += 1
                status = "✓" if not result["error"] else "✗"
                elapsed = time.time() - start_time
                logger.info(
                    f"  [{status}] [{completed}/{total_calls}] {mid} | "
                    f"Test {result['test_id']} | "
                    f"Latency: {result['latency']:.2f}s | "
                    f"Elapsed: {elapsed:.1f}s"
                )
            except FuturesTimeoutError:
                logger.error(f"Timeout for {model_id_key} test {test_id_key}")
                tidx = test_index.get(test_id_key, 0)
                results[model_id_key][tidx] = {
                    "model_id": model_id_key,
                    "test_id": test_id_key,
                    "test_category": "unknown",
                    "test_difficulty": "unknown",
                    "prompt": "",
                    "response": "ERROR: Request timed out",
                    "latency": float(REQUEST_TIMEOUT),
                    "error": True,
                    "evaluation_criteria": "",
                    "expected_elements": [],
                }
                completed += 1
            except Exception as e:
                logger.error(f"Benchmark failed for {model_id_key} test {test_id_key}: {e}")
                tidx = test_index.get(test_id_key, 0)
                results[model_id_key][tidx] = {
                    "model_id": model_id_key,
                    "test_id": test_id_key,
                    "test_category": "unknown",
                    "test_difficulty": "unknown",
                    "prompt": "",
                    "response": f"ERROR: {str(e)}",
                    "latency": 0.0,
                    "error": True,
                    "evaluation_criteria": "",
                    "expected_elements": [],
                }
                completed += 1

    total_elapsed = time.time() - start_time
    logger.info(f"Benchmark complete: {completed}/{total_calls} calls in {total_elapsed:.1f}s")

    # Filter out any None slots (shouldn't happen, but defensive)
    return {mid: [r for r in res if r is not None] for mid, res in results.items()}


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
