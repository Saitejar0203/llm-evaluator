"""Benchmarking engine: runs test suite across candidate LLMs with multi-turn tool calling."""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from .openrouter_client import call_llm
from .knowledge_base import search_knowledge_base
from .config import REQUEST_TIMEOUT, MAX_TOOL_CALLS

logger = logging.getLogger(__name__)


def run_single_test(
    model_id: str,
    test_case: dict,
    knowledge_doc: dict[str, str],
    system_prompt: str = "",
    tools: list[dict] | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    timeout: int = REQUEST_TIMEOUT,
) -> dict:
    """
    Run a single test case against one model with multi-turn tool calling.

    The model may call the search_knowledge_base tool up to max_tool_calls times.
    Each tool call result is appended to the conversation, and the model is called
    again until it produces a text response or hits the tool call cap.

    Args:
        model_id: OpenRouter model ID
        test_case: Test case dict with prompt, id, category, etc.
        knowledge_doc: Knowledge base dict for tool execution
        system_prompt: System prompt to set model role/behavior
        tools: List of tool schemas (pass [TOOL_SCHEMA] from knowledge_base)
        max_tool_calls: Maximum number of tool calls per question
        timeout: Request timeout in seconds

    Returns:
        Result dict with response, conversation chain, tool usage data
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": test_case["prompt"]})

    conversation_chain = list(messages)
    tool_call_count = 0
    tool_queries: list[str] = []
    total_tokens = {"prompt": 0, "completion": 0}
    total_latency = 0.0
    final_response = ""

    for turn in range(max_tool_calls + 1):  # +1 for final text response
        response = call_llm(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            timeout=timeout,
            tools=tools,
        )

        # Accumulate token usage and latency
        total_tokens["prompt"] += response["usage"]["prompt_tokens"]
        total_tokens["completion"] += response["usage"]["completion_tokens"]
        total_latency += response["latency"]

        # Check for errors
        if response["content"] and str(response["content"]).startswith("ERROR:"):
            final_response = response["content"]
            break

        if response["tool_calls"]:
            # Model wants to call a tool
            tool_call = response["tool_calls"][0]
            try:
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
            except (json.JSONDecodeError, AttributeError):
                query = str(tool_call.function.arguments)

            tool_queries.append(query)
            tool_call_count += 1

            # Execute tool locally
            tool_result = search_knowledge_base(query, knowledge_doc)

            # Build assistant message with tool call
            assistant_msg = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
            # Build tool result message
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }

            messages.append(assistant_msg)
            messages.append(tool_msg)
            conversation_chain = list(messages)

            logger.debug(
                f"{model_id} | Test {test_case['id']} | "
                f"Tool call {tool_call_count}: query='{query[:60]}'"
            )

            if tool_call_count >= max_tool_calls:
                # Hit cap — make one more call without tools to force text response
                response = call_llm(
                    model=model_id,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=timeout,
                    tools=None,  # No tools to force text response
                )
                total_tokens["prompt"] += response["usage"]["prompt_tokens"]
                total_tokens["completion"] += response["usage"]["completion_tokens"]
                total_latency += response["latency"]
                final_response = response["content"] or ""
                # Append final response to conversation chain
                messages.append({"role": "assistant", "content": final_response})
                conversation_chain = list(messages)
                break
        else:
            # Model gave a text response — done
            final_response = response["content"] or ""
            # Append final response to conversation chain
            messages.append({"role": "assistant", "content": final_response})
            conversation_chain = list(messages)
            break

    is_error = final_response.startswith("ERROR:") if final_response else False

    # Extract confidence self-assessment from response
    self_confidence = None
    if final_response:
        confidence_match = re.search(
            r'[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*/\s*10', final_response
        )
        if confidence_match:
            self_confidence = float(confidence_match.group(1))

    return {
        "model_id": model_id,
        "test_id": test_case["id"],
        "test_category": test_case.get("category", "in_context"),
        "test_difficulty": test_case.get("difficulty", "medium"),
        "prompt": test_case["prompt"],
        "response": final_response,
        "conversation_chain": conversation_chain,
        "tool_call_count": tool_call_count,
        "tool_queries": tool_queries,
        "total_tokens": total_tokens,
        "latency": round(total_latency, 3),
        "error": is_error,
        "self_confidence": self_confidence,
        # Pass through for evaluator
        "expected_answer": test_case.get("expected_answer", ""),
        "relevant_kb_sections": test_case.get("relevant_kb_sections", []),
        "evaluation_criteria": test_case.get("evaluation_criteria", ""),
        "expected_elements": test_case.get("expected_elements", []),
    }


def run_benchmark(
    candidates: list[dict],
    test_cases: list[dict],
    knowledge_doc: dict[str, str],
    system_prompt: str = "",
    tools: list[dict] | None = None,
    max_workers: int = 3,
) -> dict[str, list[dict]]:
    """
    Run the full benchmark suite across all candidate models in parallel.

    All (model, test) combinations are dispatched concurrently. Each test may
    involve multi-turn tool calling (up to MAX_TOOL_CALLS turns per test).

    Args:
        candidates: List of candidate model dicts (must have 'id' key)
        test_cases: List of test case dicts
        knowledge_doc: Knowledge base dict for tool execution
        system_prompt: System prompt to set model role/behavior
        tools: List of tool schemas for function calling
        max_workers: Max concurrent API calls (default 3 for multi-turn)

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
        f"Starting benchmark: {len(candidates)} models x "
        f"{len(test_cases)} tests = {total_calls} evaluations "
        f"(max_workers={max_workers})"
    )

    def _run_task(candidate: dict, tc: dict) -> tuple[str, int, dict]:
        """Execute a single (model, test) pair and return (model_id, test_idx, result)."""
        model_id = candidate["id"]
        result = run_single_test(
            model_id, tc,
            knowledge_doc=knowledge_doc,
            system_prompt=system_prompt,
            tools=tools,
        )
        return model_id, test_index[tc["id"]], result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit ALL (model x test) combinations at once
        future_to_info = {
            executor.submit(_run_task, candidate, tc): (candidate["id"], tc["id"])
            for candidate in candidates
            for tc in test_cases
        }

        for future in as_completed(future_to_info):
            model_id_key, test_id_key = future_to_info[future]
            try:
                mid, tidx, result = future.result(timeout=REQUEST_TIMEOUT * 4 + 30)
                results[mid][tidx] = result
                completed += 1
                status = "ok" if not result["error"] else "ERR"
                elapsed = time.time() - start_time
                logger.info(
                    f"  [{status}] [{completed}/{total_calls}] {mid} | "
                    f"Test {result['test_id']} | "
                    f"Tools: {result['tool_call_count']} | "
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
                    "conversation_chain": [],
                    "tool_call_count": 0,
                    "tool_queries": [],
                    "total_tokens": {"prompt": 0, "completion": 0},
                    "latency": float(REQUEST_TIMEOUT),
                    "error": True,
                    "self_confidence": None,
                    "expected_answer": "",
                    "relevant_kb_sections": [],
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
                    "conversation_chain": [],
                    "tool_call_count": 0,
                    "tool_queries": [],
                    "total_tokens": {"prompt": 0, "completion": 0},
                    "latency": 0.0,
                    "error": True,
                    "self_confidence": None,
                    "expected_answer": "",
                    "relevant_kb_sections": [],
                    "evaluation_criteria": "",
                    "expected_elements": [],
                }
                completed += 1

    total_elapsed = time.time() - start_time
    logger.info(f"Benchmark complete: {completed}/{total_calls} evaluations in {total_elapsed:.1f}s")

    # Filter out any None slots (defensive)
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
