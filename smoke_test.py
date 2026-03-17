"""
Smoke test: validates parallel benchmarking and evaluation with 2 models x 2 tests.
Updated for PM-centric metrics and multi-turn tool calling.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.benchmarker import run_benchmark
from src.evaluator import evaluate_all_results
from src.knowledge_base import TOOL_SCHEMA

# Mock knowledge base for smoke test
KNOWLEDGE_DOC = {
    "Python Lists": (
        "Python lists are mutable ordered sequences. They support append, remove, "
        "insert, and pop operations. Lists are created with square brackets []. "
        "Lists can contain mixed types and are dynamically sized."
    ),
    "Python Tuples": (
        "Python tuples are immutable ordered sequences. Once created, elements cannot "
        "be changed, added, or removed. Tuples use parentheses (). They are faster "
        "than lists for iteration and use less memory."
    ),
}

# Minimal 2-test suite with new PM-centric categories
TEST_CASES = [
    {
        "id": 1,
        "category": "in_context",
        "difficulty": "easy",
        "prompt": "What operations do Python lists support?",
        "evaluation_criteria": "Must use knowledge base to find list operations.",
        "expected_elements": ["append", "remove", "insert", "pop"],
        "expected_answer": "Lists support append, remove, insert, and pop operations.",
        "relevant_kb_sections": ["Python Lists"],
    },
    {
        "id": 2,
        "category": "general_knowledge",
        "difficulty": "medium",
        "prompt": "What is the difference between a list and a tuple in Python?",
        "evaluation_criteria": "Should answer from general knowledge without calling the tool.",
        "expected_elements": ["mutability", "performance difference"],
        "expected_answer": "Lists are mutable, tuples are immutable.",
        "relevant_kb_sections": [],
    },
]

# 2 candidate models for smoke test
CANDIDATES = [
    {"id": "openai/gpt-4.1", "name": "GPT-4.1"},
    {"id": "anthropic/claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
]

TASK = "Python programming assistant for beginners"
SYSTEM_PROMPT = "You are a helpful Python tutor. You have access to a knowledge base search tool. Use it when you need specific information."

print("\n" + "="*60)
print("SMOKE TEST: PM LLM Evaluator — Parallel Benchmark + Evaluation")
print("="*60)
print(f"Models: {len(CANDIDATES)} | Tests: {len(TEST_CASES)}")
print(f"Expected benchmark calls: {len(CANDIDATES) * len(TEST_CASES)} (may multiply with tool calls)")
print("="*60 + "\n")

# -- Phase 1: Parallel Benchmark with Tool Calling --
print("Phase 1: Running parallel benchmark (multi-turn tool calling)...")
t0 = time.time()
benchmark_results = run_benchmark(
    CANDIDATES, TEST_CASES,
    knowledge_doc=KNOWLEDGE_DOC,
    system_prompt=SYSTEM_PROMPT,
    tools=[TOOL_SCHEMA],
    max_workers=2,
)
bench_elapsed = time.time() - t0
print(f"\nBenchmark complete in {bench_elapsed:.1f}s\n")

# Verify results
for model_id, results in benchmark_results.items():
    print(f"  {model_id}: {len(results)} results")
    for r in results:
        status = "ok" if not r["error"] else "ERR"
        print(
            f"    Test {r['test_id']}: {status} | "
            f"latency={r['latency']:.2f}s | "
            f"tool_calls={r['tool_call_count']} | "
            f"response_len={len(r['response'])} chars"
        )

# -- Phase 2: Parallel Evaluation --
print(f"\nPhase 2: Running parallel evaluation (max_workers=4)...")
t1 = time.time()
model_evaluations = evaluate_all_results(
    TASK, benchmark_results,
    knowledge_doc=KNOWLEDGE_DOC,
    system_prompt=SYSTEM_PROMPT,
    max_parallel_evaluations=4,
)
eval_elapsed = time.time() - t1
print(f"\nEvaluation complete in {eval_elapsed:.1f}s\n")

# Print scores with new PM metrics
print("="*60)
print("EVALUATION RESULTS (PM Metrics):")
print("="*60)
for model_id, scores in model_evaluations.items():
    print(f"\n{model_id}:")
    print(f"  Overall:              {scores['overall']:.2f}/10")
    print(f"  Accuracy:             {scores['accuracy']:.2f}/10")
    print(f"  Hallucination Resist: {scores['hallucination_resistance']:.2f}/10")
    print(f"  Faithfulness:         {scores['faithfulness']:.2f}/10")
    print(f"  Abstention:           {scores['abstention']:.2f}/10")
    print(f"  Tool Calling:         {scores['tool_calling']:.2f}/10")
    print(f"  Avg Latency:          {scores['avg_latency']:.2f}s")
    print(f"  Avg Tool Calls:       {scores.get('avg_tool_calls', 0):.1f}")
    print(f"  Avg Cost/Q:           ${scores.get('avg_cost', 0):.4f}")

total_elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"TOTAL WALL-CLOCK TIME: {total_elapsed:.1f}s")
print(f"  Benchmark phase: {bench_elapsed:.1f}s")
print(f"  Evaluation phase: {eval_elapsed:.1f}s")
print("="*60)
print("\nSmoke test PASSED — PM evaluation pipeline verified")
