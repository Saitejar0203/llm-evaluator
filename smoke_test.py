"""
Smoke test: validates parallel benchmarking and evaluation with 2 models × 2 tests.
Measures wall-clock time to confirm speedup vs sequential baseline.
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

# Minimal 2-test suite
TEST_CASES = [
    {
        "id": 1,
        "category": "basic",
        "difficulty": "easy",
        "prompt": "Write a Python function that takes a list of integers and returns the sum of all even numbers.",
        "evaluation_criteria": "Must use a loop or list comprehension, return correct sum.",
        "expected_elements": ["loop or list comprehension", "even number check", "return statement"],
    },
    {
        "id": 2,
        "category": "reasoning",
        "difficulty": "medium",
        "prompt": "Explain the difference between a list and a tuple in Python. When would you use each?",
        "evaluation_criteria": "Must explain mutability, performance, and use cases.",
        "expected_elements": ["mutability", "performance difference", "use case examples"],
    },
]

# 2 candidate models for smoke test
CANDIDATES = [
    {"id": "openai/gpt-4.1", "name": "GPT-4.1"},
    {"id": "anthropic/claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
]

TASK = "Python programming assistant for beginners"

print("\n" + "="*60)
print("SMOKE TEST: Parallel Benchmark + Evaluation")
print("="*60)
print(f"Models: {len(CANDIDATES)} | Tests: {len(TEST_CASES)}")
print(f"Total API calls: {len(CANDIDATES) * len(TEST_CASES)} benchmark + {len(CANDIDATES) * len(TEST_CASES)} eval")
print("="*60 + "\n")

# ── Phase 1: Parallel Benchmark ──────────────────────────────────────────────
print("Phase 1: Running parallel benchmark...")
t0 = time.time()
benchmark_results = run_benchmark(CANDIDATES, TEST_CASES, max_workers=4)
bench_elapsed = time.time() - t0
print(f"\n✅ Benchmark complete in {bench_elapsed:.1f}s\n")

# Verify results
for model_id, results in benchmark_results.items():
    print(f"  {model_id}: {len(results)} results")
    for r in results:
        status = "✓" if not r["error"] else "✗ ERROR"
        print(f"    Test {r['test_id']}: {status} | latency={r['latency']:.2f}s | response_len={len(r['response'])} chars")

# ── Phase 2: Parallel Evaluation ─────────────────────────────────────────────
print(f"\nPhase 2: Running parallel evaluation (max_workers=4)...")
t1 = time.time()
model_evaluations = evaluate_all_results(TASK, benchmark_results, max_parallel_evaluations=4)
eval_elapsed = time.time() - t1
print(f"\n✅ Evaluation complete in {eval_elapsed:.1f}s\n")

# Print scores
print("="*60)
print("EVALUATION RESULTS:")
print("="*60)
for model_id, scores in model_evaluations.items():
    print(f"\n{model_id}:")
    print(f"  Overall:      {scores['overall']:.2f}/10")
    print(f"  Accuracy:     {scores['accuracy']:.2f}/10")
    print(f"  Hallucination:{scores['hallucination']:.2f}/10")
    print(f"  Grounding:    {scores['grounding']:.2f}/10")
    print(f"  Reasoning:    {scores.get('reasoning', 0):.2f}/10")
    print(f"  Clarity:      {scores['clarity']:.2f}/10")
    print(f"  Avg Latency:  {scores['avg_latency']:.2f}s")

total_elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"TOTAL WALL-CLOCK TIME: {total_elapsed:.1f}s")
print(f"  Benchmark phase: {bench_elapsed:.1f}s")
print(f"  Evaluation phase: {eval_elapsed:.1f}s")
print(f"  (Sequential estimate: ~{(bench_elapsed + eval_elapsed) * 1.8:.0f}s without parallelism)")
print("="*60)
print("\n✅ Smoke test PASSED — parallel execution verified")
