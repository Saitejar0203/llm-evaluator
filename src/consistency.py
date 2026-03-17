"""Consistency checker: runs selected questions multiple times to measure response stability."""

import logging

from .benchmarker import run_single_test

logger = logging.getLogger(__name__)


def run_consistency_check(
    candidates: list[dict],
    test_cases: list[dict],
    knowledge_doc: dict[str, str],
    system_prompt: str = "",
    tools: list[dict] | None = None,
    num_runs: int = 3,
    max_questions: int = 3,
) -> dict[str, dict]:
    """
    Run selected questions multiple times per model and measure response consistency.

    Picks up to max_questions representative questions (1 in_context, 1 out_of_context,
    1 general_knowledge if available).

    Returns dict mapping model_id -> {avg_consistency, per_question: [{question, scores}]}
    """
    # Select representative questions
    selected = _select_representative_questions(test_cases, max_questions)

    logger.info(
        f"Consistency check: {len(selected)} questions x "
        f"{len(candidates)} models x {num_runs} runs"
    )

    results = {}
    for candidate in candidates:
        model_id = candidate["id"]
        question_results = []

        for tc in selected:
            responses = []
            for run_idx in range(num_runs):
                result = run_single_test(
                    model_id, tc,
                    knowledge_doc=knowledge_doc,
                    system_prompt=system_prompt,
                    tools=tools,
                )
                responses.append(result["response"])

            # Score pairwise similarity
            similarity = _score_consistency(responses)
            question_results.append({
                "test_id": tc["id"],
                "category": tc.get("category", "unknown"),
                "consistency_score": similarity,
            })

        avg_consistency = (
            sum(q["consistency_score"] for q in question_results) / len(question_results)
            if question_results else 0.0
        )

        results[model_id] = {
            "avg_consistency": round(avg_consistency, 2),
            "per_question": question_results,
        }
        logger.info(f"  {model_id}: consistency={avg_consistency:.2f}")

    return results


def _select_representative_questions(test_cases: list[dict], max_q: int) -> list[dict]:
    """Pick up to max_q diverse questions (one per category priority)."""
    priority = [
        "in_context", "out_of_context", "general_knowledge",
        "multi_fact", "edge_case", "off_topic",
    ]
    selected = []
    used_categories = set()

    for cat in priority:
        if len(selected) >= max_q:
            break
        for tc in test_cases:
            if tc.get("category") == cat and cat not in used_categories:
                selected.append(tc)
                used_categories.add(cat)
                break

    return selected


def _score_consistency(responses: list[str]) -> float:
    """
    Score consistency of multiple responses using Jaccard similarity.

    Returns average pairwise similarity (0.0 to 1.0).
    """
    if len(responses) < 2:
        return 1.0

    # Use word-level Jaccard similarity (cheap, no API call)
    def jaccard(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a and not words_b:
            return 1.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 1.0

    pairs = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            pairs.append(jaccard(responses[i], responses[j]))

    return sum(pairs) / len(pairs) if pairs else 1.0
