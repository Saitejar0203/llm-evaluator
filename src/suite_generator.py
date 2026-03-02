"""Test suite generation using the Judge LLM."""

import json
import logging
import sys
import re

from .openrouter_client import call_judge

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


TEST_SUITE_PROMPT = """You are an expert AI evaluator designing a rigorous benchmark suite.

The user wants to evaluate LLMs for the following task:
<task_description>
{task_description}
</task_description>

Generate exactly {num_tests} diverse, comprehensive test cases that thoroughly evaluate LLM performance on this task.

Each test case must cover different aspects: basic competency, edge cases, complex reasoning, accuracy under ambiguity, and (if applicable) tool/function calling.

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "id": 1,
    "category": "basic|reasoning|edge_case|accuracy|tool_calling",
    "prompt": "The exact prompt to send to the candidate LLM",
    "evaluation_criteria": "What a correct/good response must contain or demonstrate",
    "expected_elements": ["key element 1", "key element 2"],
    "difficulty": "easy|medium|hard"
  }},
  ...
]

Rules:
- Make prompts realistic and representative of real user queries for this task
- Vary difficulty levels
- Include at least one edge case or adversarial prompt
- For coding tasks: include at least one debugging and one implementation prompt
- For math tasks: include at least one multi-step problem
- For conversational tasks: include context-dependent follow-ups
- Return ONLY the JSON array, no markdown fences, no extra text"""


def generate_test_suite(task_description: str, num_tests: int = 5) -> list[dict]:
    """
    Generate a comprehensive test suite for the given task using the Judge LLM.

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate

    Returns:
        List of test case dicts with id, category, prompt, evaluation_criteria, etc.
    """
    logger.info(f"Generating {num_tests} test cases for task: {task_description[:80]}...")

    messages = [
        {
            "role": "user",
            "content": TEST_SUITE_PROMPT.format(
                task_description=task_description,
                num_tests=num_tests,
            ),
        }
    ]

    response = call_judge(messages, temperature=0.4)

    # Parse JSON from response
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("```").strip()
        test_cases = json.loads(cleaned)
        if not isinstance(test_cases, list):
            raise ValueError("Expected a JSON array")
        logger.info(f"Successfully generated {len(test_cases)} test cases")
        return test_cases
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse test suite JSON: {e}\nRaw response:\n{response[:500]}")
        return _fallback_test_suite(task_description, num_tests)


def _fallback_test_suite(task_description: str, num_tests: int) -> list[dict]:
    """Generate a simple fallback test suite if JSON parsing fails."""
    logger.warning("Using fallback test suite generation")
    return [
        {
            "id": i + 1,
            "category": "general",
            "prompt": f"Task {i+1}: Demonstrate your capability for: {task_description}",
            "evaluation_criteria": "Response should be accurate, relevant, and well-structured",
            "expected_elements": ["relevance", "accuracy", "clarity"],
            "difficulty": "medium",
        }
        for i in range(num_tests)
    ]
