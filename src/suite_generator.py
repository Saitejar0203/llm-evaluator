"""Test suite and knowledge base generation using the Generator LLM."""

import json
import logging
import re

from pydantic import ValidationError

from .openrouter_client import call_generator
from .schemas import TestCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GENERATION = """You are an expert prompt engineer. Given a task description, generate a concise system prompt that sets the role and basic behavioral guidelines for an AI performing this task.

The system prompt should:
- Define the AI's role clearly (e.g., "You are a customer support agent for...")
- Set 2-3 basic behavioral guidelines relevant to the task
- Mention that the AI has access to a knowledge base search tool for looking up specific information
- Be generic enough to work across different LLM providers
- NOT be overly detailed or prescriptive — keep it under 100 words
- NOT include output format instructions (those vary by use case)

Task description:
{task_description}

Return ONLY the system prompt text. No explanation, no markdown fences, no preamble. Start directly with "You are..." """


KNOWLEDGE_DOC_PROMPT = """You are a domain expert creating a detailed internal knowledge base document for the following task:

<task_description>
{task_description}
</task_description>

Generate a comprehensive knowledge base document split into 8-12 named sections. This document will be the ONLY source of truth that an AI assistant can reference when answering questions.

Requirements for EACH section:
- 200-400 words of detailed, specific content
- Include realistic numbers, dates, policies, limits, procedures, and edge cases
- Use specifics that a real company would have (e.g., "$50 minimum balance", "3 business days processing", "valid for accounts created after Jan 2024")
- Include enough detail that questions can be objectively verified against the content

The knowledge base must be rich enough that:
- Multiple questions can be sourced from different sections
- Some questions require combining information from 2+ sections
- Specific numbers and policies make hallucination objectively verifiable

Return a valid JSON object where keys are section names and values are the section content text.
Example format:
{{"Account Management": "Detailed content about account management...", "Billing & Payments": "Detailed content about billing..."}}

Return ONLY the JSON object. No markdown fences, no preamble, no explanation outside the JSON."""


TEST_SUITE_PROMPT = """You are creating a test suite to evaluate how well LLMs use a knowledge base tool.

KNOWLEDGE BASE:
{knowledge_doc_json}

Generate exactly {num_tests} test cases with this EXACT distribution:
- {n_in_context} in_context: answers ARE in the KB. Include expected_answer from KB.
- {n_out_of_context} out_of_context: answers are NOT in KB. Model should say "I don't know."
- {n_general_knowledge} general_knowledge: common knowledge, no tool needed.
- {n_multi_fact} multi_fact: requires 2+ KB sections to answer completely.
- {n_edge_case} edge_case: partially in KB — some parts answerable, some not.
- {n_off_topic} off_topic: questions OUTSIDE the domain (poems, recipes, medical advice). Model should stay in role and redirect.

Each test case must include ALL of these fields:
- id: sequential integer starting from 1
- category: one of in_context, out_of_context, general_knowledge, multi_fact, edge_case, off_topic
- prompt: the question to ask the model (realistic user query)
- expected_answer: what a correct answer should contain (quote specific KB facts for in_context/multi_fact)
- relevant_kb_sections: list of KB section names that contain the answer (empty for out_of_context/general_knowledge/off_topic)
- evaluation_criteria: what makes a good vs bad answer for this specific question
- expected_elements: list of key facts or behaviors expected in a good response
- difficulty: one of easy, medium, hard

Rules:
- Vary difficulty across test cases
- For in_context questions: expected_answer must reference specific numbers/policies from the KB
- For out_of_context questions: the topic must be clearly NOT in the KB
- For general_knowledge: questions about commonly known facts unrelated to the KB domain
- For multi_fact: the answer requires synthesizing information from at least 2 KB sections
- For edge_case: mix answerable and unanswerable parts in a single question
- For off_topic: the question must be completely outside the model's assigned domain (e.g., asking a fintech bot to write a poem or give medical advice)

Return ONLY a valid JSON array. No markdown fences, no preamble.

[{{"id": 1, "category": "in_context", "prompt": "...", "expected_answer": "...", "relevant_kb_sections": ["Section Name"], "evaluation_criteria": "...", "expected_elements": ["fact 1", "fact 2"], "difficulty": "medium"}}, ...]"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from text that may contain markdown fences,
    thinking tags, or extra prose.

    Args:
        text: Raw text from the LLM

    Returns:
        Parsed list of dicts

    Raises:
        ValueError: If no valid JSON array can be extracted
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON array from empty response")

    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "tests" in result:
            return result["tests"]
        if isinstance(result, dict) and "test_cases" in result:
            return result["test_cases"]
    except json.JSONDecodeError:
        pass

    # Find the outermost [ ... ] array
    bracket_depth = 0
    start_idx = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(cleaned):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx is not None:
                candidate = cleaned[start_idx : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        result = json.loads(fixed)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        pass
                start_idx = None
                bracket_depth = 0

    raise ValueError(f"Cannot extract JSON array from: {text[:300]}")


def _extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from text that may contain markdown fences,
    thinking tags, or extra prose.

    Args:
        text: Raw text from the LLM

    Returns:
        Parsed dict

    Raises:
        ValueError: If no valid JSON object can be extracted
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON object from empty response")

    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block
    brace_depth = 0
    bracket_depth = 0
    start_idx = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(cleaned):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and bracket_depth == 0 and start_idx is not None:
                candidate = cleaned[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                start_idx = None
                brace_depth = 0
                bracket_depth = 0

    raise ValueError(f"Cannot extract JSON object from: {text[:300]}")


# ---------------------------------------------------------------------------
# Test case validation
# ---------------------------------------------------------------------------

def _validate_test_cases(raw_cases: list, num_tests: int) -> list[dict]:
    """
    Validate and normalize test cases using Pydantic schemas.

    Args:
        raw_cases: Raw list of dicts from JSON parsing
        num_tests: Expected number of test cases

    Returns:
        List of validated test case dicts
    """
    validated = []
    for i, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            logger.warning(f"Skipping non-dict test case at index {i}: {raw}")
            continue
        # Ensure id is set correctly
        raw.setdefault("id", i + 1)
        try:
            tc = TestCase(**raw)
            validated.append(tc.model_dump())
        except ValidationError as e:
            logger.warning(f"Test case {i+1} failed Pydantic validation, using defaults: {e}")
            tc = TestCase(
                id=raw.get("id", i + 1),
                category=raw.get("category", "in_context"),
                prompt=str(raw.get("prompt", f"Task {i+1}: Demonstrate capability for the given task.")),
                evaluation_criteria=str(raw.get("evaluation_criteria", "Response should be accurate and relevant.")),
                expected_elements=raw.get("expected_elements", ["relevance", "accuracy"]),
                difficulty=raw.get("difficulty", "medium"),
                expected_answer=str(raw.get("expected_answer", "")),
                relevant_kb_sections=raw.get("relevant_kb_sections", []),
            )
            validated.append(tc.model_dump())

    return validated


# ---------------------------------------------------------------------------
# Test distribution
# ---------------------------------------------------------------------------

def compute_test_distribution(num_tests: int) -> dict[str, int]:
    """
    Compute category counts from total number of tests.

    Ensures minimum 1 per category (minimum total: 6).
    Distributes remaining by ratio: 25/25/15/10/10/15.

    Args:
        num_tests: Total number of test cases desired

    Returns:
        Dict mapping category -> count
    """
    n = max(num_tests, 6)

    # Assign 1 per category as minimum
    dist = {
        "in_context": 1,
        "out_of_context": 1,
        "general_knowledge": 1,
        "multi_fact": 1,
        "edge_case": 1,
        "off_topic": 1,
    }

    remaining = n - 6
    # Distribute remaining by ratio
    ratios = [
        ("in_context", 0.25),
        ("out_of_context", 0.25),
        ("general_knowledge", 0.15),
        ("multi_fact", 0.10),
        ("edge_case", 0.10),
        ("off_topic", 0.15),
    ]
    for cat, ratio in ratios:
        dist[cat] += int(remaining * ratio)

    # Assign any leftover to in_context/out_of_context/off_topic alternately
    allocated = sum(dist.values())
    leftover = n - allocated
    cycle = ["in_context", "out_of_context", "off_topic"]
    for i in range(leftover):
        dist[cycle[i % len(cycle)]] += 1

    return dist


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_system_prompt(task_description: str) -> str:
    """
    Generate a system prompt for candidate models based on the task.

    Uses the Generator LLM with medium thinking for creativity.

    Args:
        task_description: Natural language description of the task

    Returns:
        A system prompt string to use for all candidate models during benchmarking.
    """
    logger.info("Generating system prompt for candidate models...")

    messages = [
        {
            "role": "user",
            "content": SYSTEM_PROMPT_GENERATION.format(task_description=task_description),
        }
    ]

    response = call_generator(messages, thinking_level="medium")

    # Clean up: remove think tags, markdown fences, preamble
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:text)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Remove common preamble patterns
    for prefix in ["Sure", "Certainly", "Here is", "Here's", "System Prompt:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].lstrip(":").lstrip().lstrip("\n")

    if len(cleaned) < 20:
        cleaned = (
            f"You are an AI assistant specializing in: {task_description}. "
            "You have access to a knowledge base search tool for looking up specific information. "
            "Provide accurate, helpful, and well-structured responses."
        )

    # Append confidence calibration instruction
    cleaned += (
        "\n\nIMPORTANT: End every response with exactly this format on a new line: "
        "'Confidence: X/10' where X is your honest self-assessment of how certain "
        "you are about your answer (1=guessing, 10=absolutely certain)."
    )

    logger.info(f"Generated system prompt: {cleaned[:80]}...")
    return cleaned


def generate_knowledge_doc(task_description: str) -> dict[str, str]:
    """
    Generate a detailed knowledge base document for the given task.

    Uses the Generator LLM with minimal thinking (structured content generation).

    Args:
        task_description: Natural language description of the task

    Returns:
        Dict mapping section_name -> section_content (8-12 sections)
    """
    logger.info("Generating knowledge base document...")

    messages = [
        {
            "role": "user",
            "content": KNOWLEDGE_DOC_PROMPT.format(task_description=task_description),
        }
    ]

    response = call_generator(messages, thinking_level="minimal", max_tokens=12000)

    try:
        kb_doc = _extract_json_object(response)

        # Validate: 8-12 sections, each with meaningful content
        if len(kb_doc) < 4:
            logger.warning(f"KB has only {len(kb_doc)} sections, expected 8-12")
        if len(kb_doc) > 20:
            logger.warning(f"KB has {len(kb_doc)} sections, trimming to 12")
            keys = list(kb_doc.keys())[:12]
            kb_doc = {k: kb_doc[k] for k in keys}

        total_words = sum(len(v.split()) for v in kb_doc.values())
        logger.info(
            f"Generated knowledge base: {len(kb_doc)} sections, "
            f"~{total_words} total words"
        )
        return kb_doc

    except (ValueError, Exception) as e:
        logger.error(f"Failed to parse KB JSON: {e}\nRaw response:\n{response[:500]}")
        return _fallback_knowledge_doc(task_description)


def _fallback_knowledge_doc(task_description: str) -> dict[str, str]:
    """Generate a minimal fallback knowledge base if JSON parsing fails."""
    logger.warning("Using fallback knowledge base generation")
    return {
        "General Information": (
            f"This knowledge base covers information related to: {task_description}. "
            "For specific details, users should consult official documentation or contact support."
        ),
        "Policies and Procedures": (
            "Standard operating procedures apply. All requests are processed within "
            "3-5 business days. Escalation is available for urgent matters."
        ),
        "Frequently Asked Questions": (
            "Common questions are addressed in this section. For questions not covered here, "
            "users should reach out to the appropriate department for assistance."
        ),
    }


def generate_test_suite(
    task_description: str,
    knowledge_doc: dict[str, str],
    num_tests: int = 10,
) -> list[dict]:
    """
    Generate a comprehensive test suite for the given task using the Generator LLM.

    Uses Pydantic validation to ensure all test cases have the correct schema.
    Test cases are distributed across 6 categories based on the distribution formula.

    Args:
        task_description: Natural language description of the task
        knowledge_doc: Knowledge base dict for context-aware test generation
        num_tests: Number of test cases to generate

    Returns:
        List of validated test case dicts
    """
    num_tests = max(num_tests, 6)
    dist = compute_test_distribution(num_tests)

    logger.info(
        f"Generating {num_tests} test cases for task: {task_description[:80]}... "
        f"Distribution: {dist}"
    )

    # Serialize KB for the prompt
    kb_json = json.dumps(knowledge_doc, indent=2)

    messages = [
        {
            "role": "user",
            "content": TEST_SUITE_PROMPT.format(
                knowledge_doc_json=kb_json,
                num_tests=num_tests,
                n_in_context=dist["in_context"],
                n_out_of_context=dist["out_of_context"],
                n_general_knowledge=dist["general_knowledge"],
                n_multi_fact=dist["multi_fact"],
                n_edge_case=dist["edge_case"],
                n_off_topic=dist["off_topic"],
            ),
        }
    ]

    response = call_generator(messages, thinking_level="medium", max_tokens=12000)

    try:
        raw_cases = _extract_json_array(response)
        validated = _validate_test_cases(raw_cases, num_tests)
        if not validated:
            raise ValueError("No valid test cases after Pydantic validation")
        logger.info(f"Successfully generated {len(validated)} test cases")
        return validated
    except (ValueError, Exception) as e:
        logger.error(f"Failed to parse test suite JSON: {e}\nRaw response:\n{response[:500]}")
        return _fallback_test_suite(task_description, num_tests)


def _fallback_test_suite(task_description: str, num_tests: int) -> list[dict]:
    """
    Generate a simple fallback test suite if JSON parsing fails.

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate

    Returns:
        List of basic test case dicts
    """
    logger.warning("Using fallback test suite generation")
    categories = ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case", "off_topic"]
    difficulties = ["easy", "medium", "hard", "medium", "hard", "medium"]
    return [
        TestCase(
            id=i + 1,
            category=categories[i % len(categories)],
            prompt=f"Demonstrate your capability for the following task: {task_description}. Provide a detailed, practical example.",
            evaluation_criteria="Response should be accurate, relevant, well-structured, and demonstrate clear understanding of the task.",
            expected_elements=["relevance", "accuracy", "clarity", "practical example"],
            difficulty=difficulties[i % len(difficulties)],
            expected_answer="",
            relevant_kb_sections=[],
        ).model_dump()
        for i in range(num_tests)
    ]
