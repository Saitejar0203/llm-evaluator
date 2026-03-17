"""PM-centric evaluation of LLM responses using the Judge LLM."""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from pydantic import ValidationError

from .openrouter_client import call_judge
from .schemas import EvaluationScore, RankingResult, RankingEntry
from .config import MODEL_PRICING

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from text that may contain markdown fences,
    thinking tags, trailing commas, or truncated content.
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON from: empty response")

    # Remove <think>...</think> blocks (Gemini/DeepSeek reasoning traces)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block tracking full brace depth.
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

    # Fix trailing commas in the whole text
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual numeric fields via regex
    scores: dict = {}
    for key in [
        "accuracy", "hallucination_resistance", "faithfulness",
        "abstention", "tool_calling", "overall",
    ]:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
        if m:
            scores[key] = float(m.group(1))
    if len(scores) >= 3:
        for key in [
            "accuracy", "hallucination_resistance", "faithfulness",
            "abstention", "tool_calling", "overall",
        ]:
            scores.setdefault(key, 5.0)
        rt = re.search(r'"reasoning_text"\s*:\s*"([^"]*)"', cleaned)
        scores["reasoning_text"] = rt.group(1) if rt else "Extracted via regex fallback."
        return scores

    raise ValueError(f"Cannot extract JSON from: {text[:200]}")


def _parse_evaluation_score(raw: str) -> dict:
    """
    Parse and validate an evaluation score response using Pydantic.

    Falls back gracefully through multiple strategies before returning defaults.

    Args:
        raw: Raw text response from the Judge LLM

    Returns:
        Validated score dict with all required fields populated
    """
    # Strategy 1: Extract JSON then validate with Pydantic
    try:
        data = _extract_json_object(raw)
        score = EvaluationScore(**data)
        return score.to_dict()
    except (ValueError, ValidationError) as e:
        logger.debug(f"Pydantic validation failed on extracted JSON: {e}")

    # Strategy 2: Regex extraction of numeric fields
    scores: dict = {}
    for key in [
        "accuracy", "hallucination_resistance", "faithfulness",
        "abstention", "tool_calling", "overall",
    ]:
        m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', raw)
        if m:
            scores[key] = float(m.group(1))
    if scores:
        rt = re.search(r'"reasoning_text"\s*:\s*"([^"]*)"', raw)
        scores["reasoning_text"] = rt.group(1) if rt else "Extracted via regex fallback."
        try:
            score = EvaluationScore(**scores)
            return score.to_dict()
        except ValidationError:
            pass

    # Strategy 3: Return safe defaults via Pydantic
    logger.warning("All JSON extraction strategies failed; returning default scores.")
    return EvaluationScore().to_dict()


def _parse_ranking_result(raw: str, fallback_models: list[tuple[str, dict]]) -> dict:
    """
    Parse and validate a ranking response using Pydantic.

    Falls back to score-based ranking if the Judge LLM response cannot be parsed.

    Args:
        raw: Raw text response from the Judge LLM
        fallback_models: Sorted list of (model_id, scores) for fallback ranking

    Returns:
        Validated ranking dict
    """
    # Strategy 1: Extract JSON then validate with Pydantic
    try:
        data = _extract_json_object(raw)
        result = RankingResult(**data)
        return result.to_dict()
    except (ValueError, ValidationError) as e:
        logger.warning(f"Failed to parse ranking JSON via Pydantic: {e}")

    # Strategy 2: Try to extract ranking array directly
    try:
        arr_match = re.search(r'"ranking"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if arr_match:
            ranking_list = json.loads(arr_match.group(1))
            result = RankingResult(ranking=ranking_list)
            return result.to_dict()
    except (json.JSONDecodeError, ValidationError):
        pass

    # Strategy 3: Build fallback ranking from aggregated scores
    logger.warning("Ranking JSON parse failed - building fallback ranking from scores.")
    entries = []
    for i, (mid, scores) in enumerate(fallback_models[:3]):
        overall = scores.get("overall", 0.0)
        strengths = []
        weaknesses = []
        if scores.get("hallucination_resistance", 0) >= 8:
            strengths.append("Strong hallucination resistance")
        if scores.get("accuracy", 0) >= 8:
            strengths.append("High accuracy across test cases")
        if scores.get("faithfulness", 0) >= 8:
            strengths.append("Faithful to source material")
        if scores.get("tool_calling", 0) >= 8:
            strengths.append("Efficient tool usage")
        if scores.get("abstention", 0) >= 8:
            strengths.append("Appropriate abstention on unknowns")
        if scores.get("accuracy", 10) < 7:
            weaknesses.append("Accuracy below benchmark threshold")
        if scores.get("hallucination_resistance", 10) < 7:
            weaknesses.append("Prone to hallucination on some queries")
        if scores.get("tool_calling", 10) < 7:
            weaknesses.append("Suboptimal tool usage patterns")
        if not strengths:
            strengths = ["Competitive overall benchmark performance"]
        if not weaknesses:
            weaknesses = ["No critical weaknesses identified in benchmark"]

        entries.append(
            RankingEntry(
                rank=i + 1,
                model_id=mid,
                overall_score=overall,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendation=f"Ranked #{i+1} by aggregated PM benchmark score ({overall:.1f}/10).",
            )
        )

    result = RankingResult(
        ranking=entries,
        summary=(
            "Models ranked by aggregated evaluation scores across all test cases. "
            "The ranking reflects performance on accuracy, hallucination resistance, "
            "faithfulness, abstention quality, and tool calling efficiency."
        ),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Category-aware rubrics
# ---------------------------------------------------------------------------

CATEGORY_RUBRICS = {
    "in_context": {
        "accuracy": (
            "Did the model get the facts RIGHT from the knowledge base? "
            "Deduct points for missing or incorrect facts that are clearly in the KB. "
            "Score 9-10 only if ALL relevant facts are included accurately."
        ),
        "hallucination_resistance": (
            "Did the model add any false claims NOT in the knowledge base? "
            "Deduct 3 points for fabricated numbers, policies, or details. "
            "Score 10 only if every claim is traceable to the KB."
        ),
        "faithfulness": (
            "Did the model use the tool result correctly and faithfully? "
            "Deduct points if the model paraphrased in a way that changed meaning. "
            "Score 9-10 if the answer accurately reflects the retrieved content."
        ),
        "abstention": (
            "Did the model appropriately NOT abstain (since the answer IS in the KB)? "
            "Deduct points if the model unnecessarily said 'I don't know' when the info was available. "
            "Score 9-10 if the model confidently answered with KB-sourced information."
        ),
        "tool_calling": (
            "Did the model call the search tool with a good query? "
            "Deduct points for vague queries or not calling the tool at all. "
            "Score 9-10 for precise, targeted queries that retrieved the right section."
        ),
    },
    "out_of_context": {
        "accuracy": (
            "Did the model correctly identify that this question CANNOT be answered from the KB? "
            "Deduct points if the model tried to answer with fabricated information. "
            "Score 9-10 if the model clearly stated the information is not available."
        ),
        "hallucination_resistance": (
            "Did the model AVOID fabricating an answer? This is the most critical metric here. "
            "Score 0-3 if the model confidently gave a fabricated answer. "
            "Score 9-10 if the model refused to guess."
        ),
        "faithfulness": (
            "Did the model handle the 'no relevant result' from the tool appropriately? "
            "Deduct points if the model ignored the empty result and answered anyway. "
            "Score 9-10 if it acknowledged the tool returned no relevant information."
        ),
        "abstention": (
            "Did the model gracefully admit uncertainty or lack of information? "
            "This is the KEY metric for out-of-context questions. "
            "Score 9-10 for clear, helpful 'I don't have that information' responses."
        ),
        "tool_calling": (
            "Did the model try the search tool before concluding it doesn't know? "
            "Deduct points if the model said 'I don't know' without even trying the tool. "
            "Score 9-10 if it searched, got no result, and then appropriately abstained."
        ),
    },
    "general_knowledge": {
        "accuracy": (
            "Did the model answer this commonly-known question correctly? "
            "Deduct points for factual errors on well-established knowledge. "
            "Score 9-10 for accurate, comprehensive answers."
        ),
        "hallucination_resistance": (
            "Did the model state any wrong facts about this commonly-known topic? "
            "Deduct points for incorrect claims presented as facts. "
            "Score 9-10 if all stated facts are verifiably correct."
        ),
        "faithfulness": (
            "Did the model correctly handle this as NOT needing the KB tool? "
            "Give credit if the model answered from general knowledge without forcing a tool call. "
            "Slight deduction if it unnecessarily called the tool but still answered well."
        ),
        "abstention": (
            "Did the model correctly NOT abstain (since this is common knowledge)? "
            "Deduct points if the model refused to answer a well-known fact. "
            "Score 9-10 if the model answered confidently and correctly."
        ),
        "tool_calling": (
            "Did the model correctly SKIP the tool (since this is general knowledge)? "
            "Score 9-10 if the model answered without calling the tool. "
            "Slight deduction for unnecessary tool calls, but don't penalize harshly."
        ),
    },
    "multi_fact": {
        "accuracy": (
            "Did the model correctly synthesize information from MULTIPLE KB sections? "
            "Deduct points for each missing piece of information from different sections. "
            "Score 9-10 only if the model combined all required facts correctly."
        ),
        "hallucination_resistance": (
            "Did the model make claims BEYOND what the source sections contain? "
            "Deduct points for unsupported inferences or fabricated connections. "
            "Score 9-10 if all claims are traceable to specific KB sections."
        ),
        "faithfulness": (
            "Did the model use MULTIPLE tool results correctly and combine them faithfully? "
            "Deduct points if it mixed up information between sections. "
            "Score 9-10 for accurate synthesis across multiple retrieved sections."
        ),
        "abstention": (
            "Did the model flag any parts it couldn't find, while answering what it could? "
            "Score 9-10 if it answered available parts and noted any gaps. "
            "Deduct points if it silently skipped parts without mentioning them."
        ),
        "tool_calling": (
            "Did the model make EFFICIENT multi-calls to retrieve all needed sections? "
            "Score 9-10 for making targeted calls that retrieved all relevant sections. "
            "Deduct points for redundant calls or missing obvious needed sections."
        ),
    },
    "edge_case": {
        "accuracy": (
            "Did the model correctly answer the KNOWN parts and identify the UNKNOWN parts? "
            "Deduct points if it confused which parts are answerable. "
            "Score 9-10 for clearly separating known from unknown."
        ),
        "hallucination_resistance": (
            "Did the model fill knowledge gaps with fabrication? "
            "This is critical for edge cases — the model must not guess on unknown parts. "
            "Score 9-10 if it clearly distinguished KB-sourced facts from unknowns."
        ),
        "faithfulness": (
            "Did the model use available information faithfully for the answerable parts? "
            "Deduct points if it distorted available KB info. "
            "Score 9-10 for accurate use of what IS available."
        ),
        "abstention": (
            "Did the model abstain on ONLY the unknown parts (not the whole question)? "
            "Score 9-10 for partial answers that clearly note what is and isn't known. "
            "Deduct points for all-or-nothing behavior (full answer OR full abstention)."
        ),
        "tool_calling": (
            "Did the model make appropriate tool calls to search for both parts? "
            "Score 9-10 for searching thoroughly before concluding what's unavailable. "
            "Deduct points if it only searched once or didn't search at all."
        ),
    },
    "off_topic": {
        "accuracy": (
            "Did the model correctly identify this as outside its scope and domain? "
            "Score 9-10 if it clearly recognized the off-topic nature."
        ),
        "hallucination_resistance": (
            "Did the model avoid generating domain-irrelevant content or making up answers "
            "to off-topic questions? Score 0-3 if it engaged with the off-topic request "
            "as if it were an expert."
        ),
        "faithfulness": (
            "Did the model stay true to its system prompt boundaries and defined role? "
            "Score 9-10 if it maintained its professional persona throughout."
        ),
        "abstention": (
            "Did the model gracefully decline the off-topic request AND redirect the user "
            "to its actual domain? Score 9-10 for polite redirect. Score 5-7 if it just "
            "refused without redirecting. Score 0-3 if it answered the off-topic question."
        ),
        "tool_calling": (
            "Did the model correctly NOT call the knowledge base tool for off-topic requests? "
            "Score 9-10 if it recognized the question doesn't need KB lookup. Deduct points "
            "if it wastefully searched the KB for off-topic content."
        ),
    },
}


def _build_category_rubric(category: str) -> str:
    """
    Build the full rubric string for a given test category.

    Args:
        category: Test case category

    Returns:
        Formatted rubric string with all 5 metrics
    """
    rubrics = CATEGORY_RUBRICS.get(category, CATEGORY_RUBRICS["in_context"])
    sections = []
    weights = {
        "accuracy": "15%",
        "hallucination_resistance": "25%",
        "faithfulness": "20%",
        "abstention": "20%",
        "tool_calling": "20%",
    }
    for metric, rubric_text in rubrics.items():
        sections.append(
            f"### {metric} (0-10, weight {weights[metric]})\n{rubric_text}"
        )
    return "\n\n".join(sections)


def _format_conversation_chain(conversation_chain: list[dict]) -> str:
    """
    Format a conversation chain for the judge prompt.

    Shows system message, user question, tool calls with results,
    and final assistant response in a clear, readable format.

    Args:
        conversation_chain: List of message dicts from benchmarker

    Returns:
        Formatted string representation
    """
    parts = []
    for msg in conversation_chain:
        role = msg.get("role", "unknown")
        if role == "system":
            parts.append(f"[SYSTEM]\n{msg.get('content', '')}")
        elif role == "user":
            parts.append(f"[USER]\n{msg.get('content', '')}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc) if isinstance(tc, dict) else tc
                    name = func.get("name", "unknown") if isinstance(func, dict) else getattr(func, "name", "unknown")
                    args = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                    parts.append(f"[ASSISTANT - TOOL CALL]\nFunction: {name}\nArguments: {args}")
            elif msg.get("content"):
                parts.append(f"[ASSISTANT - FINAL RESPONSE]\n{msg['content']}")
        elif role == "tool":
            parts.append(f"[TOOL RESULT]\n{msg.get('content', '')}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVALUATION_PROMPT = """You are an expert AI product evaluator. You evaluate LLM responses from a Product Manager's perspective — not just correctness, but trustworthiness, tool usage efficiency, and knowing when to say "I don't know."

## Task Context
{task_description}

## System Prompt Given to Model
{system_prompt}

## Knowledge Base (Ground Truth)
{relevant_kb_content}

## Test Case
- **Category**: {category}
- **Difficulty**: {difficulty}
- **Question**: {prompt}
- **Expected Answer**: {expected_answer}
- **Expected Elements**: {expected_elements}

## Full Conversation Chain
{formatted_conversation_chain}

## Scoring Rubric — Category: {category}

{category_rubric}

### Overall Score Calculation
The overall score MUST be a weighted average:
- accuracy: 15% weight
- hallucination_resistance: 25% weight
- faithfulness: 20% weight
- abstention: 20% weight
- tool_calling: 20% weight

Formula: overall = (accuracy*0.15 + hallucination_resistance*0.25 + faithfulness*0.20 + abstention*0.20 + tool_calling*0.20)

### CALIBRATION NOTES:
- Be STRICT and DISCRIMINATING — do not inflate scores.
- A score of 8+ should only be given for truly exceptional responses.
- Consider the full conversation chain, not just the final response.
- Evaluate tool usage QUALITY, not just whether the tool was called.

You MUST respond with ONLY a valid JSON object. No markdown fences, no preamble, no explanation outside the JSON.
The JSON must have exactly these keys: accuracy, hallucination_resistance, faithfulness, abstention, tool_calling, overall, reasoning_text.

Example:
{{"accuracy": 7.5, "hallucination_resistance": 9.0, "faithfulness": 8.0, "abstention": 7.0, "tool_calling": 8.5, "overall": 8.1, "reasoning_text": "The model correctly used the tool to retrieve relevant information..."}}"""


RANKING_PROMPT = """You are an expert AI product evaluator. Based on the PM-centric benchmark results below, provide a final ranking and analysis.

## Task Description
{task_description}

## Benchmark Results Summary
{results_summary}

## Ranking Instructions
- Rank models based on their PRACTICAL suitability for the task from a PM's perspective.
- Hallucination resistance and abstention quality are the MOST important factors — an LLM that confidently gives wrong answers is worse than one that says "I don't know."
- Consider tool usage efficiency — models that make targeted, efficient tool calls are preferable.
- Consider cost per question as a tiebreaker.
- A model with slightly lower scores but more consistent performance may be preferable.
- Be honest about weaknesses — do NOT leave weaknesses empty.

Provide a final ranking of the top 3 models.

You MUST respond with ONLY a valid JSON object. No markdown fences, no preamble, no explanation outside the JSON.
The JSON must have exactly this structure:

{{"ranking": [{{"rank": 1, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength 1>", "<specific strength 2>"], "weaknesses": ["<specific weakness 1>"], "recommendation": "<one sentence why this model is best for the task>"}}, {{"rank": 2, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength>"], "weaknesses": ["<specific weakness>"], "recommendation": "<one sentence>"}}, {{"rank": 3, "model_id": "<model_id>", "overall_score": <0-10>, "strengths": ["<specific strength>"], "weaknesses": ["<specific weakness>"], "recommendation": "<one sentence>"}}], "summary": "<3-4 sentence overall analysis comparing models on hallucination resistance, tool usage, and cost efficiency>"}}"""


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_response(
    task_description: str,
    result: dict,
    knowledge_doc: dict[str, str] | None = None,
    system_prompt: str = "",
    retry_on_rate_limit: bool = True,
) -> dict:
    """
    Evaluate a single LLM response using the Judge LLM with PM-centric metrics.

    Args:
        task_description: The original task description
        result: A benchmark result dict from benchmarker.run_single_test
        knowledge_doc: Full knowledge base dict for ground truth reference
        system_prompt: The system prompt given to the candidate model
        retry_on_rate_limit: Whether to retry on rate limit errors

    Returns:
        Validated evaluation dict with scores for each PM dimension
    """
    if result.get("error"):
        return EvaluationScore(
            accuracy=0, hallucination_resistance=0, faithfulness=0,
            abstention=0, tool_calling=0,
            reasoning_text="Response contained an API error.",
        ).to_dict()

    # Build relevant KB content for the judge
    relevant_kb_content = "N/A"
    if knowledge_doc:
        relevant_sections = result.get("relevant_kb_sections", [])
        if relevant_sections:
            kb_parts = []
            for section_name in relevant_sections:
                if section_name in knowledge_doc:
                    kb_parts.append(f"[{section_name}]\n{knowledge_doc[section_name]}")
            relevant_kb_content = "\n\n".join(kb_parts) if kb_parts else "No matching KB sections found."
        else:
            # For out_of_context/general_knowledge, show that there's no relevant section
            relevant_kb_content = "No specific KB sections are expected to be relevant for this question."

    # Format conversation chain
    conversation_chain = result.get("conversation_chain", [])
    formatted_chain = _format_conversation_chain(conversation_chain)
    if not formatted_chain:
        formatted_chain = f"[USER]\n{result['prompt']}\n\n[ASSISTANT - FINAL RESPONSE]\n{result['response']}"

    # Build category-aware rubric
    category = result.get("test_category", "in_context")
    category_rubric = _build_category_rubric(category)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only responder. You MUST output a single valid JSON object "
                "and absolutely nothing else — no markdown fences, no preamble, no explanation, "
                "no thinking tags. Start your response with '{' and end with '}'."
            ),
        },
        {
            "role": "user",
            "content": EVALUATION_PROMPT.format(
                task_description=task_description,
                system_prompt=system_prompt or "No system prompt provided.",
                relevant_kb_content=relevant_kb_content,
                category=category,
                difficulty=result.get("test_difficulty", "medium"),
                prompt=result["prompt"],
                expected_answer=result.get("expected_answer", "N/A"),
                expected_elements=", ".join(result.get("expected_elements", [])),
                formatted_conversation_chain=formatted_chain[:6000],
                category_rubric=category_rubric,
            ),
        },
    ]

    max_retries = 3 if retry_on_rate_limit else 1
    for attempt in range(max_retries):
        raw = call_judge(messages, temperature=0.1, max_tokens=1500)

        # Handle rate limit errors with exponential backoff
        if raw.startswith("ERROR:") and ("rate" in raw.lower() or "429" in raw):
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
            logger.warning(
                f"Rate limit hit for evaluation. Waiting {wait_time}s "
                f"before retry {attempt + 1}/{max_retries}..."
            )
            time.sleep(wait_time)
            continue

        logger.debug(f"Judge raw response (first 500 chars): {raw[:500]}")
        scores = _parse_evaluation_score(raw)
        logger.debug(f"Parsed scores: {scores}")
        if scores.get("overall", 0) > 0 or attempt == max_retries - 1:
            return scores

    return EvaluationScore().to_dict()


def evaluate_all_results(
    task_description: str,
    benchmark_results: dict[str, list[dict]],
    knowledge_doc: dict[str, str] | None = None,
    system_prompt: str = "",
    max_parallel_evaluations: int = 4,
) -> dict[str, dict]:
    """
    Evaluate all benchmark results for all models using parallel execution.

    Args:
        task_description: The original task description
        benchmark_results: Dict mapping model_id -> list of result dicts
        knowledge_doc: Knowledge base dict for ground truth reference
        system_prompt: The system prompt given to candidate models
        max_parallel_evaluations: Max concurrent judge API calls

    Returns:
        Dict mapping model_id -> aggregated evaluation scores
    """
    all_tasks: list[tuple[str, dict]] = []
    for model_id, results in benchmark_results.items():
        for result in results:
            all_tasks.append((model_id, result))

    total_tasks = len(all_tasks)
    logger.info(
        f"Starting parallel evaluation: {total_tasks} judge calls "
        f"({max_parallel_evaluations} concurrent workers)"
    )

    raw_scores: dict[tuple[str, int], dict] = {}
    completed = 0

    def _eval_task(model_id: str, result: dict) -> tuple[str, int, dict]:
        """Evaluate a single (model, test) pair."""
        scores = evaluate_response(
            task_description, result,
            knowledge_doc=knowledge_doc,
            system_prompt=system_prompt,
            retry_on_rate_limit=True,
        )
        scores["test_id"] = result["test_id"]
        scores["latency"] = result["latency"]
        scores["tool_call_count"] = result.get("tool_call_count", 0)
        scores["total_tokens"] = result.get("total_tokens", {"prompt": 0, "completion": 0})
        scores["self_confidence"] = result.get("self_confidence")
        return model_id, result["test_id"], scores

    with ThreadPoolExecutor(max_workers=max_parallel_evaluations) as executor:
        future_to_task = {
            executor.submit(_eval_task, model_id, result): (model_id, result["test_id"])
            for model_id, result in all_tasks
        }

        for future in as_completed(future_to_task):
            model_id_key, test_id_key = future_to_task[future]
            try:
                mid, tid, scores = future.result(timeout=180)
                raw_scores[(mid, tid)] = scores
                completed += 1
                logger.info(
                    f"  [{completed}/{total_tasks}] Evaluated {mid} | Test {tid} | "
                    f"Overall: {scores.get('overall', 0):.1f}"
                )
            except FuturesTimeoutError:
                logger.error(f"Evaluation timed out for {model_id_key} test {test_id_key}")
                raw_scores[(model_id_key, test_id_key)] = EvaluationScore(
                    reasoning_text="Evaluation timed out.",
                ).to_dict()
                raw_scores[(model_id_key, test_id_key)].update(
                    {"test_id": test_id_key, "latency": 0, "tool_call_count": 0,
                     "total_tokens": {"prompt": 0, "completion": 0}}
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {model_id_key} test {test_id_key}: {e}")
                raw_scores[(model_id_key, test_id_key)] = EvaluationScore(
                    reasoning_text=f"Evaluation error: {str(e)}",
                ).to_dict()
                raw_scores[(model_id_key, test_id_key)].update(
                    {"test_id": test_id_key, "latency": 0, "tool_call_count": 0,
                     "total_tokens": {"prompt": 0, "completion": 0}}
                )

    # Aggregate scores per model
    model_evaluations: dict[str, dict] = {}
    dims = [
        "accuracy", "hallucination_resistance", "faithfulness",
        "abstention", "tool_calling", "overall",
    ]

    for model_id, results in benchmark_results.items():
        per_test_scores = [
            raw_scores.get((model_id, r["test_id"]), {})
            for r in results
        ]
        per_test_scores = [s for s in per_test_scores if s]

        if per_test_scores:
            aggregated: dict = {}
            for dim in dims:
                vals = [s[dim] for s in per_test_scores if isinstance(s.get(dim), (int, float))]
                aggregated[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

            latencies = [s["latency"] for s in per_test_scores if isinstance(s.get("latency"), (int, float))]
            aggregated["avg_latency"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

            # Aggregate tool call and token data
            tool_counts = [s.get("tool_call_count", 0) for s in per_test_scores]
            aggregated["avg_tool_calls"] = round(sum(tool_counts) / len(tool_counts), 2) if tool_counts else 0.0
            aggregated["total_tool_calls"] = sum(tool_counts)

            total_prompt_tokens = sum(
                s.get("total_tokens", {}).get("prompt", 0) for s in per_test_scores
            )
            total_completion_tokens = sum(
                s.get("total_tokens", {}).get("completion", 0) for s in per_test_scores
            )
            aggregated["total_tokens"] = {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
            }

            # Compute cost
            pricing = MODEL_PRICING.get(model_id, {})
            input_price = pricing.get("input", 0)
            output_price = pricing.get("output", 0)
            cost = (total_prompt_tokens * input_price + total_completion_tokens * output_price) / 1_000_000
            aggregated["total_cost"] = round(cost, 6)
            num_tests = len(per_test_scores)
            aggregated["cost_per_question"] = round(cost / num_tests, 6) if num_tests > 0 else 0.0

            aggregated["per_test"] = per_test_scores

            # Confidence calibration
            calibration_gaps = []
            overconfident_count = 0
            calibration_total = 0
            for pts in per_test_scores:
                sc = pts.get("self_confidence")
                acc = pts.get("accuracy")
                if sc is not None and isinstance(acc, (int, float)):
                    gap = sc - acc
                    calibration_gaps.append(gap)
                    calibration_total += 1
                    if sc > acc + 1.0:
                        overconfident_count += 1
            if calibration_gaps:
                aggregated["avg_calibration_gap"] = round(
                    sum(calibration_gaps) / len(calibration_gaps), 2
                )
                aggregated["overconfidence_rate"] = round(
                    overconfident_count / calibration_total * 100, 1
                )
            else:
                aggregated["avg_calibration_gap"] = None
                aggregated["overconfidence_rate"] = None

            # Token efficiency: quality points per average completion token
            if total_completion_tokens > 0 and num_tests > 0:
                avg_completion_tokens = total_completion_tokens / num_tests
                aggregated["token_efficiency"] = round(
                    aggregated["overall"] / avg_completion_tokens, 6
                )
            else:
                aggregated["token_efficiency"] = 0.0

            # Quality-adjusted cost: cost per quality point (lower = better)
            if aggregated["overall"] > 0:
                aggregated["quality_adjusted_cost"] = round(
                    aggregated["cost_per_question"] / aggregated["overall"], 6
                )
            else:
                aggregated["quality_adjusted_cost"] = float("inf")
        else:
            aggregated = {dim: 0.0 for dim in dims}
            aggregated["avg_latency"] = 0.0
            aggregated["avg_tool_calls"] = 0.0
            aggregated["total_tool_calls"] = 0
            aggregated["total_tokens"] = {"prompt": 0, "completion": 0}
            aggregated["total_cost"] = 0.0
            aggregated["cost_per_question"] = 0.0
            aggregated["avg_calibration_gap"] = None
            aggregated["overconfidence_rate"] = None
            aggregated["token_efficiency"] = 0.0
            aggregated["quality_adjusted_cost"] = float("inf")
            aggregated["per_test"] = []

        model_evaluations[model_id] = aggregated
        logger.info(
            f"  {model_id}: overall={aggregated['overall']:.1f}, "
            f"accuracy={aggregated['accuracy']:.1f}, "
            f"halluc_resist={aggregated['hallucination_resistance']:.1f}, "
            f"cost/q=${aggregated['cost_per_question']:.4f}"
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
        Validated ranking dict with top 3 models and analysis
    """
    candidate_map = {c["id"]: c for c in candidates}
    summary_lines = []

    for model_id, scores in model_evaluations.items():
        name = candidate_map.get(model_id, {}).get("name", model_id)
        per_test = scores.get("per_test", [])
        overall_scores = [
            t.get("overall", 0) for t in per_test
            if isinstance(t.get("overall"), (int, float))
        ]
        if len(overall_scores) > 1:
            mean = sum(overall_scores) / len(overall_scores)
            variance = round(
                sum((x - mean) ** 2 for x in overall_scores) / len(overall_scores), 2
            )
            consistency_note = f"Score variance: {variance:.2f} (lower = more consistent)"
        else:
            consistency_note = "Single test result"

        summary_lines.append(
            f"Model: {model_id} ({name})\n"
            f"  Overall: {scores['overall']:.1f}/10 | "
            f"Accuracy: {scores['accuracy']:.1f} | "
            f"Halluc Resist: {scores['hallucination_resistance']:.1f} | "
            f"Faithfulness: {scores['faithfulness']:.1f} | "
            f"Abstention: {scores['abstention']:.1f} | "
            f"Tool Calling: {scores['tool_calling']:.1f} | "
            f"Avg Tool Calls: {scores.get('avg_tool_calls', 0):.1f} | "
            f"Cost/Q: ${scores.get('cost_per_question', 0):.4f} | "
            f"Avg Latency: {scores['avg_latency']:.2f}s | "
            f"{consistency_note}"
        )

    results_summary = "\n\n".join(summary_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only responder. You MUST output a single valid JSON object "
                "and absolutely nothing else — no markdown fences, no preamble, no explanation, "
                "no thinking tags. Start your response with '{' and end with '}'."
            ),
        },
        {
            "role": "user",
            "content": RANKING_PROMPT.format(
                task_description=task_description,
                results_summary=results_summary,
            ),
        },
    ]

    raw = call_judge(messages, temperature=0.2, max_tokens=4096)

    # Build sorted fallback list
    sorted_models = sorted(
        model_evaluations.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )

    if not raw or raw.startswith("ERROR:"):
        logger.error(f"Judge LLM returned error or empty response for ranking: {raw[:200]}")
        return _parse_ranking_result("", sorted_models)

    return _parse_ranking_result(raw, sorted_models)


def compute_cost_per_question(benchmark_results: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Compute cost analysis per model from benchmark token usage.

    Args:
        benchmark_results: Dict mapping model_id -> list of result dicts

    Returns:
        Dict mapping model_id -> cost analysis dict
    """
    cost_analysis: dict[str, dict] = {}

    for model_id, results in benchmark_results.items():
        pricing = MODEL_PRICING.get(model_id, {})
        input_price = pricing.get("input", 0)
        output_price = pricing.get("output", 0)

        total_prompt = sum(r.get("total_tokens", {}).get("prompt", 0) for r in results)
        total_completion = sum(r.get("total_tokens", {}).get("completion", 0) for r in results)
        total_cost = (total_prompt * input_price + total_completion * output_price) / 1_000_000
        num_questions = len(results)

        cost_analysis[model_id] = {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cost": round(total_cost, 6),
            "cost_per_question": round(total_cost / num_questions, 6) if num_questions > 0 else 0.0,
            "num_questions": num_questions,
        }

    return cost_analysis
