"""
Implementation tests for PM's LLM Evaluator.

Tests key components WITHOUT making any API calls.
All LLM/API functions are mocked where needed.
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_test(name, fn):
    """Run a single test function and report pass/fail."""
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as exc:
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {exc}")
        return False


# ============================================================
# 1. Import tests (no circular imports)
# ============================================================

def test_import_config():
    from src import config
    assert hasattr(config, "CANDIDATE_MODELS")
    assert hasattr(config, "MODEL_PRICING")
    assert hasattr(config, "EVAL_DIMENSIONS")
    assert hasattr(config, "GENERATOR_MODEL_ID")
    assert hasattr(config, "MAX_TOOL_CALLS")


def test_import_schemas():
    from src.schemas import EvaluationScore, TestCase, RankingEntry, RankingResult
    assert EvaluationScore is not None
    assert TestCase is not None


def test_import_knowledge_base():
    from src.knowledge_base import TOOL_SCHEMA, search_knowledge_base
    assert TOOL_SCHEMA is not None
    assert callable(search_knowledge_base)


def test_import_suite_generator():
    from src.suite_generator import (
        compute_test_distribution,
        generate_system_prompt,
        generate_knowledge_doc,
        generate_test_suite,
    )
    assert callable(compute_test_distribution)
    assert callable(generate_system_prompt)
    assert callable(generate_knowledge_doc)
    assert callable(generate_test_suite)


def test_import_benchmarker():
    from src.benchmarker import run_single_test, run_benchmark, compute_latency_stats
    assert callable(run_single_test)
    assert callable(run_benchmark)


def test_import_evaluator():
    from src.evaluator import (
        evaluate_response, evaluate_all_results, rank_models,
        compute_cost_per_question, _build_category_rubric,
    )
    assert callable(evaluate_response)
    assert callable(compute_cost_per_question)


def test_import_reporter():
    from src.reporter import (
        display_header, display_configured_models, display_evaluation_results,
        display_cost_analysis, display_ranking, save_report,
    )
    assert callable(display_header)
    assert callable(display_cost_analysis)


def test_import_model_discovery():
    from src.model_discovery import discover_candidate_models
    assert callable(discover_candidate_models)


def test_no_model_categories_in_model_discovery():
    """model_discovery must NOT import MODEL_CATEGORIES (it was removed from config)."""
    import ast, inspect
    from src import model_discovery
    source = inspect.getsource(model_discovery)
    assert "MODEL_CATEGORIES" not in source, (
        "model_discovery.py still references MODEL_CATEGORIES which was removed from config"
    )


def test_tool_schema_imported_in_main():
    """main.py must import TOOL_SCHEMA from knowledge_base."""
    main_path = Path(__file__).parent / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert "TOOL_SCHEMA" in source, "main.py does not import/use TOOL_SCHEMA"
    assert "from src.knowledge_base import TOOL_SCHEMA" in source, (
        "TOOL_SCHEMA is not imported from src.knowledge_base in main.py"
    )


def test_no_prompt_optimizer_in_main():
    """main.py must NOT reference prompt_optimizer."""
    main_path = Path(__file__).parent / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert "prompt_optimizer" not in source, (
        "main.py still references prompt_optimizer"
    )


def test_no_stream_handler_in_src():
    """Source modules must not add StreamHandlers (plan: file log only)."""
    bad_files = []
    for pyfile in (Path(__file__).parent / "src").glob("*.py"):
        if pyfile.name == "prompt_optimizer.py":
            continue  # old file, not part of new pipeline
        text = pyfile.read_text(encoding="utf-8")
        if "StreamHandler" in text:
            bad_files.append(pyfile.name)
    assert not bad_files, (
        f"These src files add StreamHandlers (should use file log only): {bad_files}"
    )


# ============================================================
# 2. EvaluationScore — new metrics and weights
# ============================================================

def test_evaluation_score_new_metric_names():
    from src.schemas import EvaluationScore
    score = EvaluationScore(
        accuracy=8.0,
        hallucination_resistance=9.0,
        faithfulness=7.0,
        abstention=8.0,
        tool_calling=7.5,
    )
    assert score.accuracy == 8.0
    assert score.hallucination_resistance == 9.0
    assert score.faithfulness == 7.0
    assert score.abstention == 8.0
    assert score.tool_calling == 7.5


def test_evaluation_score_no_old_metrics():
    from src.schemas import EvaluationScore
    score = EvaluationScore()
    assert not hasattr(score, "reasoning"), "Old 'reasoning' metric still present"
    assert not hasattr(score, "clarity"), "Old 'clarity' metric still present"
    assert not hasattr(score, "grounding"), "Old 'grounding' metric still present"
    assert not hasattr(score, "hallucination"), "Old 'hallucination' metric still present (should be hallucination_resistance)"


def test_evaluation_score_weighted_formula():
    """overall = accuracy*0.15 + hallucination_resistance*0.25 + faithfulness*0.20 + abstention*0.20 + tool_calling*0.20"""
    from src.schemas import EvaluationScore
    score = EvaluationScore(
        accuracy=10.0,
        hallucination_resistance=10.0,
        faithfulness=10.0,
        abstention=10.0,
        tool_calling=10.0,
    )
    # All 10s should give overall = 10.0
    assert score.overall == 10.0, f"Expected 10.0, got {score.overall}"


def test_evaluation_score_weighted_formula_mixed():
    """Verify exact weighted calculation."""
    from src.schemas import EvaluationScore
    score = EvaluationScore(
        accuracy=6.0,          # 6.0 * 0.15 = 0.90
        hallucination_resistance=8.0,  # 8.0 * 0.25 = 2.00
        faithfulness=7.0,      # 7.0 * 0.20 = 1.40
        abstention=9.0,        # 9.0 * 0.20 = 1.80
        tool_calling=5.0,      # 5.0 * 0.20 = 1.00
    )
    expected = round(6*0.15 + 8*0.25 + 7*0.20 + 9*0.20 + 5*0.20, 2)
    assert abs(score.overall - expected) < 0.01, (
        f"Expected overall={expected}, got {score.overall}"
    )


def test_evaluation_score_defaults():
    from src.schemas import EvaluationScore
    score = EvaluationScore()
    # All defaults are 5.0, so overall = 5.0 * (0.15+0.25+0.20+0.20+0.20) = 5.0
    assert score.overall == 5.0
    assert score.reasoning_text == "No reasoning provided."


def test_evaluation_score_coerces_strings():
    from src.schemas import EvaluationScore
    score = EvaluationScore(accuracy="7", hallucination_resistance="8.5")
    assert score.accuracy == 7.0
    assert score.hallucination_resistance == 8.5


def test_evaluation_score_clamps_out_of_range():
    from src.schemas import EvaluationScore
    score = EvaluationScore(accuracy=15.0, hallucination_resistance=-3.0)
    assert score.accuracy == 10.0
    assert score.hallucination_resistance == 0.0


def test_evaluation_score_to_dict_has_all_keys():
    from src.schemas import EvaluationScore
    score = EvaluationScore()
    d = score.to_dict()
    required_keys = {"accuracy", "hallucination_resistance", "faithfulness",
                     "abstention", "tool_calling", "overall", "reasoning_text"}
    assert required_keys == set(d.keys()), (
        f"Missing keys: {required_keys - set(d.keys())}, "
        f"Extra keys: {set(d.keys()) - required_keys}"
    )


# ============================================================
# 3. TestCase — new categories and fields
# ============================================================

def test_testcase_valid_categories():
    from src.schemas import TestCase
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        tc = TestCase(id=1, category=cat, prompt="A" * 15)
        assert tc.category == cat


def test_testcase_invalid_category_defaults_to_in_context():
    from src.schemas import TestCase
    tc = TestCase(id=1, category="basic", prompt="A" * 15)
    assert tc.category == "in_context", (
        f"Invalid category should default to 'in_context', got '{tc.category}'"
    )


def test_testcase_no_old_categories():
    """Old categories 'basic', 'reasoning' must not be accepted."""
    from src.schemas import TestCase
    for old_cat in ["basic", "reasoning", "applied"]:
        tc = TestCase(id=1, category=old_cat, prompt="A" * 15)
        assert tc.category == "in_context", (
            f"Old category '{old_cat}' was accepted instead of defaulting to 'in_context'"
        )


def test_testcase_has_expected_answer_field():
    from src.schemas import TestCase
    tc = TestCase(id=1, category="in_context", prompt="A" * 15,
                  expected_answer="The answer is 42")
    assert tc.expected_answer == "The answer is 42"


def test_testcase_has_relevant_kb_sections_field():
    from src.schemas import TestCase
    tc = TestCase(id=1, category="in_context", prompt="A" * 15,
                  relevant_kb_sections=["Section A", "Section B"])
    assert tc.relevant_kb_sections == ["Section A", "Section B"]


def test_testcase_valid_difficulties():
    from src.schemas import TestCase
    for diff in ["easy", "medium", "hard"]:
        tc = TestCase(id=1, category="in_context", prompt="A" * 15, difficulty=diff)
        assert tc.difficulty == diff


# ============================================================
# 4. compute_test_distribution()
# ============================================================

def test_distribution_n5():
    from src.suite_generator import compute_test_distribution
    d = compute_test_distribution(5)
    assert sum(d.values()) == 5
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        assert d[cat] >= 1


def test_distribution_n10():
    from src.suite_generator import compute_test_distribution
    d = compute_test_distribution(10)
    assert sum(d.values()) == 10
    assert d["in_context"] == 3
    assert d["out_of_context"] == 3
    assert d["general_knowledge"] == 2
    assert d["multi_fact"] == 1
    assert d["edge_case"] == 1


def test_distribution_n15():
    from src.suite_generator import compute_test_distribution
    d = compute_test_distribution(15)
    assert sum(d.values()) == 15
    # According to plan: N=15 → 5, 4, 3, 2, 1
    # But plan shows 5+4+3+2+1=15, and in_context=5, out_of_context=4
    # Let's verify the sum is correct and all cats >= 1
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        assert d[cat] >= 1, f"Category {cat} has {d[cat]} < 1"


def test_distribution_n20():
    from src.suite_generator import compute_test_distribution
    d = compute_test_distribution(20)
    assert sum(d.values()) == 20
    assert d["in_context"] >= 5
    assert d["out_of_context"] >= 5
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        assert d[cat] >= 1


def test_distribution_below_minimum_rounds_up():
    from src.suite_generator import compute_test_distribution
    d = compute_test_distribution(3)  # Below minimum of 5
    assert sum(d.values()) == 5, "Distribution below minimum should use 5"


def test_distribution_all_categories_present():
    from src.suite_generator import compute_test_distribution
    for n in [5, 8, 10, 12, 15, 20]:
        d = compute_test_distribution(n)
        assert len(d) == 5
        for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
            assert cat in d, f"Category '{cat}' missing from distribution for n={n}"


# ============================================================
# 5. search_knowledge_base()
# ============================================================

MOCK_KB = {
    "Account Management": (
        "Users can create accounts with a minimum balance of $50. "
        "Accounts must be verified within 30 days of creation. "
        "Deactivation requires 14-day notice."
    ),
    "Billing and Payments": (
        "Payments are processed within 3 business days. "
        "Late fees are $25 for balances over $100. "
        "Refunds take 5-7 business days to process."
    ),
    "Loan Products": (
        "Personal loans available from $1,000 to $50,000. "
        "Interest rates range from 5.9% to 24.9% APR. "
        "Loan terms: 12, 24, 36, or 60 months."
    ),
}


def test_search_kb_returns_relevant_section():
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("minimum account balance", MOCK_KB)
    assert "Account Management" in result or "$50" in result, (
        f"Expected Account Management section, got: {result[:100]}"
    )


def test_search_kb_returns_billing_for_payment_query():
    from src.knowledge_base import search_knowledge_base
    # Note: search_knowledge_base does exact keyword matching (no stemming).
    # Use plural "payments", "refunds" and "fees" which appear verbatim in the mock KB.
    result = search_knowledge_base("payments refunds fees late", MOCK_KB)
    assert "Billing" in result or "business days" in result, (
        f"Expected Billing section, got: {result[:100]}"
    )


def test_search_kb_returns_loan_section():
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("personal loan interest rate APR", MOCK_KB)
    assert "Loan" in result or "APR" in result or "interest" in result.lower(), (
        f"Expected Loan section, got: {result[:100]}"
    )


def test_search_kb_no_match_returns_not_found():
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("xyzqrst completely irrelevant gibberish", MOCK_KB)
    assert "No relevant information found" in result, (
        f"Expected no-match message, got: {result[:100]}"
    )


def test_search_kb_empty_doc_returns_not_found():
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("anything", {})
    assert "No relevant information found" in result


def test_search_kb_returns_string():
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("account", MOCK_KB)
    assert isinstance(result, str)


def test_search_kb_vague_query_returns_first_section():
    """A query with only stop words falls back to the first section."""
    from src.knowledge_base import search_knowledge_base
    result = search_knowledge_base("the a an is", MOCK_KB)
    # Should return first section (fallback for empty keyword set)
    first_key = list(MOCK_KB.keys())[0]
    assert first_key in result, (
        f"Expected fallback to first section '{first_key}', got: {result[:100]}"
    )


# ============================================================
# 6. TOOL_SCHEMA structure
# ============================================================

def test_tool_schema_structure():
    from src.knowledge_base import TOOL_SCHEMA
    assert TOOL_SCHEMA["type"] == "function"
    assert "function" in TOOL_SCHEMA
    fn = TOOL_SCHEMA["function"]
    assert fn["name"] == "search_knowledge_base"
    assert "parameters" in fn
    assert "query" in fn["parameters"]["properties"]
    assert "query" in fn["parameters"]["required"]


# ============================================================
# 7. _build_category_rubric()
# ============================================================

def test_build_category_rubric_returns_string():
    from src.evaluator import _build_category_rubric
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        result = _build_category_rubric(cat)
        assert isinstance(result, str), f"Rubric for {cat} is not a string"
        assert len(result) > 50, f"Rubric for {cat} is too short"


def test_build_category_rubric_contains_all_metrics():
    from src.evaluator import _build_category_rubric
    for cat in ["in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"]:
        result = _build_category_rubric(cat)
        for metric in ["accuracy", "hallucination_resistance", "faithfulness", "abstention", "tool_calling"]:
            assert metric in result, f"Metric '{metric}' missing from rubric for category '{cat}'"


def test_build_category_rubric_unknown_category_falls_back():
    from src.evaluator import _build_category_rubric
    result = _build_category_rubric("unknown_category")
    # Should fall back to in_context
    assert isinstance(result, str)
    assert len(result) > 0


# ============================================================
# 8. compute_cost_per_question()
# ============================================================

def test_compute_cost_per_question_basic():
    from src.evaluator import compute_cost_per_question
    # Use a known model from MODEL_PRICING
    mock_results = {
        "google/gemini-3.1-flash-lite-preview": [
            {
                "test_id": 1,
                "total_tokens": {"prompt": 1000, "completion": 200},
                "error": False,
                "latency": 1.0,
            },
            {
                "test_id": 2,
                "total_tokens": {"prompt": 800, "completion": 150},
                "error": False,
                "latency": 1.2,
            },
        ]
    }
    result = compute_cost_per_question(mock_results)
    assert "google/gemini-3.1-flash-lite-preview" in result

    data = result["google/gemini-3.1-flash-lite-preview"]
    assert data["num_questions"] == 2
    assert data["total_prompt_tokens"] == 1800
    assert data["total_completion_tokens"] == 350
    assert data["total_cost"] >= 0
    assert data["cost_per_question"] >= 0
    # Cost = (1800 * 0.25 + 350 * 1.50) / 1_000_000
    expected_cost = (1800 * 0.25 + 350 * 1.50) / 1_000_000
    assert abs(data["total_cost"] - expected_cost) < 0.000001


def test_compute_cost_per_question_unknown_model():
    """Unknown models should still return results, just with 0 cost."""
    from src.evaluator import compute_cost_per_question
    mock_results = {
        "unknown/model-xyz": [
            {"test_id": 1, "total_tokens": {"prompt": 500, "completion": 100}, "error": False}
        ]
    }
    result = compute_cost_per_question(mock_results)
    assert "unknown/model-xyz" in result
    data = result["unknown/model-xyz"]
    assert data["total_cost"] == 0.0  # No pricing data
    assert data["cost_per_question"] == 0.0


def test_compute_cost_per_question_empty_results():
    from src.evaluator import compute_cost_per_question
    result = compute_cost_per_question({})
    assert result == {}


def test_compute_cost_per_question_zero_questions():
    """Edge case: model has zero result entries."""
    from src.evaluator import compute_cost_per_question
    mock_results = {"google/gemini-3.1-flash-lite-preview": []}
    result = compute_cost_per_question(mock_results)
    data = result["google/gemini-3.1-flash-lite-preview"]
    assert data["num_questions"] == 0
    assert data["cost_per_question"] == 0.0


# ============================================================
# 9. Config values
# ============================================================

def test_config_eval_dimensions():
    from src.config import EVAL_DIMENSIONS
    expected = {"accuracy", "hallucination_resistance", "faithfulness", "abstention", "tool_calling"}
    assert set(EVAL_DIMENSIONS) == expected


def test_config_generator_model_id():
    from src.config import GENERATOR_MODEL_ID
    assert GENERATOR_MODEL_ID is not None
    assert len(GENERATOR_MODEL_ID) > 0


def test_config_max_tool_calls():
    from src.config import MAX_TOOL_CALLS
    assert MAX_TOOL_CALLS == 3


def test_config_model_pricing_has_all_candidates():
    from src.config import CANDIDATE_MODELS, MODEL_PRICING
    for model_id in CANDIDATE_MODELS:
        assert model_id in MODEL_PRICING, (
            f"Candidate model '{model_id}' has no pricing entry in MODEL_PRICING"
        )


def test_config_no_model_categories():
    """MODEL_CATEGORIES should not exist in config (removed per plan)."""
    from src import config
    assert not hasattr(config, "MODEL_CATEGORIES"), (
        "MODEL_CATEGORIES still exists in config.py but should have been removed"
    )


# ============================================================
# 10. call_llm returns dict (not tuple)
# ============================================================

def test_call_llm_returns_dict_structure():
    """Verify call_llm signature returns a dict with expected keys (mocked)."""
    from src.openrouter_client import call_llm

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello"
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    with patch("src.openrouter_client.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_llm("some/model", [{"role": "user", "content": "hello"}])

    assert isinstance(result, dict), "call_llm should return a dict, not a tuple"
    required_keys = {"content", "tool_calls", "latency", "usage", "finish_reason"}
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )
    assert isinstance(result["usage"], dict)
    assert "prompt_tokens" in result["usage"]
    assert "completion_tokens" in result["usage"]


def test_call_llm_accepts_tools_parameter():
    """call_llm must accept tools and tool_choice parameters."""
    import inspect
    from src.openrouter_client import call_llm
    sig = inspect.signature(call_llm)
    params = sig.parameters
    assert "tools" in params, "call_llm missing 'tools' parameter"
    assert "tool_choice" in params, "call_llm missing 'tool_choice' parameter"


# ============================================================
# 11. run_single_test signature accepts knowledge_doc
# ============================================================

def test_run_single_test_accepts_knowledge_doc():
    import inspect
    from src.benchmarker import run_single_test
    sig = inspect.signature(run_single_test)
    params = sig.parameters
    assert "knowledge_doc" in params, "run_single_test missing 'knowledge_doc' parameter"
    assert "tools" in params, "run_single_test missing 'tools' parameter"
    assert "system_prompt" in params, "run_single_test missing 'system_prompt' parameter"


def test_run_benchmark_accepts_knowledge_doc():
    import inspect
    from src.benchmarker import run_benchmark
    sig = inspect.signature(run_benchmark)
    params = sig.parameters
    assert "knowledge_doc" in params, "run_benchmark missing 'knowledge_doc' parameter"
    assert "tools" in params, "run_benchmark missing 'tools' parameter"
    assert "system_prompt" in params, "run_benchmark missing 'system_prompt' parameter"


# ============================================================
# 12. evaluate_all_results signature accepts knowledge_doc
# ============================================================

def test_evaluate_all_results_accepts_knowledge_doc():
    import inspect
    from src.evaluator import evaluate_all_results
    sig = inspect.signature(evaluate_all_results)
    params = sig.parameters
    assert "knowledge_doc" in params, "evaluate_all_results missing 'knowledge_doc' parameter"
    assert "system_prompt" in params, "evaluate_all_results missing 'system_prompt' parameter"


# ============================================================
# 13. discover_candidate_models — no task_description param
# ============================================================

def test_discover_candidate_models_no_task_param():
    """Per plan Slice 8: discover_candidate_models() no longer takes task_description."""
    import inspect
    from src.model_discovery import discover_candidate_models
    sig = inspect.signature(discover_candidate_models)
    params = sig.parameters
    assert "task_description" not in params, (
        "discover_candidate_models still has 'task_description' parameter "
        "(it was removed since models are fixed in config)"
    )


# ============================================================
# 14. main.py pipeline uses generate_knowledge_doc
# ============================================================

def test_main_uses_generate_knowledge_doc():
    main_path = Path(__file__).parent / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert "generate_knowledge_doc" in source, (
        "main.py does not call generate_knowledge_doc"
    )


def test_main_passes_knowledge_doc_to_run_benchmark():
    main_path = Path(__file__).parent / "main.py"
    source = main_path.read_text(encoding="utf-8")
    # knowledge_doc must appear in run_benchmark call context
    assert "knowledge_doc=knowledge_doc" in source or "knowledge_doc," in source, (
        "main.py does not pass knowledge_doc to run_benchmark"
    )


def test_main_passes_knowledge_doc_to_evaluate():
    main_path = Path(__file__).parent / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert "knowledge_doc=knowledge_doc" in source, (
        "main.py does not pass knowledge_doc to evaluate_all_results"
    )


# ============================================================
# 15. Smoke test: evaluate_response with mocked judge
# ============================================================

def test_evaluate_response_returns_all_metrics():
    """evaluate_response should return all 5 PM metrics + overall + reasoning_text."""
    from src.evaluator import evaluate_response

    mock_result = {
        "error": False,
        "test_id": 1,
        "test_category": "in_context",
        "test_difficulty": "medium",
        "prompt": "What is the minimum account balance?",
        "response": "The minimum balance is $50.",
        "conversation_chain": [
            {"role": "user", "content": "What is the minimum account balance?"},
            {"role": "assistant", "content": "The minimum balance is $50."},
        ],
        "tool_call_count": 1,
        "tool_queries": ["minimum balance"],
        "total_tokens": {"prompt": 200, "completion": 50},
        "latency": 1.5,
        "expected_answer": "Minimum balance is $50",
        "relevant_kb_sections": ["Account Management"],
        "expected_elements": ["$50", "minimum"],
        "evaluation_criteria": "Should state minimum balance accurately",
    }

    judge_response = '{"accuracy": 8.0, "hallucination_resistance": 9.0, "faithfulness": 8.5, "abstention": 7.0, "tool_calling": 9.0, "overall": 8.55, "reasoning_text": "Good response."}'

    with patch("src.evaluator.call_judge", return_value=judge_response):
        result = evaluate_response(
            task_description="Fintech customer support",
            result=mock_result,
            knowledge_doc=MOCK_KB,
            system_prompt="You are a support agent.",
        )

    required_keys = {"accuracy", "hallucination_resistance", "faithfulness", "abstention", "tool_calling", "overall", "reasoning_text"}
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )


def test_evaluate_response_error_result_returns_zeros():
    """Error results should short-circuit to zero scores."""
    from src.evaluator import evaluate_response

    error_result = {
        "error": True,
        "test_id": 1,
        "test_category": "in_context",
        "test_difficulty": "medium",
        "prompt": "What is the minimum balance?",
        "response": "ERROR: timeout",
        "conversation_chain": [],
        "tool_call_count": 0,
        "tool_queries": [],
        "total_tokens": {"prompt": 0, "completion": 0},
        "latency": 0.0,
        "expected_answer": "",
        "relevant_kb_sections": [],
        "expected_elements": [],
        "evaluation_criteria": "",
    }

    with patch("src.evaluator.call_judge") as mock_judge:
        result = evaluate_response(
            task_description="test",
            result=error_result,
        )
        mock_judge.assert_not_called()  # Should not call judge for error results

    assert result["accuracy"] == 0
    assert result["hallucination_resistance"] == 0


# ============================================================
# 16. Fallback test suite uses new categories
# ============================================================

def test_fallback_test_suite_uses_new_categories():
    from src.suite_generator import _fallback_test_suite
    cases = _fallback_test_suite("test task", 10)
    valid_categories = {"in_context", "out_of_context", "general_knowledge", "multi_fact", "edge_case"}
    for tc in cases:
        assert tc["category"] in valid_categories, (
            f"Fallback test case has invalid category: {tc['category']}"
        )


# ============================================================
# Main runner
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PM's LLM Evaluator — Implementation Test Suite")
    print("=" * 60 + "\n")

    tests = [
        # Import tests
        ("Import: config", test_import_config),
        ("Import: schemas", test_import_schemas),
        ("Import: knowledge_base", test_import_knowledge_base),
        ("Import: suite_generator", test_import_suite_generator),
        ("Import: benchmarker", test_import_benchmarker),
        ("Import: evaluator", test_import_evaluator),
        ("Import: reporter", test_import_reporter),
        ("Import: model_discovery", test_import_model_discovery),
        ("No MODEL_CATEGORIES in model_discovery", test_no_model_categories_in_model_discovery),
        ("TOOL_SCHEMA imported in main.py", test_tool_schema_imported_in_main),
        ("No prompt_optimizer in main.py", test_no_prompt_optimizer_in_main),
        ("No StreamHandler in src modules", test_no_stream_handler_in_src),
        # EvaluationScore
        ("EvaluationScore: new metric names", test_evaluation_score_new_metric_names),
        ("EvaluationScore: no old metrics", test_evaluation_score_no_old_metrics),
        ("EvaluationScore: weighted formula (all 10s)", test_evaluation_score_weighted_formula),
        ("EvaluationScore: weighted formula (mixed)", test_evaluation_score_weighted_formula_mixed),
        ("EvaluationScore: defaults", test_evaluation_score_defaults),
        ("EvaluationScore: coerces strings", test_evaluation_score_coerces_strings),
        ("EvaluationScore: clamps out of range", test_evaluation_score_clamps_out_of_range),
        ("EvaluationScore: to_dict has all keys", test_evaluation_score_to_dict_has_all_keys),
        # TestCase
        ("TestCase: valid categories", test_testcase_valid_categories),
        ("TestCase: invalid category defaults", test_testcase_invalid_category_defaults_to_in_context),
        ("TestCase: no old categories", test_testcase_no_old_categories),
        ("TestCase: expected_answer field", test_testcase_has_expected_answer_field),
        ("TestCase: relevant_kb_sections field", test_testcase_has_relevant_kb_sections_field),
        ("TestCase: valid difficulties", test_testcase_valid_difficulties),
        # compute_test_distribution
        ("Distribution N=5", test_distribution_n5),
        ("Distribution N=10", test_distribution_n10),
        ("Distribution N=15", test_distribution_n15),
        ("Distribution N=20", test_distribution_n20),
        ("Distribution below minimum rounds up", test_distribution_below_minimum_rounds_up),
        ("Distribution all categories present", test_distribution_all_categories_present),
        # search_knowledge_base
        ("KB search: returns relevant section", test_search_kb_returns_relevant_section),
        ("KB search: billing section", test_search_kb_returns_billing_for_payment_query),
        ("KB search: loan section", test_search_kb_returns_loan_section),
        ("KB search: no match", test_search_kb_no_match_returns_not_found),
        ("KB search: empty doc", test_search_kb_empty_doc_returns_not_found),
        ("KB search: returns string", test_search_kb_returns_string),
        ("KB search: vague query fallback", test_search_kb_vague_query_returns_first_section),
        # TOOL_SCHEMA
        ("TOOL_SCHEMA structure", test_tool_schema_structure),
        # _build_category_rubric
        ("Rubric: returns string", test_build_category_rubric_returns_string),
        ("Rubric: contains all metrics", test_build_category_rubric_contains_all_metrics),
        ("Rubric: unknown category fallback", test_build_category_rubric_unknown_category_falls_back),
        # compute_cost_per_question
        ("Cost analysis: basic", test_compute_cost_per_question_basic),
        ("Cost analysis: unknown model", test_compute_cost_per_question_unknown_model),
        ("Cost analysis: empty results", test_compute_cost_per_question_empty_results),
        ("Cost analysis: zero questions", test_compute_cost_per_question_zero_questions),
        # Config
        ("Config: EVAL_DIMENSIONS", test_config_eval_dimensions),
        ("Config: GENERATOR_MODEL_ID", test_config_generator_model_id),
        ("Config: MAX_TOOL_CALLS=3", test_config_max_tool_calls),
        ("Config: MODEL_PRICING has all candidates", test_config_model_pricing_has_all_candidates),
        ("Config: no MODEL_CATEGORIES", test_config_no_model_categories),
        # call_llm returns dict
        ("call_llm returns dict", test_call_llm_returns_dict_structure),
        ("call_llm accepts tools param", test_call_llm_accepts_tools_parameter),
        # Signatures
        ("run_single_test signature", test_run_single_test_accepts_knowledge_doc),
        ("run_benchmark signature", test_run_benchmark_accepts_knowledge_doc),
        ("evaluate_all_results signature", test_evaluate_all_results_accepts_knowledge_doc),
        ("discover_candidate_models: no task_description param", test_discover_candidate_models_no_task_param),
        # main.py pipeline
        ("main.py: uses generate_knowledge_doc", test_main_uses_generate_knowledge_doc),
        ("main.py: passes knowledge_doc to run_benchmark", test_main_passes_knowledge_doc_to_run_benchmark),
        ("main.py: passes knowledge_doc to evaluate", test_main_passes_knowledge_doc_to_evaluate),
        # evaluate_response
        ("evaluate_response: all metrics returned", test_evaluate_response_returns_all_metrics),
        ("evaluate_response: error short-circuits", test_evaluate_response_error_result_returns_zeros),
        # Fallback suite
        ("Fallback test suite: new categories", test_fallback_test_suite_uses_new_categories),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed / {failed} failed / {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
