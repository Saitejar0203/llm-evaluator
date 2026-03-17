"""
PM's LLM Evaluator — CLI entry point.

Evaluates LLMs using PM-centric metrics: accuracy, hallucination resistance,
faithfulness, abstention quality, and tool calling efficiency.
Uses Gemini 3.1 Pro as the Judge and Gemini 3 Flash as the Generator via OpenRouter.
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_api_key, MAX_TEST_CASES
from src.suite_generator import (
    generate_test_suite, generate_system_prompt,
    generate_knowledge_doc, compute_test_distribution,
)
from src.knowledge_base import TOOL_SCHEMA
from src.model_discovery import discover_candidate_models
from src.benchmarker import run_benchmark
from src.evaluator import evaluate_all_results, rank_models, compute_cost_per_question
from src.reporter import (
    display_header,
    display_configured_models,
    display_test_suite,
    display_candidates,
    display_evaluation_results,
    display_cost_analysis,
    display_ranking,
    save_report,
    console,
)

# Shared timestamp for log and results files
_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging() -> None:
    """
    Configure two-layer logging: file handler only (Rich handles terminal output).

    Creates logs/ directory and writes detailed DEBUG logs to
    logs/eval_{timestamp}.log. No console handler — Rich panels, tables,
    and progress bars handle all terminal output.
    """
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"eval_{_run_timestamp}.log"

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Configure root logger with file handler only
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove any existing handlers to prevent duplicate logs
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers that cause thread-safety issues
    for noisy_logger in ("httpcore", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Log file: {log_file}")


logger = logging.getLogger(__name__)


def run_evaluation(
    task_description: str,
    num_tests: int = MAX_TEST_CASES,
    save_results: bool = True,
    output_dir: str = "./results",
    run_consistency: bool = False,
) -> dict:
    """
    Run the full PM's LLM evaluation pipeline (7 steps).

    Args:
        task_description: Natural language description of the task
        num_tests: Number of test cases to generate (minimum 6)
        save_results: Whether to save a markdown report
        output_dir: Directory to save the report
        run_consistency: Whether to run the consistency check

    Returns:
        Dict with full results including ranking and cost analysis
    """
    display_header(task_description)

    # ── Step 1/7: Generate system prompt ──────────────────────────────────
    console.print(Rule("[bold green]Step 1/7 — Generating System Prompt[/bold green]"))
    with console.status("[bold green]Generating system prompt...", spinner="dots"):
        system_prompt = generate_system_prompt(task_description)
    console.print(f"[green]Done.[/green] {system_prompt[:100]}...\n")

    # ── Step 2/7: Generate knowledge document ─────────────────────────────
    console.print(Rule("[bold green]Step 2/7 — Building Knowledge Base[/bold green]"))
    with console.status("[bold green]Generating knowledge base...", spinner="dots"):
        knowledge_doc = generate_knowledge_doc(task_description)
    console.print(
        f"[green]Done.[/green] {len(knowledge_doc)} sections, "
        f"~{sum(len(v.split()) for v in knowledge_doc.values())} words\n"
    )

    # ── Step 3/7: Generate test suite ─────────────────────────────────────
    console.print(Rule("[bold green]Step 3/7 — Creating Test Suite[/bold green]"))
    distribution = compute_test_distribution(num_tests)
    with console.status(f"[bold green]Generating {num_tests} test cases...", spinner="dots"):
        test_cases = generate_test_suite(task_description, knowledge_doc, num_tests=num_tests)
    display_test_suite(test_cases, distribution)
    console.print()

    # ── Step 4/7: Load candidate models ───────────────────────────────────
    console.print(Rule("[bold blue]Step 4/7 — Loading Candidate Models[/bold blue]"))
    with console.status("[bold blue]Fetching model metadata...", spinner="dots"):
        candidates = discover_candidate_models()
    display_candidates(candidates)
    console.print()

    # ── Step 5/7: Run benchmark ───────────────────────────────────────────
    console.print(Rule("[bold yellow]Step 5/7 — Running Benchmarks[/bold yellow]"))
    total_calls = len(candidates) * len(test_cases)
    console.print(
        f"[dim]{len(test_cases)} tests x {len(candidates)} models = "
        f"{total_calls} evaluations (multi-turn tool calling)[/dim]\n"
    )
    benchmark_results = run_benchmark(
        candidates, test_cases,
        knowledge_doc=knowledge_doc,
        system_prompt=system_prompt,
        tools=[TOOL_SCHEMA],
        max_workers=3,
    )

    # ── Step 6/7: Evaluate & rank ─────────────────────────────────────────
    console.print(Rule("[bold magenta]Step 6/7 — Judge Evaluating Responses[/bold magenta]"))
    console.print(
        f"[dim]Judge LLM scoring {total_calls} responses on accuracy, "
        f"hallucination resistance, faithfulness, abstention, tool calling...[/dim]\n"
    )
    model_evaluations = evaluate_all_results(
        task_description, benchmark_results,
        knowledge_doc=knowledge_doc,
        system_prompt=system_prompt,
    )
    display_evaluation_results(model_evaluations, candidates)

    ranking = rank_models(task_description, model_evaluations, candidates)

    # ── Step 7/7: Compute derived metrics ─────────────────────────────────
    console.print(Rule("[bold]Step 7/7 — Computing Results[/bold]"))
    cost_analysis = compute_cost_per_question(benchmark_results)
    display_cost_analysis(cost_analysis)

    # ── Optional: Consistency check ───────────────────────────────────────
    consistency_results = None
    if run_consistency:
        from src.consistency import run_consistency_check
        console.print(Rule("[bold]Consistency Check[/bold]"))
        console.print("[dim]Running consistency check (3 runs per question, 3 questions)...[/dim]\n")
        consistency_results = run_consistency_check(
            candidates=candidates,
            test_cases=test_cases,
            knowledge_doc=knowledge_doc,
            system_prompt=system_prompt,
            tools=[TOOL_SCHEMA],
        )
        # Display consistency results
        from rich.table import Table
        from rich import box
        cons_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        cons_table.add_column("Model", style="cyan", min_width=28)
        cons_table.add_column("Avg Consistency", justify="center", width=16)
        for model_id in sorted(
            consistency_results.keys(),
            key=lambda m: consistency_results[m].get("avg_consistency", 0),
            reverse=True,
        ):
            cons = consistency_results[model_id]
            score = cons.get("avg_consistency", 0)
            color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
            cons_table.add_row(model_id, f"[{color}]{score:.2f}[/{color}]")
        console.print(cons_table)

    display_ranking(ranking, model_evaluations, consistency_results=consistency_results)

    # ── Save report ───────────────────────────────────────────────────────
    report_path = ""
    if save_results:
        report_path = save_report(
            task_description=task_description,
            system_prompt=system_prompt,
            knowledge_doc=knowledge_doc,
            test_cases=test_cases,
            candidates=candidates,
            benchmark_results=benchmark_results,
            model_evaluations=model_evaluations,
            ranking=ranking,
            cost_analysis=cost_analysis,
            output_dir=output_dir,
            consistency_results=consistency_results,
        )

    # Show log file location
    log_file = Path(__file__).parent / "logs" / f"eval_{_run_timestamp}.log"
    if log_file.exists():
        console.print(f"[dim]Detailed logs: [cyan]{log_file}[/cyan][/dim]")

    console.print("\n[bold green]Evaluation complete![/bold green]\n")

    return {
        "task_description": task_description,
        "system_prompt": system_prompt,
        "knowledge_doc": knowledge_doc,
        "test_cases": test_cases,
        "candidates": candidates,
        "benchmark_results": benchmark_results,
        "model_evaluations": model_evaluations,
        "ranking": ranking,
        "cost_analysis": cost_analysis,
        "consistency_results": consistency_results,
        "report_path": report_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="pm-llm-evaluator",
        description="PM's LLM Evaluator — Evaluate LLMs like a Product Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "Customer support chatbot for a fintech app"
  python main.py --task "Internal HR knowledge assistant" --num-tests 15
  python main.py --task "Technical documentation Q&A bot" --no-save
  python main.py  # Interactive mode (prompts for task)
        """,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Natural language description of the task to evaluate LLMs for",
    )
    parser.add_argument(
        "--num-tests", "-n",
        type=int,
        default=MAX_TEST_CASES,
        help=f"Number of test cases to generate (default: {MAX_TEST_CASES}, minimum: 6)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./results",
        help="Directory to save markdown report (default: ./results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save markdown report to disk",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Run consistency check (3 runs per question, 3 representative questions per model)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the PM's LLM Evaluator CLI."""
    # Set up two-layer logging (file only — Rich handles terminal)
    _setup_logging()

    # Validate API key early
    try:
        load_api_key()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)

    args = parse_args()

    # Validate num-tests minimum
    if args.num_tests < 6:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] --num-tests must be at least 6. "
            f"Received {args.num_tests}, using 6 instead.\n"
        )
        args.num_tests = 6

    # Show configured models on startup
    display_configured_models()

    # Welcome message
    console.print(Panel(
        "[bold cyan]Welcome to PM's LLM Evaluator![/bold cyan]\n\n"
        "Evaluate LLMs the way a Product Manager would — not just accuracy,\n"
        "but trust, safety, and real-world reliability.\n\n"
        "[bold]Judge-scored:[/bold] accuracy, hallucination resistance, faithfulness,\n"
        "abstention, tool calling, and role adherence\n\n"
        "[bold]Computed:[/bold] confidence calibration, consistency, token efficiency,\n"
        "quality-adjusted cost, and cost per question",
        border_style="cyan",
    ))

    # --- Interactive: Task description ---
    task_description = args.task
    if not task_description:
        console.print(
            "\n[dim]Example tasks you can evaluate:[/dim]\n"
            "  [dim]- Customer support chatbot for a fintech app[/dim]\n"
            "  [dim]- Internal HR knowledge assistant for employee policies[/dim]\n"
            "  [dim]- Technical documentation Q&A bot for a SaaS product[/dim]\n"
            "  [dim]- Travel booking assistant for an airline website[/dim]\n"
        )
        task_description = Prompt.ask(
            "[bold yellow]What task should the LLMs be evaluated for?[/bold yellow]",
        )

    task_description = task_description.strip()
    if len(task_description) < 10:
        console.print("[bold red]Task description too short. Please provide at least 10 characters.[/bold red]")
        sys.exit(1)
    if len(task_description) > 2000:
        console.print("[bold red]Task description too long. Please keep it under 2000 characters.[/bold red]")
        sys.exit(1)

    # --- Interactive: Number of tests ---
    num_tests = args.num_tests
    if args.task is None:
        # Only ask interactively if user didn't pass --task (fully interactive mode)
        num_tests = IntPrompt.ask(
            "\n[bold yellow]How many test cases? [dim](minimum 6, recommended 10-15)[/dim][/bold yellow]",
        )
        if not num_tests or num_tests < 6:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Minimum is 6 (1 per category). Using 6.\n"
            )
            num_tests = 6

    # --- Interactive: Consistency check ---
    run_consistency = args.consistency
    if args.task is None:
        consistency_input = Prompt.ask(
            "\n[bold yellow]Run consistency check? [dim](runs each question 3x, adds ~3 min)[/dim][/bold yellow]",
            choices=["y", "n"],
        )
        run_consistency = consistency_input == "y"

    console.print()

    # Run evaluation
    try:
        run_evaluation(
            task_description=task_description,
            num_tests=num_tests,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            run_consistency=run_consistency,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
        logger.exception("Unexpected error during evaluation")
        sys.exit(1)


if __name__ == "__main__":
    main()
