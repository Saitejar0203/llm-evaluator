"""Rich terminal reporting for benchmark results."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.syntax import Syntax

console = Console()
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def display_header(task_description: str) -> None:
    """Display the tool header and task description."""
    console.print()
    console.print(Panel(
        Text("🤖 LLM Fitness Tool", style="bold cyan", justify="center"),
        subtitle="Automated LLM Selection & Evaluation",
        border_style="cyan",
    ))
    console.print(Panel(
        f"[bold]Task:[/bold] {task_description}",
        title="[yellow]Evaluation Target[/yellow]",
        border_style="yellow",
    ))


def display_test_suite(test_cases: list[dict]) -> None:
    """Display the generated test suite."""
    console.print(Rule("[bold green]Generated Test Suite[/bold green]"))
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Category", style="cyan", width=14)
    table.add_column("Difficulty", width=10)
    table.add_column("Prompt Preview", style="white")

    for tc in test_cases:
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
            tc.get("difficulty", "medium"), "white"
        )
        table.add_row(
            str(tc["id"]),
            tc.get("category", "general"),
            f"[{diff_color}]{tc.get('difficulty', 'medium')}[/{diff_color}]",
            tc["prompt"][:80] + ("..." if len(tc["prompt"]) > 80 else ""),
        )
    console.print(table)


def display_candidates(candidates: list[dict]) -> None:
    """Display discovered candidate models."""
    console.print(Rule("[bold blue]Discovered Candidate Models[/bold blue]"))
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Category", style="yellow")
    table.add_column("Context", style="dim")

    for i, c in enumerate(candidates, 1):
        ctx = f"{c.get('context_length', 0) // 1000}K" if c.get("context_length") else "N/A"
        table.add_row(str(i), c["id"], c.get("name", c["id"]), c.get("category", ""), ctx)
    console.print(table)


def display_benchmark_progress(model_id: str, test_id: int, latency: float, error: bool) -> None:
    """Display a single benchmark result line."""
    status = "[red]✗[/red]" if error else "[green]✓[/green]"
    console.print(f"  {status} [cyan]{model_id}[/cyan] | Test {test_id} | {latency:.2f}s")


def display_evaluation_results(model_evaluations: dict[str, dict], candidates: list[dict]) -> None:
    """Display evaluation scores table."""
    console.print(Rule("[bold magenta]Evaluation Results[/bold magenta]"))
    candidate_map = {c["id"]: c.get("name", c["id"]) for c in candidates}

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", min_width=30)
    table.add_column("Overall", justify="center", width=9)
    table.add_column("Accuracy", justify="center", width=9)
    table.add_column("Anti-Halluc", justify="center", width=12)
    table.add_column("Grounding", justify="center", width=10)
    table.add_column("Tool-Call", justify="center", width=10)
    table.add_column("Clarity", justify="center", width=8)
    table.add_column("Avg Latency", justify="center", width=12)

    sorted_models = sorted(
        model_evaluations.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )

    for model_id, scores in sorted_models:
        name = candidate_map.get(model_id, model_id)
        overall = scores.get("overall", 0)
        color = "green" if overall >= 7 else "yellow" if overall >= 5 else "red"
        table.add_row(
            model_id,
            f"[{color}]{overall:.1f}[/{color}]",
            f"{scores.get('accuracy', 0):.1f}",
            f"{scores.get('hallucination', 0):.1f}",
            f"{scores.get('grounding', 0):.1f}",
            f"{scores.get('tool_calling', 0):.1f}",
            f"{scores.get('clarity', 0):.1f}",
            f"{scores.get('avg_latency', 0):.2f}s",
        )
    console.print(table)


def display_ranking(ranking: dict, model_evaluations: dict[str, dict]) -> None:
    """Display the final top-3 ranking."""
    console.print(Rule("[bold gold1]🏆 Final Rankings — Top 3 LLMs[/bold gold1]"))

    medals = ["🥇", "🥈", "🥉"]
    rank_colors = ["gold1", "grey70", "orange3"]

    for entry in ranking.get("ranking", [])[:3]:
        rank = entry.get("rank", 1) - 1
        medal = medals[rank] if rank < 3 else "  "
        color = rank_colors[rank] if rank < 3 else "white"
        model_id = entry.get("model_id", "unknown")
        scores = model_evaluations.get(model_id, {})

        content = (
            f"[bold {color}]{medal} #{entry.get('rank')} — {model_id}[/bold {color}]\n\n"
            f"[bold]Overall Score:[/bold] {entry.get('overall_score', scores.get('overall', 0)):.1f}/10\n"
            f"[bold]Avg Latency:[/bold]  {scores.get('avg_latency', 0):.2f}s\n\n"
            f"[bold green]Strengths:[/bold green] {', '.join(entry.get('strengths', []))}\n"
            f"[bold red]Weaknesses:[/bold red] {', '.join(entry.get('weaknesses', ['None noted']))}\n\n"
            f"[italic]{entry.get('recommendation', '')}[/italic]"
        )
        console.print(Panel(content, border_style=color, padding=(1, 2)))

    if ranking.get("summary"):
        console.print(Panel(
            f"[italic]{ranking['summary']}[/italic]",
            title="[bold]Analysis Summary[/bold]",
            border_style="dim",
        ))


def display_optimized_prompt(optimized_prompt: str, model_id: str) -> None:
    """Display the optimized system prompt."""
    console.print(Rule("[bold cyan]🎯 Optimized System Prompt[/bold cyan]"))
    console.print(Panel(
        f"[dim]Optimized for:[/dim] [cyan]{model_id}[/cyan]",
        border_style="cyan",
    ))
    console.print(Panel(
        optimized_prompt,
        title="[bold cyan]System Prompt[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


def save_report(
    task_description: str,
    test_cases: list[dict],
    candidates: list[dict],
    benchmark_results: dict,
    model_evaluations: dict,
    ranking: dict,
    optimized_prompt: str,
    output_dir: str = "./analysis",
) -> str:
    """
    Save a full JSON report to disk.

    Args:
        task_description: Original task description
        test_cases: Generated test suite
        candidates: Candidate models
        benchmark_results: Raw benchmark results
        model_evaluations: Aggregated evaluation scores
        ranking: Final ranking dict
        optimized_prompt: Generated system prompt
        output_dir: Directory to save report

    Returns:
        Path to saved report file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/llm_fitness_report_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "task_description": task_description,
        "test_suite": test_cases,
        "candidates": candidates,
        "benchmark_results": benchmark_results,
        "model_evaluations": model_evaluations,
        "ranking": ranking,
        "optimized_prompt": optimized_prompt,
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[dim]📄 Full report saved to: [cyan]{filename}[/cyan][/dim]")
    return filename
