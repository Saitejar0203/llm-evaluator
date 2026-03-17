"""Rich terminal reporting and markdown report generation for PM's LLM Evaluator."""

import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()
logger = logging.getLogger(__name__)


def display_header(task_description: str) -> None:
    """Display the tool header and task description."""
    console.print()
    console.print(Panel(
        Text("PM's LLM Evaluator v1.0", style="bold cyan", justify="center"),
        subtitle="Evaluate LLMs like a Product Manager",
        border_style="cyan",
    ))
    console.print(Panel(
        f"[bold]Task:[/bold] {task_description}",
        title="[yellow]Evaluation Target[/yellow]",
        border_style="yellow",
    ))


def display_configured_models() -> None:
    """Show configured candidate models and pricing on startup."""
    from .config import CANDIDATE_MODELS, MODEL_PRICING

    console.print(Rule("[bold]Configured Candidate Models[/bold]"))
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="cyan")
    table.add_column("Input $/M", justify="right", width=12)
    table.add_column("Output $/M", justify="right", width=12)

    for i, model_id in enumerate(CANDIDATE_MODELS, 1):
        pricing = MODEL_PRICING.get(model_id, {})
        input_price = pricing.get("input", "?")
        output_price = pricing.get("output", "?")
        table.add_row(
            str(i),
            model_id,
            f"${input_price}",
            f"${output_price}",
        )
    console.print(table)
    console.print("[dim]To change models, edit CANDIDATE_MODELS in src/config.py[/dim]\n")

    # Warn about missing pricing
    missing = [m for m in CANDIDATE_MODELS if m not in MODEL_PRICING]
    if missing:
        console.print(
            f"[yellow]Warning: No pricing data for: {', '.join(missing)}. "
            f"Cost tracking will be unavailable for these models.[/yellow]\n"
        )


def display_test_suite(test_cases: list[dict], distribution: dict[str, int] | None = None) -> None:
    """Display the generated test suite with category distribution."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Difficulty", width=10)
    table.add_column("Prompt Preview", style="white")

    for tc in test_cases:
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
            tc.get("difficulty", "medium"), "white"
        )
        table.add_row(
            str(tc["id"]),
            tc.get("category", "in_context"),
            f"[{diff_color}]{tc.get('difficulty', 'medium')}[/{diff_color}]",
            tc["prompt"][:80] + ("..." if len(tc["prompt"]) > 80 else ""),
        )
    console.print(table)

    if distribution:
        dist_str = ", ".join(f"{count} {cat}" for cat, count in distribution.items())
        console.print(f"[dim]Distribution: {dist_str}[/dim]")


def display_candidates(candidates: list[dict]) -> None:
    """Display discovered candidate models."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Context", style="dim", width=8)

    for i, c in enumerate(candidates, 1):
        ctx = f"{c.get('context_length', 0) // 1000}K" if c.get("context_length") else "N/A"
        table.add_row(str(i), c["id"], c.get("name", c["id"]), ctx)
    console.print(table)


def display_evaluation_results(model_evaluations: dict[str, dict], candidates: list[dict]) -> None:
    """Display evaluation scores table with PM-centric metrics."""
    console.print(Rule("[bold magenta]Evaluation Results[/bold magenta]"))
    candidate_map = {c["id"]: c.get("name", c["id"]) for c in candidates}

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", min_width=28)
    table.add_column("Overall", justify="center", width=8)
    table.add_column("Accuracy", justify="center", width=9)
    table.add_column("Halluc", justify="center", width=8)
    table.add_column("Faithful", justify="center", width=9)
    table.add_column("Abstain", justify="center", width=8)
    table.add_column("ToolCall", justify="center", width=9)
    table.add_column("Cost/Q", justify="center", width=10)
    table.add_column("Latency", justify="center", width=9)
    table.add_column("Calib", justify="center", width=7)
    table.add_column("Effic", justify="center", width=8)

    sorted_models = sorted(
        model_evaluations.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )

    for model_id, scores in sorted_models:
        overall = scores.get("overall", 0)
        color = "green" if overall >= 7 else "yellow" if overall >= 5 else "red"
        cost_q = scores.get("cost_per_question", 0)

        # Format calibration gap
        calib_gap = scores.get("avg_calibration_gap")
        if calib_gap is not None:
            calib_color = "green" if abs(calib_gap) <= 1.0 else "yellow" if abs(calib_gap) <= 2.0 else "red"
            calib_str = f"[{calib_color}]{calib_gap:+.1f}[/{calib_color}]"
        else:
            calib_str = "[dim]N/A[/dim]"

        # Format token efficiency
        token_eff = scores.get("token_efficiency", 0)
        effic_str = f"{token_eff:.4f}" if token_eff > 0 else "[dim]N/A[/dim]"

        table.add_row(
            model_id,
            f"[{color}]{overall:.1f}[/{color}]",
            f"{scores.get('accuracy', 0):.1f}",
            f"{scores.get('hallucination_resistance', 0):.1f}",
            f"{scores.get('faithfulness', 0):.1f}",
            f"{scores.get('abstention', 0):.1f}",
            f"{scores.get('tool_calling', 0):.1f}",
            f"${cost_q:.4f}",
            f"{scores.get('avg_latency', 0):.1f}s",
            calib_str,
            effic_str,
        )
    console.print(table)


def display_cost_analysis(cost_analysis: dict[str, dict]) -> None:
    """Display cost analysis table per model."""
    console.print(Rule("[bold]Cost Analysis[/bold]"))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Model", style="cyan", min_width=28)
    table.add_column("Prompt Tokens", justify="right", width=14)
    table.add_column("Completion Tokens", justify="right", width=18)
    table.add_column("Total Cost", justify="right", width=12)
    table.add_column("Cost/Question", justify="right", width=14)

    sorted_costs = sorted(
        cost_analysis.items(),
        key=lambda x: x[1].get("cost_per_question", 0),
    )

    for model_id, data in sorted_costs:
        table.add_row(
            model_id,
            f"{data['total_prompt_tokens']:,}",
            f"{data['total_completion_tokens']:,}",
            f"${data['total_cost']:.4f}",
            f"${data['cost_per_question']:.4f}",
        )
    console.print(table)

    # Highlight cheapest and most expensive
    if sorted_costs:
        cheapest = sorted_costs[0]
        most_expensive = sorted_costs[-1]
        console.print(
            f"\n[dim]Cheapest: [green]{cheapest[0]}[/green] "
            f"(${cheapest[1]['cost_per_question']:.4f}/question) | "
            f"Most expensive: [red]{most_expensive[0]}[/red] "
            f"(${most_expensive[1]['cost_per_question']:.4f}/question)[/dim]"
        )


def display_ranking(
    ranking: dict,
    model_evaluations: dict[str, dict],
    consistency_results: dict[str, dict] | None = None,
) -> None:
    """Display the final top-3 ranking with PM-centric insights."""
    console.print(Rule("[bold gold1]Final Rankings[/bold gold1]"))

    medals = ["#1", "#2", "#3"]
    rank_colors = ["gold1", "grey70", "orange3"]

    for entry in ranking.get("ranking", [])[:3]:
        rank = entry.get("rank", 1) - 1
        medal = medals[rank] if rank < 3 else f"#{rank+1}"
        color = rank_colors[rank] if rank < 3 else "white"
        model_id = entry.get("model_id", "unknown")
        scores = model_evaluations.get(model_id, {})

        content = (
            f"[bold {color}]{medal} — {model_id}[/bold {color}]\n\n"
            f"[bold]Overall Score:[/bold] {entry.get('overall_score', scores.get('overall', 0)):.1f}/10\n"
            f"[bold]Cost/Question:[/bold] ${scores.get('cost_per_question', 0):.4f}\n"
            f"[bold]Avg Latency:[/bold]  {scores.get('avg_latency', 0):.1f}s\n"
        )

        # Add calibration gap if available
        calib_gap = scores.get("avg_calibration_gap")
        if calib_gap is not None:
            content += f"[bold]Calibration Gap:[/bold] {calib_gap:+.1f}\n"

        # Add consistency if available
        if consistency_results and model_id in consistency_results:
            cons_score = consistency_results[model_id].get("avg_consistency", 0)
            content += f"[bold]Consistency:[/bold] {cons_score:.2f}\n"

        content += (
            f"\n[bold green]Strengths:[/bold green] {', '.join(entry.get('strengths', []))}\n"
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

    console.print("\n[dim]For detailed per-question breakdown and metric definitions, see the results file.[/dim]")


METRIC_DEFINITIONS = """
## Metric Definitions

### Judge-Scored Metrics (0-10 scale)

| Metric | Weight | Description |
|--------|--------|-------------|
| **Accuracy** | 15% | Does the model get the facts right? For in-context questions, this means citing correct information from the knowledge base. For out-of-context questions, this means correctly identifying that the answer isn't available. |
| **Hallucination Resistance** | 25% | Does the model avoid fabricating facts? This is the highest-weighted metric because confident hallucinations are the #1 cause of AI product failures. A model that invents a policy (like Air Canada's chatbot) can cause legal and financial damage. |
| **Faithfulness** | 20% | Does the model use retrieved information accurately? When the knowledge base tool returns a result, does the model faithfully represent that information, or does it paraphrase in ways that change the meaning? |
| **Abstention** | 20% | Does the model know when to say "I don't know"? For questions outside its knowledge, a good model gracefully admits uncertainty rather than guessing. For answerable questions, it should NOT unnecessarily abstain. |
| **Tool Calling** | 20% | Does the model use the knowledge base tool effectively? This measures query quality (targeted vs. vague), call efficiency (1 precise call vs. 3 redundant ones), and decision quality (knowing when to skip the tool for general knowledge). |

### Computed Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Cost/Question** | (prompt_tokens x input_price + completion_tokens x output_price) / num_questions | Total API cost to answer one question, including all tool calling turns. |
| **Confidence Calibration** | self_rated_confidence - judge_accuracy_score | Measures how well a model knows what it knows. Positive = overconfident (dangerous). Negative = underconfident (wasteful but safe). Zero = perfectly calibrated. |
| **Overconfidence Rate** | % of responses where confidence > accuracy + 1.0 | How often the model is dangerously overconfident. Lower is better. |
| **Consistency** | Average pairwise Jaccard similarity across 3 runs | Do you get the same answer if you ask the same question twice? Score 0-1 where 1.0 = identical responses every time. Models with low consistency are unreliable in production. |
| **Token Efficiency** | overall_score / avg_completion_tokens | Quality per token — how concise is the model while maintaining quality? Higher = better. Verbose models cost more and frustrate users. |
| **Quality-Adjusted Cost** | cost_per_question / overall_score | Cost per unit of quality — enables value comparison. A $0.003/Q model scoring 7.0 (QAC=0.00043) is worse value than a $0.001/Q model scoring 5.0 (QAC=0.00020). Lower = better value. |
| **Avg Latency** | Total response time / num_questions | Average time to get a complete answer including all tool calling rounds. Important for real-time applications like customer support chat. |

### Test Categories

| Category | What It Tests | Good Model Behavior |
|----------|--------------|-------------------|
| **in_context** | Answer IS in the knowledge base | Call tool with precise query, cite correct facts, don't add unsupported claims |
| **out_of_context** | Answer is NOT in the knowledge base | Try the tool, recognize no relevant result, gracefully say "I don't have that information" |
| **general_knowledge** | Common knowledge, no tool needed | Answer from general knowledge WITHOUT calling the tool |
| **multi_fact** | Requires 2+ KB sections | Make multiple targeted tool calls, synthesize information correctly |
| **edge_case** | Partially in KB | Answer available parts, clearly flag unavailable parts, don't fabricate missing info |
| **off_topic** | Outside the model's assigned domain entirely | Stay in role, politely decline, redirect user to the model's actual domain |
"""


def save_report(
    task_description: str,
    system_prompt: str,
    knowledge_doc: dict[str, str],
    test_cases: list[dict],
    candidates: list[dict],
    benchmark_results: dict,
    model_evaluations: dict,
    ranking: dict,
    cost_analysis: dict,
    output_dir: str = "./results",
    consistency_results: dict[str, dict] | None = None,
) -> str:
    """
    Save a readable markdown report to disk.

    Args:
        task_description: Original task description
        system_prompt: Generated system prompt
        knowledge_doc: Knowledge base document
        test_cases: Generated test suite
        candidates: Candidate models
        benchmark_results: Raw benchmark results
        model_evaluations: Aggregated evaluation scores
        ranking: Final ranking dict
        cost_analysis: Cost analysis data
        output_dir: Directory to save report
        consistency_results: Optional consistency check results per model

    Returns:
        Path to saved report file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/eval_{timestamp}.md"

    lines = []
    lines.append(f"# PM's LLM Evaluator Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Task:** {task_description}\n")

    # System prompt
    lines.append("## System Prompt")
    lines.append(f"```\n{system_prompt}\n```\n")

    # Knowledge base
    lines.append("## Knowledge Base")
    for section_name, section_content in knowledge_doc.items():
        lines.append(f"### {section_name}")
        lines.append(f"{section_content[:500]}{'...' if len(section_content) > 500 else ''}\n")

    # Test suite
    lines.append("## Test Suite")
    lines.append(f"| # | Category | Difficulty | Prompt |")
    lines.append(f"|---|----------|------------|--------|")
    for tc in test_cases:
        prompt_preview = tc["prompt"][:60].replace("|", "\\|")
        lines.append(
            f"| {tc['id']} | {tc.get('category', 'N/A')} | "
            f"{tc.get('difficulty', 'N/A')} | {prompt_preview}... |"
        )
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append(
        "| Model | Overall | Accuracy | Halluc Resist | Faithful | Abstain | ToolCall | Cost/Q | Latency |"
    )
    lines.append(
        "|-------|---------|----------|---------------|----------|---------|----------|--------|---------|"
    )
    sorted_models = sorted(
        model_evaluations.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )
    for model_id, scores in sorted_models:
        lines.append(
            f"| {model_id} | {scores.get('overall', 0):.1f} | "
            f"{scores.get('accuracy', 0):.1f} | "
            f"{scores.get('hallucination_resistance', 0):.1f} | "
            f"{scores.get('faithfulness', 0):.1f} | "
            f"{scores.get('abstention', 0):.1f} | "
            f"{scores.get('tool_calling', 0):.1f} | "
            f"${scores.get('cost_per_question', 0):.4f} | "
            f"{scores.get('avg_latency', 0):.1f}s |"
        )
    lines.append("")

    # Ranking
    lines.append("## Ranking")
    for entry in ranking.get("ranking", [])[:3]:
        model_id = entry.get("model_id", "unknown")
        lines.append(f"### #{entry.get('rank', '?')} — {model_id}")
        lines.append(f"- **Overall Score:** {entry.get('overall_score', 0):.1f}/10")
        lines.append(f"- **Strengths:** {', '.join(entry.get('strengths', []))}")
        lines.append(f"- **Weaknesses:** {', '.join(entry.get('weaknesses', []))}")
        lines.append(f"- **Recommendation:** {entry.get('recommendation', '')}\n")

    if ranking.get("summary"):
        lines.append(f"**Summary:** {ranking['summary']}\n")

    # Cost analysis
    lines.append("## Cost Analysis")
    lines.append("| Model | Prompt Tokens | Completion Tokens | Total Cost | Cost/Question |")
    lines.append("|-------|--------------|-------------------|------------|---------------|")
    for model_id, data in sorted(cost_analysis.items(), key=lambda x: x[1].get("cost_per_question", 0)):
        lines.append(
            f"| {model_id} | {data['total_prompt_tokens']:,} | "
            f"{data['total_completion_tokens']:,} | "
            f"${data['total_cost']:.4f} | ${data['cost_per_question']:.4f} |"
        )
    lines.append("")

    # Tool calling analysis
    lines.append("## Tool Calling Analysis")
    lines.append("| Model | Avg Tool Calls | Total Tool Calls |")
    lines.append("|-------|---------------|-----------------|")
    for model_id, scores in sorted_models:
        lines.append(
            f"| {model_id} | {scores.get('avg_tool_calls', 0):.1f} | "
            f"{scores.get('total_tool_calls', 0)} |"
        )
    lines.append("")

    # Confidence Calibration Analysis
    lines.append("## Confidence Calibration Analysis")
    lines.append("| Model | Avg Calibration Gap | Overconfidence Rate |")
    lines.append("|-------|--------------------|--------------------|")
    for model_id, scores in sorted_models:
        calib_gap = scores.get("avg_calibration_gap")
        overconf_rate = scores.get("overconfidence_rate")
        calib_str = f"{calib_gap:+.2f}" if calib_gap is not None else "N/A"
        overconf_str = f"{overconf_rate:.1f}%" if overconf_rate is not None else "N/A"
        lines.append(f"| {model_id} | {calib_str} | {overconf_str} |")
    lines.append("")

    # Consistency Analysis (if available)
    if consistency_results:
        lines.append("## Consistency Analysis")
        lines.append("| Model | Avg Consistency |")
        lines.append("|-------|----------------|")
        for model_id in sorted(
            consistency_results.keys(),
            key=lambda m: consistency_results[m].get("avg_consistency", 0),
            reverse=True,
        ):
            cons = consistency_results[model_id]
            lines.append(f"| {model_id} | {cons.get('avg_consistency', 0):.2f} |")
        lines.append("")

    # Token Efficiency & Quality-Adjusted Cost
    lines.append("## Token Efficiency & Quality-Adjusted Cost")
    lines.append("| Model | Overall | Token Efficiency | Cost/Q | Quality-Adj Cost |")
    lines.append("|-------|---------|-----------------|--------|-----------------|")
    for model_id, scores in sorted_models:
        token_eff = scores.get("token_efficiency", 0)
        qac = scores.get("quality_adjusted_cost", float("inf"))
        qac_str = f"${qac:.6f}" if qac != float("inf") else "N/A"
        lines.append(
            f"| {model_id} | {scores.get('overall', 0):.1f} | "
            f"{token_eff:.6f} | "
            f"${scores.get('cost_per_question', 0):.4f} | {qac_str} |"
        )
    lines.append("")

    # Per-question breakdown
    lines.append("## Per-Question Breakdown")
    for model_id, results in benchmark_results.items():
        lines.append(f"\n### {model_id}")
        for result in results:
            test_id = result.get("test_id", "?")
            category = result.get("test_category", "?")
            tool_calls = result.get("tool_call_count", 0)
            response_preview = result.get("response", "")[:200].replace("\n", " ")
            lines.append(f"**Test {test_id}** ({category}) | Tools: {tool_calls}")
            lines.append(f"> {response_preview}{'...' if len(result.get('response', '')) > 200 else ''}\n")

    # Metric Definitions (always appended)
    lines.append(METRIC_DEFINITIONS)

    report_content = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    console.print(f"\n[dim]Full results saved to: [cyan]{filename}[/cyan][/dim]")
    return filename
