"""Prompt optimization using the Judge LLM."""

import logging
import sys

from .openrouter_client import call_judge

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


PROMPT_OPTIMIZATION_TEMPLATE = """You are an expert prompt engineer. Your task is to create an optimized system prompt for a specific AI task.

## Task Description
{task_description}

## Top Recommended Model
{top_model_id} ({top_model_name})

## Model's Strengths
{strengths}

## Benchmark Performance
- Overall Score: {overall_score}/10
- Accuracy: {accuracy}/10
- Hallucination Resistance: {hallucination}/10
- Grounding: {grounding}/10
- Clarity: {clarity}/10

## Instructions
Create a structured, task-specific system prompt that:
1. Clearly defines the AI's role and expertise for this task
2. Sets behavioral guidelines to maximize accuracy and minimize hallucination
3. Specifies output format requirements
4. Includes any task-specific constraints or best practices
5. Is optimized for the model's known strengths

The system prompt should be production-ready and immediately usable.

Return ONLY the system prompt text (no explanations, no markdown headers, just the prompt itself)."""


def generate_optimized_prompt(
    task_description: str,
    top_model: dict,
    evaluation_scores: dict,
) -> str:
    """
    Generate an optimized system prompt for the top-ranked model.

    Args:
        task_description: The original task description
        top_model: Dict with model id, name, strengths, recommendation
        evaluation_scores: Aggregated evaluation scores for the model

    Returns:
        Optimized system prompt string
    """
    logger.info(f"Generating optimized prompt for {top_model.get('model_id', 'unknown')}...")

    messages = [
        {
            "role": "user",
            "content": PROMPT_OPTIMIZATION_TEMPLATE.format(
                task_description=task_description,
                top_model_id=top_model.get("model_id", "unknown"),
                top_model_name=top_model.get("model_id", "unknown").split("/")[-1],
                strengths=", ".join(top_model.get("strengths", ["Strong performance"])),
                overall_score=evaluation_scores.get("overall", 0),
                accuracy=evaluation_scores.get("accuracy", 0),
                hallucination=evaluation_scores.get("hallucination", 0),
                grounding=evaluation_scores.get("grounding", 0),
                clarity=evaluation_scores.get("clarity", 0),
            ),
        }
    ]

    prompt = call_judge(messages, temperature=0.5, max_tokens=1024)
    return prompt.strip()
