"""Knowledge base tool schema and search function for candidate LLMs."""

import logging
import re

logger = logging.getLogger(__name__)

# Tool schema exposed to candidate models via OpenRouter's tools API
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for information. Returns the most relevant section.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what information you need",
                }
            },
            "required": ["query"],
        },
    },
}


def search_knowledge_base(query: str, knowledge_doc: dict[str, str]) -> str:
    """
    Simple keyword matching search over the knowledge document.

    Tokenizes the query into keywords, scores each section by keyword overlap,
    and returns the best-matching section content.

    Args:
        query: Search query from the candidate model
        knowledge_doc: Dict mapping section_name -> section_content

    Returns:
        Best-matching section content, or a "no results" message
    """
    if not knowledge_doc:
        return "No relevant information found."

    # Tokenize query into lowercase keywords, filtering short/common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "above", "below", "and", "or",
        "but", "not", "no", "nor", "so", "if", "then", "than", "too", "very",
        "just", "that", "this", "it", "its", "what", "which", "who", "whom",
        "how", "when", "where", "why", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "only", "own", "same",
    }
    query_words = set(re.findall(r"\w+", query.lower())) - stop_words
    query_words = {w for w in query_words if len(w) > 2}

    if not query_words:
        # Fall back to first section if query is too vague
        first_key = next(iter(knowledge_doc))
        return f"[{first_key}]\n{knowledge_doc[first_key]}"

    best_score = 0.0
    best_section = ""
    best_name = ""

    for section_name, section_content in knowledge_doc.items():
        # Combine section name and content for matching
        section_text = f"{section_name} {section_content}".lower()
        section_words = set(re.findall(r"\w+", section_text))

        # Score by keyword overlap ratio
        overlap = query_words & section_words
        score = len(overlap) / len(query_words) if query_words else 0.0

        if score > best_score:
            best_score = score
            best_section = section_content
            best_name = section_name

    # Threshold: at least 20% keyword overlap to be considered a match
    if best_score < 0.2:
        logger.debug(
            f"KB search: no match above threshold for query '{query}' "
            f"(best score: {best_score:.2f})"
        )
        return "No relevant information found."

    logger.debug(
        f"KB search: query='{query}' -> section='{best_name}' "
        f"(score={best_score:.2f})"
    )
    return f"[{best_name}]\n{best_section}"
