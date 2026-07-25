"""
Plain-language ToS clause explanations via Ollama (local or cloud-routed).

Called by POST /api/explain_tos_scores after Legal-BERT classification. Filters
clauses by harm class, then generates a ToS;DR-style point title and description
for each matching clause.

Environment (loaded by api/server.py from repo-root .env):
    VITE_OLLAMA_BASE_URL  — default http://localhost:11434
    VITE_OLLAMA_MODEL     — default gemma4:31b-cloud
                            offline fallback: gemma4:e4b (local, ~30x slower)

Public API:
    normalize_harm_filter()  — resolve user aliases to canonical harm classes
    explain_clauses()        — filter + sequential Ollama enrichment
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import ollama

logger = logging.getLogger(__name__)

# Canonical harm classes produced by Legal-BERT (api/server.py HARM_CLASS_NAMES values)
HARM_CLASS_HARMFUL = "Harmful"
HARM_CLASS_NEUTRAL = "Neutral"
HARM_CLASS_FAIR = "Fair"

# UI-facing labels (ToS;DR style)
HARM_LABEL_BY_CLASS: dict[str, str] = {
    HARM_CLASS_HARMFUL: "bad",
    HARM_CLASS_NEUTRAL: "neutral",
    HARM_CLASS_FAIR: "good",
}

# Accepted harm_filter aliases → canonical harm_class
#   bad, harmful, -1  → Harmful
#   neutral, 0          → Neutral
#   good, fair, +1      → Fair
HARM_FILTER_ALIASES: dict[str, str] = {
    "bad": HARM_CLASS_HARMFUL,
    "harmful": HARM_CLASS_HARMFUL,
    "-1": HARM_CLASS_HARMFUL,
    "neutral": HARM_CLASS_NEUTRAL,
    "0": HARM_CLASS_NEUTRAL,
    HARM_CLASS_NEUTRAL.lower(): HARM_CLASS_NEUTRAL,
    "good": HARM_CLASS_FAIR,
    "fair": HARM_CLASS_FAIR,
    "+1": HARM_CLASS_FAIR,
    HARM_CLASS_FAIR.lower(): HARM_CLASS_FAIR,
    HARM_CLASS_HARMFUL.lower(): HARM_CLASS_HARMFUL,
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:31b-cloud"
# Offline fallback: set VITE_OLLAMA_MODEL=gemma4:e4b to run fully local.
FALLBACK_OLLAMA_MODEL = "gemma4:e4b"

EXPLAIN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["title", "description"],
}


class OllamaExplainError(Exception):
    """Raised when Ollama is unreachable or returns unusable output."""


class ClauseExplainError(Exception):
    """Raised when a single clause cannot be explained after retries."""

    def __init__(self, chunk_id: int, message: str):
        super().__init__(message)
        self.chunk_id = chunk_id


def get_ollama_config() -> tuple[str, str]:
    """Read Ollama host and model from environment."""
    base_url = os.getenv("VITE_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model = os.getenv("VITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return base_url, model


def normalize_harm_filter(harm_filter: list[str]) -> set[str]:
    """
    Resolve user-supplied harm filter tokens to canonical harm class names.

    Examples:
        ["bad"]       → {"Harmful"}
        ["harmful", "neutral"] → {"Harmful", "Neutral"}
        ["fair"]      → {"Fair"}
    """
    if not harm_filter:
        return {HARM_CLASS_HARMFUL}

    resolved: set[str] = set()
    for token in harm_filter:
        key = token.strip().lower()
        if not key:
            continue
        canonical = HARM_FILTER_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unknown harm_filter value: {token!r}. "
                f"Accepted: bad, harmful, neutral, good, fair (or -1, 0, +1)"
            )
        resolved.add(canonical)

    if not resolved:
        raise ValueError("harm_filter must contain at least one valid harm class")

    return resolved


def build_explain_prompt(
    clause_text: str,
    predicted_topics: list[str],
    harm_class: str,
) -> str:
    """
    Build a compact user prompt for one clause.

    Intent: ToS;DR-style plain-language point. Guidelines are embedded in the
    system message; this message supplies only the clause context.
    """
    topics_str = ", ".join(predicted_topics) if predicted_topics else "unclassified"
    return (
        f"Topics: {topics_str}\n"
        f"Harm rating: {harm_class}\n"
        f"Clause:\n{clause_text}"
    )


def build_explain_system_prompt() -> str:
    """System prompt encoding style and output constraints for clause explanations."""
    return (
        "You write plain-language Terms of Service points in the style of ToS;DR. "
        "Given a clause, its topic labels, and a harm rating, return JSON with "
        '"title" and "description".\n'
        "title: one short takeaway in simple words (~15 words max). "
        'Use "this service" — never a company name.\n'
        "description: 1–3 sentences on what the clause means for users. "
        "Direct statements only. No meta-commentary about scores or ratings. "
        "Do not address the reader. Do not repeat the clause verbatim."
    )


def _strip_code_fence(raw: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one.

    Ollama's `format` schema constraint is only enforced for locally served
    models. Cloud-routed models (gemma4:31b-cloud and friends) are proxied
    upstream with the constraint dropped, so they return the JSON wrapped in a
    markdown fence. Models that honour `format` emit no fence and pass through
    this function unchanged.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text[:4].lower() == "json":
        text = text[4:]
    return text.strip().removesuffix("```").strip()


def _parse_explain_response(raw: str) -> dict[str, str]:
    """Parse and validate Ollama JSON output for title + description."""
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Model output must be a JSON object")

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()

    if not title or not description:
        raise ValueError("Both title and description must be non-empty strings")

    return {"title": title, "description": description}


def explain_clause(
    clause_text: str,
    predicted_topics: list[str],
    harm_class: str,
    *,
    chunk_id: int,
    base_url: str,
    model: str,
    max_retries: int = 1,
) -> dict[str, str]:
    """
    Generate a point title and description for one clause via Ollama.

    Returns {"title": "...", "description": "..."}.
    Retries once on parse failure; raises ClauseExplainError if still invalid.
    """
    client = ollama.Client(host=base_url)
    messages = [
        {"role": "system", "content": build_explain_system_prompt()},
        {
            "role": "user",
            "content": build_explain_prompt(clause_text, predicted_topics, harm_class),
        },
    ]

    last_error: str | None = None
    attempts = max_retries + 1

    for attempt in range(attempts):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                format=EXPLAIN_JSON_SCHEMA,
                options={"temperature": 0.3},
            )
            content = response["message"]["content"]
            return _parse_explain_response(content)
        except (ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
            logger.warning(
                "Clause %d explain attempt %d/%d failed: %s",
                chunk_id,
                attempt + 1,
                attempts,
                last_error,
            )
        except Exception as exc:
            raise OllamaExplainError(
                f"Ollama request failed ({base_url}, model={model}): {exc}"
            ) from exc

    raise ClauseExplainError(
        chunk_id,
        f"Failed to explain clause {chunk_id} after {attempts} attempts: {last_error}",
    )


def _primary_topic(predicted_topics: list[str]) -> str:
    return predicted_topics[0] if predicted_topics else "unclassified"


def explain_clauses(
    clauses: list[dict[str, Any]],
    harm_filter: list[str],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """
    Filter clauses by harm class, then enrich each match with Ollama explanations.

    Args:
        clauses: BERT output from /api/analyze_tos (harm_class, text, etc.).
        harm_filter: Tokens like ["harmful"] or ["bad", "neutral"].

    Returns:
        (points, model_name, normalized_filter_classes)
        points contains only clauses that matched harm_filter, each with
        point_title, description, quoted_text, harm_label, primary_topic, etc.
    """
    allowed_classes = normalize_harm_filter(harm_filter)
    order = {HARM_CLASS_HARMFUL: 0, HARM_CLASS_NEUTRAL: 1, HARM_CLASS_FAIR: 2}
    normalized_filter = sorted(allowed_classes, key=lambda c: order[c])

    filtered = [c for c in clauses if c.get("harm_class") in allowed_classes]
    base_url, model = get_ollama_config()

    logger.info(
        "Explaining %d/%d clauses (filter=%s, model=%s)",
        len(filtered),
        len(clauses),
        allowed_classes,
        model,
    )

    points: list[dict[str, Any]] = []

    for idx, clause in enumerate(filtered, start=1):
        chunk_id = clause["chunk_id"]
        logger.info("Enriching clause %d/%d (chunk_id=%d)", idx, len(filtered), chunk_id)

        explained = explain_clause(
            clause_text=clause["text"],
            predicted_topics=clause.get("predicted_topics", []),
            harm_class=clause["harm_class"],
            chunk_id=chunk_id,
            base_url=base_url,
            model=model,
        )

        harm_class = clause["harm_class"]
        points.append({
            "chunk_id": chunk_id,
            "primary_topic": _primary_topic(clause.get("predicted_topics", [])),
            "predicted_topics": clause.get("predicted_topics", []),
            "harm_class": harm_class,
            "harm_label": HARM_LABEL_BY_CLASS.get(harm_class, "neutral"),
            "harm_confidence": clause.get("harm_confidence", 0.0),
            "point_title": explained["title"],
            "description": explained["description"],
            "quoted_text": clause["text"],
        })

    return points, model, normalized_filter


if __name__ == "__main__":
    # Offline check that both model families parse. Run: python -m api.llm_interpreter
    expected = {"title": "a", "description": "b"}
    payloads = [
        '{"title": "a", "description": "b"}',                 # local, format honoured
        '```json\n{"title": "a", "description": "b"}\n```',   # cloud-routed, fenced
        '```\n{"title": "a", "description": "b"}\n```',       # fenced, no language tag
    ]
    for payload in payloads:
        assert _parse_explain_response(payload) == expected, payload

    for bad in ("not json at all", '{"title": "", "description": "b"}'):
        try:
            _parse_explain_response(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")

    print("llm_interpreter self-check: fence stripping + validation ✓")
