#!/usr/bin/env python3
"""
End-to-end test: Legal-BERT classification + Ollama score explanations.

Chains two API calls:
  1. POST /api/analyze_tos        — upload .txt, get clause classifications
  2. POST /api/explain_tos_scores — enrich harm-filtered clauses with titles/descriptions

Prerequisites:
  - Lawgic API running on localhost:8000 (thesis-env + uvicorn)
  - Ollama running with gemma4:e4b pulled (see repo-root .env)

Usage:
    /Users/riki/anaconda3/envs/thesis-env/bin/python3 \\
        notebooks/lawgic_pipeline/test_llm_inference.py

    /Users/riki/anaconda3/envs/thesis-env/bin/python3 \
        notebooks/lawgic_pipeline/test_llm_inference.py \
        --harm-filter bad --file data/new_tos/apollo_io.txt

Full commands to run: 
/Users/riki/anaconda3/envs/thesis-env/bin/python3 -m uvicorn api.server:app \
  --host 0.0.0.0 --port 8000 --reload
/Users/riki/anaconda3/envs/thesis-env/bin/python3 \
  notebooks/lawgic_pipeline/test_llm_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_FILE = REPO_ROOT / "data" / "new_tos" / "apollo_io.txt"
DEFAULT_SERVICE_NAME = "Apollo.io"
DEFAULT_HARM_FILTER = "harmful"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Legal-BERT + Ollama explanation pipeline via Lawgic API",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Lawgic API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to .txt ToS file to upload",
    )
    parser.add_argument(
        "--service-name",
        default=DEFAULT_SERVICE_NAME,
        help="Service display name sent to the API",
    )
    parser.add_argument(
        "--harm-filter",
        default=DEFAULT_HARM_FILTER,
        help="Comma-separated harm filter (default: harmful). E.g. bad,neutral",
    )
    return parser.parse_args()


def analyze_tos(api_url: str, file_path: Path, service_name: str) -> dict:
    """Step 1: upload file and run Legal-BERT classification."""
    url = f"{api_url.rstrip('/')}/api/analyze_tos"
    with file_path.open("rb") as f:
        response = requests.post(
            url,
            files={"file": (file_path.name, f, "text/plain")},
            data={"service_name": service_name},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def explain_scores(
    api_url: str,
    service_name: str,
    clauses: list[dict],
    harm_filter: list[str],
) -> dict:
    """Step 2: generate plain-language explanations for harm-filtered clauses."""
    url = f"{api_url.rstrip('/')}/api/explain_tos_scores"
    payload = {
        "service_name": service_name,
        "clauses": clauses,
        "harm_filter": harm_filter,
    }
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def print_prerequisites_hint() -> None:
    print(
        "\nPrerequisites:\n"
        "  1. Start API:  python -m uvicorn api.server:app --host 0.0.0.0 --port 8000\n"
        "  2. Start Ollama and pull model:  ollama pull gemma4:e4b\n"
        "  3. Check .env has VITE_OLLAMA_BASE_URL and VITE_OLLAMA_MODEL\n",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    file_path = args.file.resolve()
    harm_filter = [t.strip() for t in args.harm_filter.split(",") if t.strip()]

    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    print(f"API:          {args.api_url}")
    print(f"File:         {file_path}")
    print(f"Service:      {args.service_name}")
    print(f"Harm filter:  {harm_filter}")
    print()

    try:
        print("Step 1: POST /api/analyze_tos ...")
        analyze_result = analyze_tos(args.api_url, file_path, args.service_name)
        clauses = analyze_result["clauses"]
        print(f"  -> {len(clauses)} clauses classified")

        harmful_count = sum(1 for c in clauses if c["harm_class"] == "Harmful")
        neutral_count = sum(1 for c in clauses if c["harm_class"] == "Neutral")
        fair_count = sum(1 for c in clauses if c["harm_class"] == "Fair")
        print(f"  -> Harmful: {harmful_count}, Neutral: {neutral_count}, Fair: {fair_count}")
        print()

        print("Step 2: POST /api/explain_tos_scores ...")
        explain_result = explain_scores(
            args.api_url,
            args.service_name,
            clauses,
            harm_filter,
        )
        print(f"  -> {explain_result['point_count']} points explained (model: {explain_result['llm_model']})")
        print()

        for i, point in enumerate(explain_result["points"][:3], start=1):
            print(f"--- Point {i} (chunk_id={point['chunk_id']}, {point['harm_label']}) ---")
            print(f"Topic:       {point['primary_topic']}")
            print(f"Title:       {point['point_title']}")
            print(f"Description: {point['description']}")
            print(f"Confidence:  {point['harm_confidence']}")
            print()

        if explain_result["point_count"] > 3:
            print(f"... and {explain_result['point_count'] - 3} more points")

        print("\nFull explain response (JSON):")
        print(json.dumps(explain_result, indent=2))
        return 0

    except requests.ConnectionError:
        print("Error: could not connect to Lawgic API.", file=sys.stderr)
        print_prerequisites_hint()
        return 1
    except requests.HTTPError as exc:
        print(f"Error: API returned {exc.response.status_code}", file=sys.stderr)
        try:
            print(exc.response.json(), file=sys.stderr)
        except Exception:
            print(exc.response.text, file=sys.stderr)
        if exc.response.status_code in (502, 503):
            print_prerequisites_hint()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
