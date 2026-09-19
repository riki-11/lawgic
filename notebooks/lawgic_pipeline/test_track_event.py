#!/usr/bin/env python3
"""
Smoke test for POST /api/track_event — participant interaction telemetry.

Prerequisites:
  - Lawgic API running on localhost:8000 (thesis-env + uvicorn)

Usage:
    /Users/riki/anaconda3/envs/thesis-env/bin/python3 \\
        notebooks/lawgic_pipeline/test_track_event.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "generated_files" / "interaction_logs" / "events.jsonl"


def post_event(session_id: str, event_type: str, screen: str | None, data: dict) -> dict:
    resp = requests.post(
        f"{API_BASE}/api/track_event",
        json={
            "session_id": session_id,
            "event_type": event_type,
            "screen": screen,
            "client_ts": "2026-01-01T00:00:00.000Z",
            "data": data,
        },
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    session_id = f"test-{uuid.uuid4()}"

    resp1 = post_event(session_id, "session_start", None, {})
    assert resp1 == {"logged": True}, f"session_start not logged: {resp1}"

    resp2 = post_event(session_id, "click", "profile", {"tag": "button", "label": "Analyze"})
    assert resp2 == {"logged": True}, f"click not logged: {resp2}"

    resp3 = post_event(session_id, "screen_view", "profile", {"duration_ms": 4200})
    assert resp3 == {"logged": True}, f"screen_view not logged: {resp3}"

    assert LOG_PATH.exists(), f"log file was not created at {LOG_PATH}"
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    this_session_lines = [json.loads(line) for line in lines if session_id in line]
    assert len(this_session_lines) == 3, (
        f"expected 3 events for {session_id}, found {len(this_session_lines)}"
    )
    assert {e["event_type"] for e in this_session_lines} == {"session_start", "click", "screen_view"}
    assert this_session_lines[2]["data"]["duration_ms"] == 4200
    assert "server_ts" in this_session_lines[0]

    print(f"OK — 3 events round-tripped through {API_BASE} and {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
