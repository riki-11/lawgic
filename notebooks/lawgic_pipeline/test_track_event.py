#!/usr/bin/env python3
"""
Smoke test for the researcher-controlled telemetry session flow:
/api/session/{start,end,status} and POST /api/track_event.

Prerequisites:
  - Lawgic API running (thesis-env + uvicorn) with LAWGIC_ADMIN_TOKEN set,
    and NO session currently active.
  - Same LAWGIC_ADMIN_TOKEN in this script's environment.
  - Optional LAWGIC_TEST_URL (default http://localhost:8000).

Usage:
    LAWGIC_ADMIN_TOKEN=... /Users/riki/anaconda3/envs/thesis-env/bin/python3 \\
        notebooks/lawgic_pipeline/test_track_event.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("LAWGIC_TEST_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("LAWGIC_ADMIN_TOKEN", "")
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "generated_files" / "interaction_logs" / "events.jsonl"


def call(method: str, path: str, body: dict | None = None, token: str | None = TOKEN) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Admin-Token"] = token
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def event(event_type: str, screen: str | None, n: int) -> tuple[int, dict]:
    return call("POST", "/api/track_event", {
        "event_type": event_type,
        "screen": screen,
        "client_id": "test-client",
        "client_seq": n,
        "client_ts": "2026-01-01T00:00:00.000Z",
        "data": {"n": n},
    })


def session_lines(session_id: str) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["session_id"] == session_id]


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main() -> int:
    check(bool(TOKEN), "LAWGIC_ADMIN_TOKEN must be set in the test environment")

    # (a) no token -> refused
    code, _ = call("POST", "/api/session/start", {"participant_id": "9999"}, token=None)
    check(code == 401, f"start without token: expected 401, got {code}")
    # non-ASCII token must be refused, not crash the comparison (was a 500)
    code, _ = call("POST", "/api/session/start", {"participant_id": "9999"}, token="é")
    check(code == 401, f"start with non-ASCII token: expected 401, got {code}")

    # (b) event before any session is dropped
    code, res = event("click", "profile", 0)
    check(code == 200 and res == {"logged": False, "reason": "no_active_session"}, f"pre-session event: {code} {res}")

    # (c) non-numeric participant id
    code, _ = call("POST", "/api/session/start", {"participant_id": "abc"})
    check(code == 422, f"participant 'abc': expected 422, got {code}")

    # (d) valid start
    code, started = call("POST", "/api/session/start", {"participant_id": "9999"})
    check(code == 200, f"start 9999: expected 200, got {code} {started}")
    sid = started["session_id"]

    # (e) second start conflicts
    code, res = call("POST", "/api/session/start", {"participant_id": "9999"})
    check(code == 409, f"second start: expected 409, got {code}")
    check("9999" in res.get("detail", ""), f"409 detail should name the active participant: {res}")

    # (f) two events are stamped and ordered
    for n, (etype, screen) in enumerate([("click", "profile"), ("screen_view", "results")], start=1):
        code, res = event(etype, screen, n)
        check(code == 200 and res == {"logged": True}, f"event {n}: {code} {res}")
    lines = session_lines(sid)
    events = [r for r in lines if r["event_type"] in ("click", "screen_view")]
    check(len(events) == 2, f"expected 2 client events for session, found {len(events)}")
    check(all(r["participant_id"] == "9999" for r in events), "events not stamped with participant 9999")
    check([r["seq"] for r in events] == [1, 2], f"seq should be [1, 2], got {[r['seq'] for r in events]}")
    check(0 <= events[0]["t_ms"] <= events[1]["t_ms"], f"t_ms not non-negative/increasing: {[r['t_ms'] for r in events]}")
    check(lines[0]["event_type"] == "session_start" and lines[0]["seq"] == 0, "first line should be session_start with seq 0")

    # (g) status
    code, status = call("GET", "/api/session/status")
    check(code == 200 and status["active"] is True, f"status: {code} {status}")
    check(status["event_count"] == 2, f"status event_count: {status}")
    check(status["current_screen"] == "results", f"status current_screen: {status}")

    # (h) clients cannot inject server-only event types
    code, _ = event("session_start", None, 3)
    check(code == 422, f"client session_start: expected 422, got {code}")

    # (i) end
    code, res = call("POST", "/api/session/end", {"note": "smoke test"})
    check(code == 200, f"end: expected 200, got {code} {res}")
    end_lines = [r for r in session_lines(sid) if r["event_type"] == "session_end"]
    check(len(end_lines) == 1, f"expected 1 session_end line, found {len(end_lines)}")
    check(end_lines[0]["data"]["note"] == "smoke test" and end_lines[0]["data"]["duration_ms"] >= 0, f"session_end data: {end_lines[0]['data']}")
    n_lines = len(session_lines(sid))

    # (j) events after end are dropped
    code, res = event("click", "results", 4)
    check(code == 200 and res == {"logged": False, "reason": "no_active_session"}, f"post-end event: {code} {res}")
    check(len(session_lines(sid)) == n_lines, "post-end event added a log line")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
