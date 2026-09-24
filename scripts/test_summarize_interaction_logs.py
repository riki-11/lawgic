#!/usr/bin/env python3
"""
Self-check for scripts/summarize_interaction_logs.py. Builds a fixture JSONL
in a temp dir, runs main() and the CLI on it, and asserts hand-computed values.

Usage:
    /Users/riki/anaconda3/envs/thesis-env/bin/python3 scripts/test_summarize_interaction_logs.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize_interaction_logs as S  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


class Session:
    """Builds one session's event lines, stamping seq in call order."""

    def __init__(self, pid: str, sid: str):
        self.pid, self.sid, self.seq, self.lines = pid, sid, 0, []

    def ev(self, t_ms, etype, screen=None, data=None, commit=None):
        line = {"server_ts": f"2026-09-25T10:00:{self.seq:02d}+00:00", "participant_id": self.pid,
                "session_id": self.sid, "seq": self.seq, "t_ms": t_ms, "event_type": etype,
                "screen": screen, "client_id": "c", "client_seq": self.seq,
                "client_ts": None, "app_commit": commit, "data": data or {}}
        self.seq += 1
        self.lines.append(line)


def cards(**kw):  # cards(c1=(visible, expanded), ...)
    return {k: {"visible_ms": v, "expanded_ms": e} for k, (v, e) in kw.items()}


def session_a() -> Session:
    a = Session("101", "sA")
    a.ev(0, "session_start", data={"model_dir": "models/x"})
    a.ev(100, "client_hello", "profile", {"ollama_model": "llama3", "risk_scorer": "lr",
                                          "diff_inject_classifier_context": True}, commit="abc123")
    a.ev(150, "client_hello", "profile", {"ollama_model": "other", "risk_scorer": "rf",
                                          "diff_inject_classifier_context": False}, commit="zzz")
    a.ev(160, "screen_view", "profile", {"duration_ms": 3, "active_ms": 0})  # artifact, no heartbeat
    a.ev(500, "click", "profile", {"tag": "button", "trackId": "t", "label": "go"})
    a.ev(600, "profile_selection", "profile", {"field": "mode", "value": "case_study"})
    a.ev(700, "profile_selection", "profile", {"field": "service", "value": "Netflix"})
    a.ev(800, "profile_selection", "profile", {"field": "service", "value": "Spotify"})
    a.ev(900, "profile_selection", "profile", {"field": "role", "value": "student"})
    a.ev(1000, "profile_selection", "profile", {"field": "concerns", "value": ["privacy", "cost"]})
    a.ev(1100, "profile_selection", "profile", {"field": "context", "length": 12, "non_empty": True})
    a.ev(5000, "heartbeat", "profile", {"screen": "profile", "screen_ms": 5000, "active_ms": 4000,
                                        "under": [], "visible": True})
    a.ev(10000, "heartbeat", "profile", {"screen": "profile", "screen_ms": 10000, "active_ms": 7000,
                                         "under": [], "visible": True})
    a.ev(12000, "screen_view", "profile", {"duration_ms": 12000, "active_ms": 8000})
    a.ev(17000, "heartbeat", "analyzing", {"screen": "analyzing", "screen_ms": 5000, "active_ms": 1000,
                                           "under": [], "visible": True})
    a.ev(19300, "screen_view", "analyzing", {"duration_ms": 7300, "active_ms": 1200})
    a.ev(24000, "analysis_result", "analyzing", {"service": "Spotify", "n_changes": 3, "cards": [
        {"card_id": "c1", "title": "Data sharing", "risk_grade": "harmful", "category": "Privacy",
         "position": 1, "change_type": "added"},
        {"card_id": "c2", "title": "Refunds", "risk_grade": "neutral", "category": "Billing",
         "position": 2, "change_type": "modified"},
        {"card_id": "c3", "title": "Cancellation", "risk_grade": "fair", "category": "Billing",
         "position": 3, "change_type": "removed"}]})
    a.ev(25000, "click", "results", {"tag": "div", "trackId": "t2", "label": "x"})
    a.ev(30000, "heartbeat", "results", {"screen": "results", "screen_ms": 5000, "active_ms": 4000,
                                         "under": [], "visible": True,
                                         "cards": cards(c1=(3000, 0), c2=(800, 0))})
    a.ev(32000, "change_card_expand", "results", {"card_id": "c1", "title": "Data sharing", "position": 1})
    a.ev(32001, "original_clause_open", "results", {"card_id": "c1", "which": "old"})
    a.ev(36000, "change_card_collapse", "results", {"card_id": "c1"})
    a.ev(38000, "change_card_expand", "results", {"card_id": "c1"})
    a.ev(38001, "original_clause_open", "results", {"card_id": "c1", "which": "new"})
    a.ev(41000, "change_card_expand", "results", {"card_id": "c2"})
    a.ev(41500, "category_toggle", "results", {"category": "Billing", "open": True, "n_cards": 2})
    a.ev(41800, "category_toggle", "results", {"category": "Billing", "open": False, "n_cards": 2})
    a.ev(35000, "heartbeat", "results", {"screen": "results", "screen_ms": 10000, "active_ms": 8000,
                                         "under": [], "visible": True,
                                         "cards": cards(c1=(6000, 4000), c2=(1500, 0))})
    a.ev(45000, "ask_ai_open", "results", {"source": "card", "card_title": "Data sharing"})
    a.ev(50000, "heartbeat", "Ask AI", {
        "screen": "Ask AI", "screen_ms": 5000, "active_ms": 3000, "visible": True,
        "under": [{"screen": "results", "screen_ms": 15000, "active_ms": 10000,
                   "cards": cards(c1=(7000, 4000), c2=(1500, 500))}]})
    a.ev(52000, "ask_ai_message", "Ask AI", {"text": "why?", "length": 20, "index": 0, "source": "typed"})
    a.ev(53000, "ask_ai_message", "Ask AI", {"text": "more", "length": 30, "index": 1, "source": "suggested"})
    a.ev(55000, "heartbeat", "Ask AI", {
        "screen": "Ask AI", "screen_ms": 10000, "active_ms": 6000, "visible": True,
        "under": [{"screen": "results", "screen_ms": 20000, "active_ms": 12000,
                   "cards": cards(c1=(8000, 5000), c2=(1500, 500))}]})
    a.ev(57000, "screen_view", "Ask AI", {"duration_ms": 12000, "active_ms": 7000})
    a.ev(60000, "heartbeat", "results", {"screen": "results", "screen_ms": 25000, "active_ms": 14000,
                                         "under": [], "visible": True,
                                         "cards": cards(c1=(9000, 5500), c2=(1500, 500))})
    a.ev(60100, "screen_view", "results", {"duration_ms": 3, "active_ms": 0})  # artifact over live record
    a.ev(62000, "screen_view", "results", {"duration_ms": 27000, "active_ms": 15000,
                                           "cards": cards(c1=(9500, 5500), c2=(1500, 500))})
    a.ev(63000, "preview_toggle", "preview", {"service": "Spotify", "version": "old"})
    a.ev(68000, "heartbeat", "preview", {"screen": "preview", "screen_ms": 5000, "active_ms": 3000,
                                         "under": [], "visible": True})
    a.ev(73000, "heartbeat", "preview", {"screen": "preview", "screen_ms": 10000, "active_ms": 6000,
                                         "under": [], "visible": True})
    a.ev(80000, "heartbeat", "preview", {"screen": "preview", "screen_ms": 5000, "active_ms": 1000,
                                         "under": [], "visible": True})  # re-entered: resets lower
    a.ev(81000, "screen_view", "preview", {"duration_ms": 6000, "active_ms": 2000})
    a.ev(82000, "preview_toggle", "preview", {"service": "Spotify", "version": "new"})
    a.ev(200000, "session_end", data={"end_reason": "researcher_ended", "duration_ms": 200000, "note": "ok"})
    return a


def session_b() -> Session:
    b = Session("102", "sB")
    b.ev(0, "session_start", data={"model_dir": "models/y"})
    b.ev(100, "client_hello", "profile", {"ollama_model": "qwen", "risk_scorer": "rf",
                                          "diff_inject_classifier_context": False}, commit="def456")
    b.ev(200, "profile_selection", "profile", {"field": "role", "value": "parent"})
    b.ev(300, "profile_selection", "profile", {"field": "concerns", "value": ["cost"]})
    b.ev(5100, "heartbeat", "profile", {"screen": "profile", "screen_ms": 5000, "active_ms": 5000,
                                        "under": [], "visible": True})
    b.ev(7000, "change_card_expand", "results", {"card_id": "x7", "title": "T7", "risk_grade": "harmful",
                                                 "category": "Cat", "position": 4})
    b.ev(8000, "heartbeat", "results", {"screen": "results", "screen_ms": 4000, "active_ms": 3000,
                                        "under": [], "visible": True, "cards": cards(x7=(2000, 1000))})
    b.ev(8500, "heartbeat", "results", {"screen": "results", "screen_ms": 1000, "active_ms": 800,
                                        "under": [], "visible": True, "cards": cards(x7=(500, 0))})
    b.ev(9000, "click", "results", {"tag": "a", "trackId": "t", "label": "l"})
    # no session_end; file order differs from seq order for the two last-but-one events
    b.lines[-2], b.lines[-3] = b.lines[-3], b.lines[-2]
    return b


def session_c() -> Session:
    c = Session("9", "sC")
    c.ev(0, "session_start", data={"model_dir": "models/z"})
    return c


def session_d() -> Session:
    # The participant's profile page was open 25 s BEFORE the researcher pressed
    # Start, so its cumulative screen_ms includes time outside the session.
    d = Session("200", "sD")
    d.ev(0, "session_start", data={"model_dir": "models/z"})
    d.ev(300, "client_hello", "profile")
    d.ev(5000, "heartbeat", "profile", {"screen": "profile", "screen_ms": 30000, "active_ms": 25000, "under": [], "cards": {}})
    d.ev(10000, "heartbeat", "profile", {"screen": "profile", "screen_ms": 35000, "active_ms": 30000, "under": [], "cards": {}})
    d.ev(12000, "screen_view", "profile", {"duration_ms": 37000, "active_ms": 32000})
    d.ev(17000, "heartbeat", "results", {"screen": "results", "screen_ms": 5000, "active_ms": 4000, "under": [], "cards": {}})
    d.ev(18000, "session_end", data={"end_reason": "researcher_ended", "duration_ms": 18000, "note": None})
    return d


def write_fixture(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in (session_a(), session_b(), session_c(), session_d()):
            for i, line in enumerate(s.lines):
                f.write(json.dumps(line) + "\n")
                if s.sid == "sA" and i == 10:
                    f.write("{this is not json\n")  # malformed line


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expect(row: dict, want: dict, label: str) -> None:
    for k, v in want.items():
        check(row[k] == v, f"{label}.{k}: expected {v!r}, got {row[k]!r}")


def verify(out: Path) -> None:
    sess = read_csv(out / "interaction_sessions.csv")
    cds = read_csv(out / "interaction_cards.csv")
    check(list(sess[0].keys()) == S.SESSION_COLS, "session columns")
    check(list(cds[0].keys()) == S.CARD_COLS, "card columns")
    check([r["participant_id"] for r in sess] == ["9", "101", "102", "200"], "session order (numeric participant)")

    # D: profile was open 25 s before Start; only the in-session 12 s counts (37 s - 25 s).
    # Active time is capped at in-session wall time (an upper bound for such screens).
    expect(sess[3], {"session_id": "sD", "duration_ms": "18000", "n_client_events": "5", "n_heartbeats": "3",
                     "profile_wall_s": "12.0", "profile_active_s": "12.0",
                     "results_wall_s": "5.0", "results_active_s": "4.0"}, "D")

    expect(sess[0], {"session_id": "sC", "ended": "false", "n_client_events": "0", "duration_ms": "0",
                     "model_dir": "models/z", "results_wall_s": "0.0", "time_to_results_s": "",
                     "n_cards_shown": ""}, "C")

    expect(sess[1], {
        "session_id": "sA", "ended": "true", "end_reason": "researcher_ended", "duration_ms": "200000",
        "note": "ok", "app_commit": "abc123", "model_dir": "models/x", "ollama_model": "llama3",
        "risk_scorer": "lr", "diff_inject_classifier_context": "true",
        "n_client_events": "42", "n_heartbeats": "11",
        "profile_wall_s": "12.0", "profile_active_s": "8.0",
        "analyzing_wall_s": "7.3", "analyzing_active_s": "1.2",
        "results_wall_s": "27.0", "results_active_s": "15.0",
        "preview_wall_s": "16.0", "preview_active_s": "8.0",
        "ask_ai_wall_s": "12.0", "ask_ai_active_s": "7.0",
        "time_to_results_s": "25.0",
        "profile_service": "Spotify", "profile_role": "student", "profile_concerns": "privacy;cost",
        "profile_mode": "case_study", "profile_context_length": "12",
        "n_cards_shown": "3", "n_harmful_shown": "1",
        "n_cards_expanded_unique": "2", "n_expands": "3", "n_original_clause_open_unique": "1",
        "n_cards_seen_1s": "2", "n_category_toggles": "2", "n_preview_toggles": "2",
        "n_ask_ai_opens": "1", "n_ask_ai_messages": "2", "ask_ai_chars": "50", "n_clicks": "2",
    }, "A")

    expect(sess[2], {
        "session_id": "sB", "ended": "false", "end_reason": "", "note": "", "duration_ms": "9000",
        "app_commit": "def456", "model_dir": "models/y", "ollama_model": "qwen",
        "diff_inject_classifier_context": "false", "n_client_events": "8", "n_heartbeats": "3",
        "profile_wall_s": "5.0", "profile_active_s": "5.0",
        "results_wall_s": "5.0", "results_active_s": "3.8",  # two results entries: 4.0+1.0, 3.0+0.8
        "time_to_results_s": "7.0", "profile_service": "", "profile_role": "parent",
        "profile_concerns": "cost", "n_cards_shown": "", "n_harmful_shown": "",
        "n_cards_expanded_unique": "1", "n_expands": "1", "n_original_clause_open_unique": "0",
        "n_cards_seen_1s": "1", "n_clicks": "1",
    }, "B")

    check(len(cds) == 4, f"card row count {len(cds)}")
    want = [
        {"session_id": "sA", "card_id": "c1", "title": "Data sharing", "category": "Privacy",
         "position": "1", "risk_grade": "harmful", "change_type": "added", "n_expands": "2",
         "expanded_ms": "5500", "visible_ms": "9500", "opened_original": "true", "first_expand_t_ms": "32000"},
        {"session_id": "sA", "card_id": "c2", "title": "Refunds", "category": "Billing",
         "position": "2", "risk_grade": "neutral", "change_type": "modified", "n_expands": "1",
         "expanded_ms": "500", "visible_ms": "1500", "opened_original": "false", "first_expand_t_ms": "41000"},
        {"session_id": "sA", "card_id": "c3", "title": "Cancellation", "category": "Billing",
         "position": "3", "risk_grade": "fair", "change_type": "removed", "n_expands": "0",
         "expanded_ms": "0", "visible_ms": "0", "opened_original": "false", "first_expand_t_ms": ""},
        {"session_id": "sB", "card_id": "x7", "title": "T7", "category": "Cat", "position": "4",
         "risk_grade": "harmful", "change_type": "", "n_expands": "1", "expanded_ms": "1000",
         "visible_ms": "2500", "opened_original": "false", "first_expand_t_ms": "7000"},  # 2000 + 500
    ]
    for r, w in zip(cds, want):
        expect(r, w, f"card {w['card_id']}")


def main() -> int:
    script = HERE / "summarize_interaction_logs.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log = tmp / "events.jsonl"
        write_fixture(log)

        check(S.main(["--log", str(log), "--out-dir", str(tmp / "inproc")]) == 0, "main() return code")
        verify(tmp / "inproc")

        r = subprocess.run([sys.executable, str(script), "--log", str(log), "--out-dir", str(tmp / "cli")],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"CLI exit {r.returncode}: {r.stderr}")
        check("skipped 1 blank/malformed" in r.stdout, f"malformed count in output: {r.stdout!r}")
        verify(tmp / "cli")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
