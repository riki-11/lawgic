#!/usr/bin/env python3
"""
Summarize participant interaction telemetry (events.jsonl) into two CSVs:

  interaction_sessions.csv  one row per (participant_id, session_id)
  interaction_cards.csv     one row per (participant_id, session_id, card_id)

Screen-time rule: heartbeat screen_ms / active_ms / cards are CUMULATIVE per
screen ENTRY. Per session, in seq order, each screen keeps a current-entry
record holding the max seen. A heartbeat whose screen_ms is lower than the
record starts a new entry (the old one is finalized into the running total);
a screen_view finalizes the entry with max(record, screen_view values); any
record still open at the end of the session is finalized. Totals are sums over
entries. screen_view events under 100 ms are strict-mode artifacts and are
ignored. Per-card visible/expanded ms follow the same rule using results
entries as the unit.

Usage:
    python3 scripts/summarize_interaction_logs.py [--log PATH] [--out-dir DIR]

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "generated_files" / "interaction_logs"
SCREENS = ["profile", "analyzing", "results", "preview", "ask_ai"]
MIN_REAL_VIEW_MS = 100

SESSION_COLS = (
    ["participant_id", "session_id", "started_at", "ended", "end_reason", "duration_ms",
     "note", "app_commit", "model_dir", "ollama_model", "risk_scorer",
     "diff_inject_classifier_context", "n_client_events", "n_heartbeats"]
    + [f"{s}_{k}_s" for s in SCREENS for k in ("wall", "active")]
    + ["time_to_results_s", "profile_service", "profile_role", "profile_concerns",
       "profile_mode", "profile_context_length", "n_cards_shown", "n_harmful_shown",
       "n_cards_expanded_unique", "n_expands", "n_original_clause_open_unique",
       "n_cards_seen_1s", "n_category_toggles", "n_preview_toggles", "n_ask_ai_opens",
       "n_ask_ai_messages", "ask_ai_chars", "n_clicks"]
)
CARD_COLS = ["participant_id", "session_id", "card_id", "title", "category", "position",
             "risk_grade", "change_type", "n_expands", "expanded_ms", "visible_ms",
             "opened_original", "first_expand_t_ms"]


def num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


def as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def screen_key(name) -> str:
    return str(name).strip().lower().replace(" ", "_")


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return v


def secs(ms: float) -> str:
    return f"{ms / 1000:.1f}"


class ScreenTimes:
    """Per-screen cumulative-entry accounting (see module docstring)."""

    def __init__(self):
        self.rec: dict[str, dict] = {}
        self.tot: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.cards: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # results only

    def finalize(self, s: str) -> None:
        r = self.rec.pop(s, None)
        if r is None:
            return
        self.tot[s][0] += r["sm"]
        self.tot[s][1] += r["am"]
        if s == "results":
            for cid, (v, e) in r["cards"].items():
                self.cards[cid][0] += v
                self.cards[cid][1] += e

    def update(self, s: str, sm: float, am: float, cards, new_entry_on_drop: bool = True) -> None:
        r = self.rec.get(s)
        if r is not None and new_entry_on_drop and sm < r["sm"]:
            self.finalize(s)
            r = None
        if r is None:
            r = self.rec[s] = {"sm": 0.0, "am": 0.0, "cards": {}}
        r["sm"] = max(r["sm"], sm)
        r["am"] = max(r["am"], am)
        for cid, v in as_dict(cards).items():
            v = as_dict(v)
            c = r["cards"].setdefault(str(cid), [0.0, 0.0])
            c[0] = max(c[0], num(v.get("visible_ms")))
            c[1] = max(c[1], num(v.get("expanded_ms")))

    def heartbeat(self, ev_screen, d: dict) -> None:
        top = screen_key(d.get("screen") or ev_screen or "")
        if top:
            self.update(top, num(d.get("screen_ms")), num(d.get("active_ms")),
                        d.get("cards") if top == "results" else None)
        for row in d.get("under") or []:
            row = as_dict(row)
            if row.get("screen"):
                s = screen_key(row["screen"])
                self.update(s, num(row.get("screen_ms")), num(row.get("active_ms")),
                            row.get("cards") if s == "results" else None)

    def screen_view(self, ev_screen, d: dict) -> None:
        s = screen_key(ev_screen or d.get("screen") or "")
        dur = num(d.get("duration_ms"))
        if not s or dur < MIN_REAL_VIEW_MS:
            return  # strict-mode artifact: no-op, never clears a real record
        self.update(s, dur, num(d.get("active_ms")), d.get("cards") if s == "results" else None,
                    new_entry_on_drop=False)
        self.finalize(s)

    def close(self) -> None:
        for s in list(self.rec):
            self.finalize(s)


def summarize_session(pid: str, sid: str, evs: list[dict]):
    st = ScreenTimes()
    start = next((e for e in evs if e.get("event_type") == "session_start"), None)
    end = next((e for e in evs if e.get("event_type") == "session_end"), None)
    hello = next((e for e in evs if e.get("event_type") == "client_hello"), None)
    hd = as_dict(hello.get("data")) if hello else {}
    sd = as_dict(start.get("data")) if start else {}
    ed = as_dict(end.get("data")) if end else {}
    first_ts = next((e.get("server_ts") for e in evs if e.get("server_ts")), "")

    analysis = []  # analysis_result data dicts, in order
    prof: dict = {}
    n_client = n_hb = n_click = n_cat = n_prev = n_open = n_msg = 0
    chars = 0.0
    expands: dict[str, int] = defaultdict(int)
    first_expand: dict[str, float] = {}
    opened: set[str] = set()
    seen_info: dict[str, dict] = {}  # card metadata gleaned from card events
    t_results = None
    max_t = 0.0

    for e in evs:
        et = e.get("event_type")
        d = as_dict(e.get("data"))
        if isinstance(e.get("t_ms"), (int, float)):
            max_t = max(max_t, e["t_ms"])
        if et not in ("session_start", "session_end"):
            n_client += 1
        if t_results is None and screen_key(e.get("screen") or "") == "results" and isinstance(e.get("t_ms"), (int, float)):
            t_results = e["t_ms"]
        if et == "heartbeat":
            n_hb += 1
            st.heartbeat(e.get("screen"), d)
        elif et == "screen_view":
            st.screen_view(e.get("screen"), d)
        elif et == "click":
            n_click += 1
        elif et == "profile_selection":
            f = d.get("field")
            if f == "context":
                prof["context"] = d.get("length")
            elif f:
                prof[f] = d.get("value")
        elif et == "analysis_result":
            analysis.append(d)
        elif et in ("change_card_expand", "change_card_collapse", "original_clause_open"):
            cid = d.get("card_id")
            if cid is None:
                continue
            cid = str(cid)
            seen_info.setdefault(cid, {k: d.get(k) for k in ("title", "category", "position", "risk_grade")})
            if et == "change_card_expand":
                expands[cid] += 1
                if cid not in first_expand and isinstance(e.get("t_ms"), (int, float)):
                    first_expand[cid] = e["t_ms"]
            elif et == "original_clause_open":
                opened.add(cid)
        elif et == "category_toggle":
            n_cat += 1
        elif et == "preview_toggle":
            n_prev += 1
        elif et == "ask_ai_open":
            n_open += 1
        elif et == "ask_ai_message":
            n_msg += 1
            chars += num(d.get("length"))
    st.close()

    # ---- cards table
    info: dict[str, dict] = {}
    for a in analysis:
        for c in a.get("cards") or []:
            c = as_dict(c)
            if c.get("card_id") is not None:
                info.setdefault(str(c["card_id"]), c)
    for cid, meta in seen_info.items():
        info.setdefault(cid, meta)
    for cid in list(st.cards) + list(expands) + list(opened):
        info.setdefault(cid, {})
    card_rows = []
    for cid, c in info.items():
        vis, exp = st.cards.get(cid, (0.0, 0.0)) if cid in st.cards else (0.0, 0.0)
        card_rows.append({
            "participant_id": pid, "session_id": sid, "card_id": cid,
            "title": cell(c.get("title")), "category": cell(c.get("category")),
            "position": cell(c.get("position")), "risk_grade": cell(c.get("risk_grade")),
            "change_type": cell(c.get("change_type")),
            "n_expands": expands.get(cid, 0), "expanded_ms": int(exp), "visible_ms": int(vis),
            "opened_original": cell(cid in opened),
            "first_expand_t_ms": cell(first_expand.get(cid)),
        })

    first = analysis[0] if analysis else None
    shown = [as_dict(c) for c in (first or {}).get("cards") or []]
    row = {
        "participant_id": pid, "session_id": sid,
        "started_at": (start or {}).get("server_ts") or first_ts or "",
        "ended": cell(end is not None),
        "end_reason": cell(ed.get("end_reason")),
        "duration_ms": cell(ed.get("duration_ms") if end and ed.get("duration_ms") is not None else int(max_t)),
        "note": cell(ed.get("note")),
        "app_commit": next((e["app_commit"] for e in evs if e.get("app_commit")), ""),
        "model_dir": cell(sd.get("model_dir")),
        "ollama_model": cell(hd.get("ollama_model")),
        "risk_scorer": cell(hd.get("risk_scorer")),
        "diff_inject_classifier_context": cell(hd.get("diff_inject_classifier_context")),
        "n_client_events": n_client, "n_heartbeats": n_hb,
        "time_to_results_s": "" if t_results is None else secs(t_results),
        "profile_service": cell(prof.get("service")),
        "profile_role": cell(prof.get("role")),
        "profile_concerns": ";".join(str(x) for x in prof["concerns"]) if isinstance(prof.get("concerns"), list) else cell(prof.get("concerns")),
        "profile_mode": cell(prof.get("mode")),
        "profile_context_length": cell(prof.get("context")),
        "n_cards_shown": len(shown) if first is not None else "",
        "n_harmful_shown": sum(1 for c in shown if c.get("risk_grade") == "harmful") if first is not None else "",
        "n_cards_expanded_unique": len(expands), "n_expands": sum(expands.values()),
        "n_original_clause_open_unique": len(opened),
        "n_cards_seen_1s": sum(1 for r in card_rows if r["visible_ms"] >= 1000),
        "n_category_toggles": n_cat, "n_preview_toggles": n_prev,
        "n_ask_ai_opens": n_open, "n_ask_ai_messages": n_msg,
        "ask_ai_chars": int(chars), "n_clicks": n_click,
    }
    for s in SCREENS:
        w, a = st.tot.get(s, (0.0, 0.0))
        row[f"{s}_wall_s"], row[f"{s}_active_s"] = secs(w), secs(a)
    return row, card_rows


def read_events(path: Path):
    events, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            try:
                ev = json.loads(line)
            except ValueError:
                ev = None
            if not isinstance(ev, dict):
                skipped += 1
                continue
            ev["_i"] = i
            events.append(ev)
    return events, skipped


def pid_sort_key(pid: str):
    return (not pid.isdigit(), int(pid) if pid.isdigit() else 0, pid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_DIR / "events.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)

    events, skipped = read_events(args.log)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in events:
        groups[(str(e.get("participant_id", "")), str(e.get("session_id", "")))].append(e)

    sess_rows, card_rows = [], []
    for (pid, sid), evs in groups.items():
        # seq is the stable order; file order breaks ties / missing seq
        evs.sort(key=lambda e: (num(e.get("seq")) if isinstance(e.get("seq"), (int, float)) else float("inf"), e["_i"]))
        r, c = summarize_session(pid, sid, evs)
        sess_rows.append(r)
        card_rows.extend(c)
    sess_rows.sort(key=lambda r: (pid_sort_key(r["participant_id"]), r["started_at"], r["session_id"]))
    order = {(r["participant_id"], r["session_id"]): i for i, r in enumerate(sess_rows)}
    card_rows.sort(key=lambda r: order[(r["participant_id"], r["session_id"])])  # stable: keeps card order

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, cols, rows in (("interaction_sessions.csv", SESSION_COLS, sess_rows),
                             ("interaction_cards.csv", CARD_COLS, card_rows)):
        with open(args.out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"Read {len(events)} events from {args.log}; skipped {skipped} blank/malformed line(s)")
    print(f"Wrote {len(sess_rows)} session row(s) -> {args.out_dir / 'interaction_sessions.csv'}")
    print(f"Wrote {len(card_rows)} card row(s) -> {args.out_dir / 'interaction_cards.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
