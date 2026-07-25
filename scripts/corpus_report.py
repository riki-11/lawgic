"""Corpus supervision report for the fused Lawgic taxonomy dataset.

Produces the numbers cited in Chapter 4 of the manuscript (panel critique
items B2, B3, B8, C10):

- B2: count of clauses annotated by more than one source (multi-source overlap)
- B3: per-topic supervised positives / observed negatives / source composition
- B8: harm-mask statistics (rows whose harm score resolved vs. not)
- C10: Legal-BERT token-length distribution and truncation fraction at 256

Usage:
    python scripts/corpus_report.py

Outputs:
    generated_files/lawgic_taxonomy/reports/corpus_report.json
    generated_files/lawgic_taxonomy/reports/per_topic_supervision.csv
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDE_CSV = REPO_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv"
FUSION_SUMMARY = REPO_ROOT / "generated_files/lawgic_taxonomy/lawgic_fusion_summary.json"
TOKENIZER_JSON = REPO_ROOT / "saved_models/lawgic_classifier_legal-bert_v3/tokenizer.json"
OUT_DIR = REPO_ROOT / "generated_files/lawgic_taxonomy/reports"

MAX_LENGTH = 256


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(WIDE_CSV)
    topic_ids = json.loads(FUSION_SUMMARY.read_text())["topic_ids"]

    report: dict = {"total_rows": int(len(df))}

    # ---- B2: multi-source overlap -------------------------------------
    sources = df["sources"].apply(ast.literal_eval)
    n_sources = sources.apply(len)
    multi = sources[n_sources > 1].apply(lambda s: " + ".join(sorted(s)))
    report["multi_source_rows"] = int((n_sources > 1).sum())
    report["multi_source_breakdown"] = multi.value_counts().to_dict()

    # ---- B3: per-topic supervision ------------------------------------
    presence = np.array([ast.literal_eval(x) for x in df["labels_presence"]])
    mask = np.array([ast.literal_eval(x) for x in df["topic_mask"]])
    positives = (presence * mask).sum(axis=0).astype(int)
    negatives = ((1 - presence) * mask).sum(axis=0).astype(int)

    # Per-topic source composition of positives.
    source_lists = sources.tolist()
    composition: dict[str, dict[str, int]] = {t: {} for t in topic_ids}
    pos_rows, pos_topics = np.nonzero(presence * mask)
    for row_idx, topic_idx in zip(pos_rows, pos_topics):
        topic = topic_ids[topic_idx]
        for src in source_lists[row_idx]:
            composition[topic][src] = composition[topic].get(src, 0) + 1

    per_topic = pd.DataFrame(
        {
            "topic": topic_ids,
            "supervised_positives": positives,
            "observed_negatives": negatives,
            "positive_sources": [json.dumps(composition[t]) for t in topic_ids],
        }
    ).sort_values("supervised_positives")
    per_topic.to_csv(OUT_DIR / "per_topic_supervision.csv", index=False)

    substantive = per_topic[per_topic["topic"] != "unclassified"]
    report["topics_with_zero_positives"] = substantive[
        substantive["supervised_positives"] == 0
    ]["topic"].tolist()
    report["topics_under_50_positives"] = substantive[
        substantive["supervised_positives"] < 50
    ]["topic"].tolist()
    report["topics_with_zero_negatives"] = substantive[
        substantive["observed_negatives"] == 0
    ]["topic"].tolist()

    # ---- B8: harm mask -------------------------------------------------
    report["harm_mask_resolved"] = int((df["harm_mask"] == 1).sum())
    report["harm_mask_unresolved"] = int((df["harm_mask"] == 0).sum())

    # ---- C10: token lengths ---------------------------------------------
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        lengths = np.array(
            [len(e.ids) for e in tokenizer.encode_batch(df["text"].astype(str).tolist())]
        )
        report["token_lengths"] = {
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            "over_max_length": int((lengths > MAX_LENGTH).sum()),
            "over_max_length_pct": round(float((lengths > MAX_LENGTH).mean()) * 100, 2),
        }
    except Exception as exc:  # tokenizer unavailable: report and continue
        report["token_lengths"] = f"skipped: {exc}"

    out_path = OUT_DIR / "corpus_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path} and {OUT_DIR / 'per_topic_supervision.csv'}")


if __name__ == "__main__":
    main()
