"""Corpus supervision report for the fused Lawgic taxonomy dataset.

Produces the numbers cited in Chapter 4 of the manuscript (panel critique
items B2, B3, B8, C10):

- B2: count of clauses annotated by more than one source (multi-source overlap)
- B3: per-topic supervised positives / observed negatives / source composition
- B8: harm-mask statistics (rows whose harm score resolved vs. not)
- C10: Legal-BERT token-length distribution and truncation fraction at 256

Usage:
    python scripts/corpus_report.py           # v1 (44-topic) corpus
    python scripts/corpus_report.py v2        # v2 (42-topic) corpus

Outputs (suffixed to match the requested corpus version):
    generated_files/lawgic_taxonomy/reports/corpus_report[_v2].json
    generated_files/lawgic_taxonomy/reports/per_topic_supervision[_v2].csv
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Pass "v2" on the command line to report on the 42-topic corpus instead.
ARTIFACT_VERSION = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
WIDE_CSV = REPO_ROOT / f"generated_files/lawgic_taxonomy/lawgic_multihead_wide{ARTIFACT_VERSION}.csv"
FUSION_SUMMARY = REPO_ROOT / f"generated_files/lawgic_taxonomy/lawgic_fusion_summary{ARTIFACT_VERSION}.json"
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

    # Per-topic source composition of positives. Credit a source for a topic
    # only where that source's own annotation maps to it (native_annotations),
    # not merely because the source is present on the row -- `sources` lists
    # every source on the row, which over-credits the 109 multi-source rows.
    topic_index = {t: i for i, t in enumerate(topic_ids)}
    composition_rows: dict[str, dict[str, set[int]]] = {t: {} for t in topic_ids}
    native_annotations = df["native_annotations"].apply(json.loads)
    for row_idx, annotations in enumerate(native_annotations):
        for ann in annotations:
            topic = ann["lawgic_topic_id"]
            idx = topic_index.get(topic)
            if idx is None or mask[row_idx, idx] != 1:
                continue
            composition_rows[topic].setdefault(ann["source_dataset"], set()).add(row_idx)
    composition = {
        t: {src: len(rows) for src, rows in composition_rows[t].items()} for t in topic_ids
    }

    per_topic = pd.DataFrame(
        {
            "topic": topic_ids,
            "supervised_positives": positives,
            "observed_negatives": negatives,
            "positive_sources": [json.dumps(composition[t]) for t in topic_ids],
        }
    ).sort_values("supervised_positives")
    per_topic.to_csv(OUT_DIR / f"per_topic_supervision{ARTIFACT_VERSION}.csv", index=False)

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

    out_path = OUT_DIR / f"corpus_report{ARTIFACT_VERSION}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path} and {OUT_DIR / f'per_topic_supervision{ARTIFACT_VERSION}.csv'}")


if __name__ == "__main__":
    main()
