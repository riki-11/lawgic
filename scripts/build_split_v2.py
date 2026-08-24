"""Build the seed-42 stratified split for the v2 (42-topic) corpus.

`scripts/lawgic_eval_core.py` stays pointed at the v1 (44-topic) corpus and
checkpoint on purpose (see the NOTE on its `DATA_PATH`/`TAXONOMY_PATH`) --
`persist_split()` there cannot be pointed at v2 data without shape-mismatching
the existing v3 checkpoint's eval harness. `build_split_assignment()` is a
pure function of a dataframe, though, so this script reuses it directly
against the v2 corpus instead of duplicating the stratification logic.

Usage:
    python scripts/build_split_v2.py

Writes generated_files/lawgic_taxonomy/splits/split_seed42_v2.csv. Does not
touch splits/split_seed42.csv (the v1 split).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lawgic_eval_core import (
    EXCLUDED_TOPIC_IDS,
    PROJECT_ROOT,
    build_split_assignment,
)

DATA_PATH_V2 = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide_v2.csv"
SPLIT_PATH_V2 = PROJECT_ROOT / "generated_files/lawgic_taxonomy/splits/split_seed42_v2.csv"
TEXT_COLUMN = "text"


def _parse_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_corpus_v2() -> pd.DataFrame:
    """Same compaction load_corpus() performs in lawgic_eval_core.py, against v2 paths."""
    raw = pd.read_csv(DATA_PATH_V2)
    df = raw.copy()
    for column in ["active_topic_ids", "harm_score_class"]:
        if df[column].dtype == object:
            df[column] = df[column].map(_parse_json_value)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype("string").str.strip()
    df["normalized_text"] = df["normalized_text"].astype("string").str.strip()
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN] != ""].copy()

    df["active_topic_ids_predicted"] = df["active_topic_ids"].map(
        lambda v: [t for t in v if t not in EXCLUDED_TOPIC_IDS]
    )
    df["harm_class"] = df["harm_score_class"].fillna(-1).astype(int)

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def main() -> Path:
    df = load_corpus_v2()
    assignment = build_split_assignment(df)
    counts = assignment.value_counts().to_dict()
    print(f"v2 corpus rows: {len(df)}")
    print(f"Split counts: {counts}")

    SPLIT_PATH_V2.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": df["row_id"].to_numpy(),
            "split": assignment.to_numpy(),
            "normalized_text_sha": pd.util.hash_pandas_object(
                df["normalized_text"], index=False
            ).to_numpy(),
        }
    ).to_csv(SPLIT_PATH_V2, index=False)
    print(f"Wrote {SPLIT_PATH_V2}")
    return SPLIT_PATH_V2


if __name__ == "__main__":
    main()


def demo() -> None:
    """Smallest runnable check: split counts sum to the corpus size, no overlap."""
    df = load_corpus_v2()
    assignment = build_split_assignment(df)  # raises on leakage internally
    counts = assignment.value_counts()
    assert counts.sum() == len(df), (counts.sum(), len(df))
    assert set(counts.index) <= {"train", "validation", "test"}
    print("demo: OK —", counts.to_dict())
