"""Near-duplicate contamination audit across train/validation/test splits.

Panel critique item A6: the exact-text leakage check in the fine-tuning
notebook catches identical clauses only. Consumer ToS are heavily templated,
so near-duplicate clauses (same boilerplate, one service name changed) can
land in train and test and inflate reported metrics. This script measures
that contamination.

It replicates the notebook's split exactly (same seed, same stratification
key, same two-stage 80/10/10 procedure), then reports, for each validation
and test clause, the maximum TF-IDF character n-gram cosine similarity to
any training clause.

Usage:
    python scripts/near_duplicate_split_audit.py

Outputs:
    generated_files/lawgic_taxonomy/reports/near_duplicate_audit.json
    generated_files/lawgic_taxonomy/reports/near_duplicate_pairs.csv
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDE_CSV = REPO_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv"
OUT_DIR = REPO_ROOT / "generated_files/lawgic_taxonomy/reports"

# Mirror the fine-tuning notebook configuration.
SEED = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.8, 0.1, 0.1
EXCLUDED_TOPIC_IDS = {"unclassified"}
TEXT_COLUMN = "text"

SIMILARITY_THRESHOLDS = (0.80, 0.90, 0.95)


def primary_stratify_label(active_topic_ids: list[str]) -> str:
    for topic_id in active_topic_ids:
        if topic_id not in EXCLUDED_TOPIC_IDS:
            return topic_id
    return "no_active_topic"


def stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def make_stratified_split(df: pd.DataFrame):
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=stratify_or_none(df["stratify_topic"]),
    )
    relative_val_size = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=SEED,
        shuffle=True,
        stratify=stratify_or_none(train_val_df["stratify_topic"]),
    )
    return train_df, val_df, test_df


def audit_split(name, eval_df, train_matrix, train_texts, vectorizer):
    eval_matrix = vectorizer.transform(eval_df[TEXT_COLUMN].astype(str))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_matrix)
    distances, indices = nn.kneighbors(eval_matrix)
    similarities = 1.0 - distances.ravel()

    summary = {
        f"frac_over_{t:.2f}": round(float((similarities >= t).mean()), 4)
        for t in SIMILARITY_THRESHOLDS
    }
    summary["rows"] = int(len(eval_df))
    summary["max_similarity_median"] = round(float(np.median(similarities)), 4)

    pairs = pd.DataFrame(
        {
            "split": name,
            "eval_text": eval_df[TEXT_COLUMN].astype(str).values,
            "nearest_train_text": [train_texts[i] for i in indices.ravel()],
            "cosine_similarity": similarities,
        }
    )
    return summary, pairs[pairs["cosine_similarity"] >= min(SIMILARITY_THRESHOLDS)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(WIDE_CSV)
    active = df["active_topic_ids"].apply(ast.literal_eval)
    df = df.assign(stratify_topic=active.map(primary_stratify_label))

    train_df, val_df, test_df = make_stratified_split(df)
    print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    train_texts = train_df[TEXT_COLUMN].astype(str).tolist()
    train_matrix = vectorizer.fit_transform(train_texts)

    report = {"seed": SEED, "thresholds": list(SIMILARITY_THRESHOLDS)}
    all_pairs = []
    for name, eval_df in (("validation", val_df), ("test", test_df)):
        summary, pairs = audit_split(name, eval_df, train_matrix, train_texts, vectorizer)
        report[name] = summary
        all_pairs.append(pairs)
        print(name, json.dumps(summary, indent=2))

    pd.concat(all_pairs).sort_values("cosine_similarity", ascending=False).to_csv(
        OUT_DIR / "near_duplicate_pairs.csv", index=False
    )
    (OUT_DIR / "near_duplicate_audit.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
