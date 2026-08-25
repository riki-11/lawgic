"""Near-duplicate contamination audit for the v2 (42-topic) corpus.

Same procedure as `scripts/near_duplicate_split_audit.py` -- reuses its
functions rather than duplicating them -- pointed at the v2 wide CSV. The v1
script and its reported numbers stay untouched; the corpus grew by 75 clauses
(see docs/lawgic_taxonomy_revisions.md) so the v1 contamination figures are
void for v2 and must be regenerated, not edited in place.

Usage:
    python scripts/near_duplicate_split_audit_v2.py

Outputs:
    generated_files/lawgic_taxonomy/reports/near_duplicate_audit_v2.json
    generated_files/lawgic_taxonomy/reports/near_duplicate_pairs_v2.csv
"""

from __future__ import annotations

import ast
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from near_duplicate_split_audit import (
    OUT_DIR,
    REPO_ROOT,
    SEED,
    SIMILARITY_THRESHOLDS,
    TEXT_COLUMN,
    audit_split,
    make_stratified_split,
    primary_stratify_label,
)

WIDE_CSV_V2 = REPO_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide_v2.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(WIDE_CSV_V2)
    active = df["active_topic_ids"].apply(ast.literal_eval)
    df = df.assign(stratify_topic=active.map(primary_stratify_label))

    train_df, val_df, test_df = make_stratified_split(df)
    print(f"v2 split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    train_texts = train_df[TEXT_COLUMN].astype(str).tolist()
    train_matrix = vectorizer.fit_transform(train_texts)

    report = {"seed": SEED, "thresholds": list(SIMILARITY_THRESHOLDS), "corpus": "v2"}
    all_pairs = []
    for name, eval_df in (("validation", val_df), ("test", test_df)):
        summary, pairs = audit_split(name, eval_df, train_matrix, train_texts, vectorizer)
        report[name] = summary
        all_pairs.append(pairs)
        print(name, json.dumps(summary, indent=2))

    pd.concat(all_pairs).sort_values("cosine_similarity", ascending=False).to_csv(
        OUT_DIR / "near_duplicate_pairs_v2.csv", index=False
    )
    (OUT_DIR / "near_duplicate_audit_v2.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote v2 reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
