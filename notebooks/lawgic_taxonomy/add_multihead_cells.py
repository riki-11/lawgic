"""Add multihead wide-format generation cells to the taxonomy notebook.

This script appends new cells to lawgic_taxonomy.ipynb that:
1. Build a source-coverage lookup from lawgic_topics.json
2. Generate corrected topic masks with supervised negatives
3. Resolve per-row harm scores using pessimistic consumer protection
4. Export lawgic_multihead_wide.csv

Run from the project root:
    python3 notebooks/lawgic_taxonomy/add_multihead_cells.py
"""

import json
from pathlib import Path


def make_markdown_cell(source: str) -> dict:
    """Create a notebook markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def make_code_cell(source: str) -> dict:
    """Create a notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

MD_INTRO = """\
## Dual-Head: Multihead Wide-Format Generation

This section generates `lawgic_multihead_wide.csv`, a corrected wide-format dataset
designed for dual-head model training. It fixes the degenerate positive-only masking
from the original `build_wide_df()` and adds a resolved per-row harm score.

**Key changes from the original wide format:**

1. **Source-aware topic masking**: For ToS;DR and 100 ToS rows, `topic_mask[c] = 1`
   for *all* topics in that source's taxonomy coverage (from `source_mappings`),
   not just topics where `presence = 1`. For CLAUDETTE rows, Option A masking keeps
   the mask narrow to only the directly mapped topic(s).

2. **Pessimistic harm score resolution**: Instead of nullifying conflict rows,
   apply `min()` across all non-null scores. If any source scores a topic as −1,
   that text is treated as harmful.

3. **Row-level harm classification**: A single `harm_score_class ∈ {0, 1, 2}`
   maps `{−1 → 0, 0 → 1, 1 → 2}` for cross-entropy training.

See `docs/lawgic_dual_head_architecture.md` for the full architectural rationale.\
"""

CODE_SOURCE_COVERAGE = '''\
def build_source_coverage_mask(taxonomy: list[dict], topic_id_to_index: dict[str, int]) -> dict[str, list[float]]:
    """Build a per-source mask vector indicating which topics each source covers.

    For each source (tos_dr, 100_tos, claudette), creates a vector of length
    len(topic_ids) where position c is 1.0 if that source has a source_mapping
    entry for topic c, and 0.0 otherwise.

    Args:
        taxonomy: List of topic dicts from lawgic_topics.json, each containing
            'id' and 'source_mappings'.
        topic_id_to_index: Mapping from topic ID string to its integer index.

    Returns:
        Dict mapping source name to a list of floats (coverage mask vector).
        Keys are 'tos_dr', '100_tos', 'claudette'.
    """
    sources = ["tos_dr", "100_tos", "claudette"]
    coverage: dict[str, list[float]] = {src: [0.0] * len(topic_id_to_index) for src in sources}

    for topic_entry in taxonomy:
        topic_id = topic_entry["id"]
        if topic_id not in topic_id_to_index:
            continue
        idx = topic_id_to_index[topic_id]
        source_mappings = topic_entry.get("source_mappings", {})
        for src in sources:
            if src in source_mappings and source_mappings[src]:
                coverage[src][idx] = 1.0

    for src in sources:
        covered = int(sum(coverage[src]))
        print(f"  {src} taxonomy coverage: {covered} / {len(topic_id_to_index)} topics")

    return coverage


# Load taxonomy entries for source_mappings lookup.
with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
    taxonomy_full = json.load(f)

source_coverage = build_source_coverage_mask(
    taxonomy_full["topics"], topic_id_to_index
)
'''

CODE_BUILD_MULTIHEAD_WIDE = '''\
HARM_SCORE_TO_CLASS = {-1: 0, 0: 1, 1: 2}
MULTIHEAD_WIDE_OUTPUT_PATH = OUTPUT_DIR / "lawgic_multihead_wide.csv"


def resolve_topic_score_pessimistic(score_values: list[int]) -> int:
    """Resolve conflicting topic scores using pessimistic consumer protection.

    If any source scores a topic as -1 (harmful), the resolved score is -1.
    Otherwise, the resolved score is the minimum (most pessimistic) value.

    Args:
        score_values: Non-empty list of integer scores from {-1, 0, 1}.

    Returns:
        Single resolved integer score.
    """
    return min(score_values)


def resolve_row_harm_score(per_topic_scores: list[int | None]) -> int | None:
    """Derive a single row-level harm score from the per-topic score vector.

    Applies the pessimistic policy: take the minimum across all non-null
    per-topic scores. Returns None only when ALL topic scores are null.

    Args:
        per_topic_scores: List of scores (one per topic), where None means
            the topic was not annotated or is outside source coverage.

    Returns:
        Resolved harm score in {-1, 0, 1}, or None if no scores are available.
    """
    non_null = [s for s in per_topic_scores if s is not None]
    if not non_null:
        return None
    return min(non_null)


def compute_source_aware_mask(
    group: pd.DataFrame,
    source_coverage: dict[str, list[float]],
    claudette_topic_rules: dict[str, list[str]],
    topic_id_to_index: dict[str, int],
    num_topics: int,
) -> list[float]:
    """Compute the corrected topic mask for one normalized text group.

    For each source that contributed annotations to this text:
    - ToS;DR / 100 ToS: activate mask for ALL topics in that source's taxonomy
      coverage (from source_mappings in lawgic_topics.json).
    - CLAUDETTE (Option A): activate mask ONLY for the specific topic(s) mapped
      by CLAUDETTE_TOPIC_RULES for the native label in this row.

    The final mask is the element-wise maximum (union) across all sources.

    Args:
        group: DataFrame subset for one normalized_text, from the long table.
        source_coverage: Per-source coverage vectors from build_source_coverage_mask().
        claudette_topic_rules: CLAUDETTE native-code → Lawgic-topic-ID mapping.
        topic_id_to_index: Topic ID → integer index.
        num_topics: Total number of topics (45).

    Returns:
        List of floats (length num_topics), where 1.0 = supervised, 0.0 = unknown.
    """
    mask = [0.0] * num_topics

    contributing_sources = set(group["source_dataset"].astype(str))

    # ToS;DR and 100 ToS: apply full source-level coverage masks.
    for src in ["tos_dr", "100_tos"]:
        if src in contributing_sources:
            for c in range(num_topics):
                if source_coverage[src][c] == 1.0:
                    mask[c] = 1.0

    # CLAUDETTE: Option A — only the directly mapped topic(s) per native label.
    if "claudette" in contributing_sources:
        claudette_rows = group[group["source_dataset"] == "claudette"]
        for native_label in claudette_rows["native_label"].unique():
            mapped_topic_ids = claudette_topic_rules.get(str(native_label), [])
            for tid in mapped_topic_ids:
                if tid in topic_id_to_index:
                    mask[topic_id_to_index[tid]] = 1.0

    return mask


def build_multihead_wide_df(
    long_df: pd.DataFrame,
    source_coverage: dict[str, list[float]],
) -> pd.DataFrame:
    """Convert long records into one dual-head training row per normalized text.

    Unlike the original build_wide_df(), this function:
    1. Uses source-aware masking to generate supervised negatives.
    2. Resolves score conflicts pessimistically instead of nullifying them.
    3. Adds a row-level harm_score, harm_score_class, and harm_mask.

    Args:
        long_df: Combined long-format DataFrame with all source annotations.
        source_coverage: Per-source taxonomy coverage from build_source_coverage_mask().

    Returns:
        DataFrame with one row per normalized text, containing corrected topic
        masks, presence labels, and resolved harm scores.
    """
    wide_records: list[dict[str, Any]] = []

    for normalized_text, group in long_df.groupby("normalized_text", sort=False):
        # --- Topic presence labels (unchanged from original) ---
        labels_presence = [0.0] * len(topic_ids)
        scores: list[int | None] = [None] * len(topic_ids)
        topic_scores: dict[str, int | None] = {}
        active_topic_ids: list[str] = []
        conflict_topic_ids: list[str] = []

        for topic_id, topic_group in group.groupby("lawgic_topic_id", sort=False):
            topic_index = topic_id_to_index[topic_id]
            labels_presence[topic_index] = 1.0
            active_topic_ids.append(topic_id)

            # Pessimistic resolution instead of nullifying conflicts.
            score_values = sorted(set(int(s) for s in topic_group["mapped_score"]))
            resolved = resolve_topic_score_pessimistic(score_values)
            scores[topic_index] = resolved
            topic_scores[topic_id] = resolved

            if len(score_values) > 1:
                conflict_topic_ids.append(topic_id)

        # --- Source-aware topic mask (the key fix) ---
        topic_mask = compute_source_aware_mask(
            group, source_coverage, CLAUDETTE_TOPIC_RULES,
            topic_id_to_index, len(topic_ids),
        )

        # --- Row-level harm score ---
        harm_score = resolve_row_harm_score(scores)
        harm_score_class = HARM_SCORE_TO_CLASS[harm_score] if harm_score is not None else None
        harm_mask = 1.0 if harm_score is not None else 0.0

        # --- Audit trail ---
        native_annotations = group[[
            "source_dataset", "source_id", "lawgic_topic_id",
            "mapped_score", "native_label", "native_tag", "mapping_rule",
        ]].to_dict("records")

        wide_records.append({
            "text": choose_display_text(group["text"]),
            "normalized_text": normalized_text,
            "sources": compact_json(sorted(set(group["source_dataset"].astype(str)))),
            "labels_presence": compact_json(labels_presence),
            "topic_mask": compact_json(topic_mask),
            "scores": compact_json(scores),
            "topic_scores": compact_json(topic_scores),
            "active_topic_ids": compact_json(sorted(active_topic_ids, key=topic_id_to_index.get)),
            "conflict_topic_ids": compact_json(sorted(conflict_topic_ids, key=topic_id_to_index.get)),
            "has_score_conflict": bool(conflict_topic_ids),
            "harm_score": harm_score,
            "harm_score_class": harm_score_class,
            "harm_mask": harm_mask,
            "native_annotations": compact_json(native_annotations),
        })

    return pd.DataFrame.from_records(wide_records)


multihead_wide_df = build_multihead_wide_df(long_df, source_coverage)
multihead_wide_df.head()
'''

CODE_VALIDATE_AND_SAVE = '''\
import numpy as np

# --- Validate the corrected mask creates supervised negatives ---
labels_matrix = np.array([json.loads(lp) for lp in multihead_wide_df["labels_presence"]])
mask_matrix = np.array([json.loads(tm) for tm in multihead_wide_df["topic_mask"]])

supervised_positive = int(((mask_matrix == 1) & (labels_matrix == 1)).sum())
supervised_negative = int(((mask_matrix == 1) & (labels_matrix == 0)).sum())
unsupervised = int((mask_matrix == 0).sum())
positive_ratio = supervised_positive / max(supervised_positive + supervised_negative, 1)

print("=== Multihead Wide Mask Sanity Check ===")
print(f"  Supervised positives  (mask=1 & label=1): {supervised_positive:,}")
print(f"  Supervised negatives  (mask=1 & label=0): {supervised_negative:,}")
print(f"  Unsupervised/ignored  (mask=0):           {unsupervised:,}")
print(f"  Positive ratio among supervised:          {positive_ratio:.4f}")
print()

if supervised_negative == 0:
    raise ValueError(
        "CRITICAL: Still zero supervised negatives! "
        "The source-aware masking did not produce any negatives."
    )

# --- Validate harm score distribution ---
harm_dist = multihead_wide_df["harm_score_class"].value_counts(dropna=False).sort_index()
print("=== Harm Score Class Distribution ===")
print(harm_dist.to_string())
print(f"  Rows without harm label (harm_mask=0): {int((multihead_wide_df['harm_mask'] == 0.0).sum())}")
print()

# --- Save ---
multihead_wide_df.to_csv(MULTIHEAD_WIDE_OUTPUT_PATH, index=False)
print(f"Wrote multihead wide records: {MULTIHEAD_WIDE_OUTPUT_PATH}")
print(f"  Shape: {multihead_wide_df.shape}")
'''

# ---------------------------------------------------------------------------
# Assemble and inject
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    notebook_path = project_root / "notebooks" / "lawgic_taxonomy" / "lawgic_taxonomy.ipynb"

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the "Write Outputs" markdown cell (cell 19) and insert before it.
    insert_index = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "Write Outputs" in src and cell["cell_type"] == "markdown":
            insert_index = i
            break

    if insert_index is None:
        # Fallback: insert before the last two cells.
        insert_index = len(nb["cells"]) - 2

    new_cells = [
        make_markdown_cell(MD_INTRO),
        make_code_cell(CODE_SOURCE_COVERAGE),
        make_code_cell(CODE_BUILD_MULTIHEAD_WIDE),
        make_code_cell(CODE_VALIDATE_AND_SAVE),
    ]

    nb["cells"] = nb["cells"][:insert_index] + new_cells + nb["cells"][insert_index:]

    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Inserted {len(new_cells)} cells at position {insert_index}")
    print(f"Updated: {notebook_path}")


if __name__ == "__main__":
    main()
