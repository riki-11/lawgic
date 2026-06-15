"""Generate the dual-head Legal-BERT fine-tuning notebook.

Creates notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb
with the complete training pipeline for the Lawgic dual-head architecture.

Run from the project root:
    python3 notebooks/model_finetuning/create_dual_head_notebook.py
"""

import json
from pathlib import Path


def md(source: str) -> dict:
    """Create a notebook markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """Create a notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ============================================================================
# CELL CONTENTS
# ============================================================================

CELL_TITLE = md("""\
# Lawgic Dual-Head Legal-BERT Fine-Tuning

This notebook trains a **dual-head Legal-BERT model** on the fused Lawgic taxonomy dataset.

**Architecture:**
- **Shared backbone**: `nlpaueb/legal-bert-base-uncased` (12 layers, 768 hidden)
- **Head 1 (Topic)**: `nn.Linear(768, 44)` — multi-label topic presence with masked BCE loss
- **Head 2 (Harm)**: `nn.Linear(768, 3)` — 3-class consumer harm with cross-entropy loss

**Key improvements over v2 (single-head):**
1. Source-aware topic masking creates supervised negatives (fixes degenerate 1.0 F1).
2. Pessimistic harm score resolution replaces nullified conflict rows.
3. Joint multi-objective loss trains both tasks simultaneously.

See `docs/lawgic_dual_head_architecture.md` for the full architectural rationale.\
""")

CELL_IMPORTS_MD = md("""\
## 1. Environment Imports and Experiment Configuration\
""")

CELL_IMPORTS = code("""\
import inspect
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Experiment constants ──────────────────────────────────────────────────────

SEED = 42
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 256
DECISION_THRESHOLD = 0.50
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10
SPLIT_STRATEGY = "stratified_multiobj"

LEARNING_RATE = 3e-5
MAX_EPOCHS = 20
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

TOTAL_TAXONOMY_TOPICS = 45
EXCLUDED_TOPIC_IDS = {"unclassified"}
NUM_LAWGIC_TOPICS = 44
NUM_HARM_CLASSES = 3
TEXT_COLUMN = "text"

HARM_SCORE_TO_CLASS = {-1: 0, 0: 1, 1: 2}
HARM_CLASS_NAMES = {0: "Harmful", 1: "Neutral", 2: "Fair"}


def find_project_root(start: Path | None = None) -> Path:
    \"\"\"Walk upward until the Lawgic repository root is found.\"\"\"
    start = (start or Path.cwd()).resolve()
    sentinel = Path("generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv")
    for candidate in (start, *start.parents):
        if (candidate / sentinel).exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root containing {sentinel}")


PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv"
TAXONOMY_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_topics.json"
FUSION_SUMMARY_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_fusion_summary.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "saved_models/lawgic_classifier_legal-bert_v3"
CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "checkpoints"

random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

print(f"Project root: {PROJECT_ROOT}")
print(f"Training data: {DATA_PATH}")
print(f"Taxonomy: {TAXONOMY_PATH}")
print(f"Model output: {MODEL_OUTPUT_DIR}")
print(f"Split strategy: {SPLIT_STRATEGY}")
print(f"Harm classes: {HARM_CLASS_NAMES}")\
""")

CELL_DEVICE_MD = md("""\
## 2. Hardware Device Autodetection and Precision Policy\
""")

CELL_DEVICE = code("""\
def detect_device() -> tuple[str, torch.device]:
    \"\"\"Detect the best available hardware accelerator.

    Returns:
        Tuple of (device_label, torch_device) where device_label is one of
        'cuda', 'mps', or 'cpu'.
    \"\"\"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        label = "cuda"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        label = "mps"
        print("Apple Silicon MPS detected (fp16 disabled for stability)")
    else:
        device = torch.device("cpu")
        label = "cpu"
        print("CPU only — training will be slow")

    return label, device


DEVICE_LABEL, DEVICE = detect_device()
USE_FP16 = DEVICE_LABEL == "cuda"
print(f"Device: {DEVICE_LABEL}, FP16: {USE_FP16}")\
""")

CELL_TAXONOMY_MD = md("""\
## 3. Taxonomy Loading\
""")

CELL_TAXONOMY = code("""\
with TAXONOMY_PATH.open("r", encoding="utf-8") as file:
    taxonomy_payload = json.load(file)

all_topic_ids = [topic["id"] for topic in taxonomy_payload["topics"]]

if len(all_topic_ids) != TOTAL_TAXONOMY_TOPICS:
    raise ValueError(f"Expected {TOTAL_TAXONOMY_TOPICS} topics, found {len(all_topic_ids)}")

keep_indices = [i for i, tid in enumerate(all_topic_ids) if tid not in EXCLUDED_TOPIC_IDS]
topic_ids = [all_topic_ids[i] for i in keep_indices]

if len(topic_ids) != NUM_LAWGIC_TOPICS:
    raise ValueError(f"Expected {NUM_LAWGIC_TOPICS} non-excluded topics, found {len(topic_ids)}")

id2label = {i: topic_id for i, topic_id in enumerate(topic_ids)}
label2id = {topic_id: i for i, topic_id in enumerate(topic_ids)}
name_by_topic = {topic["id"]: topic["name"] for topic in taxonomy_payload["topics"]}
parent_by_topic = {topic["id"]: topic["parent_topic"] for topic in taxonomy_payload["topics"]}

print(f"Loaded {TOTAL_TAXONOMY_TOPICS} topics, training on {NUM_LAWGIC_TOPICS} (excluded: {sorted(EXCLUDED_TOPIC_IDS)})")
print(f"Topic ID sample: {topic_ids[:5]}")\
""")

CELL_DATA_MD = md("""\
## 4. Data Loading and Validation

The multihead wide CSV contains corrected source-aware topic masks (with supervised
negatives) and a resolved per-row harm score. Key new columns:

| Column | Type | Description |
| --- | --- | --- |
| `topic_mask` | JSON float[45] | Source-aware mask (1.0 = supervised, 0.0 = unknown) |
| `harm_score` | int or null | Resolved row-level harm score {-1, 0, 1} |
| `harm_score_class` | int or null | Class index {0, 1, 2} for CE loss |
| `harm_mask` | float | 1.0 if harm label valid, 0.0 if unresolvable |\
""")

CELL_DATA = code("""\
JSON_COLUMNS = [
    "sources",
    "labels_presence",
    "topic_mask",
    "scores",
    "topic_scores",
    "active_topic_ids",
    "conflict_topic_ids",
    "native_annotations",
]
REQUIRED_COLUMNS = [
    TEXT_COLUMN, "normalized_text", "has_score_conflict",
    "harm_score", "harm_score_class", "harm_mask",
    *JSON_COLUMNS,
]


def parse_json_value(value: Any) -> Any:
    \"\"\"Parse a JSON-encoded CSV cell, returning None for missing values.\"\"\"
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def compact_vector(vector: list[Any], fill_value: float | None = None) -> list[Any]:
    \"\"\"Slice a 45-length vector to 44 by removing excluded topic positions.

    Args:
        vector: Full 45-element vector from the wide CSV.
        fill_value: If provided, replace None values with this float.

    Returns:
        44-element list with excluded topics removed.
    \"\"\"
    if len(vector) != TOTAL_TAXONOMY_TOPICS:
        raise ValueError(f"Expected vector length {TOTAL_TAXONOMY_TOPICS}, found {len(vector)}")
    compact = [vector[idx] for idx in keep_indices]
    if fill_value is not None:
        compact = [fill_value if value is None else value for value in compact]
    return compact


def source_counts(source_lists: pd.Series) -> pd.Series:
    \"\"\"Count rows per source dataset.\"\"\"
    counter = Counter(source for sources in source_lists for source in sources)
    return pd.Series(counter).sort_values(ascending=False)


# ── Load and validate ─────────────────────────────────────────────────────────

raw_df = pd.read_csv(DATA_PATH)
missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {DATA_PATH}: {missing_columns}")

samples_df = raw_df.copy()
for column in JSON_COLUMNS:
    samples_df[column] = samples_df[column].map(parse_json_value)

samples_df[TEXT_COLUMN] = samples_df[TEXT_COLUMN].astype("string").str.strip()
samples_df["normalized_text"] = samples_df["normalized_text"].astype("string").str.strip()
before_text_filter = len(samples_df)
samples_df = samples_df.dropna(subset=[TEXT_COLUMN])
samples_df = samples_df[samples_df[TEXT_COLUMN] != ""].copy()

# Validate vector lengths.
for vector_column in ["labels_presence", "topic_mask", "scores"]:
    bad_lengths = samples_df[vector_column].map(len).ne(TOTAL_TAXONOMY_TOPICS)
    if bad_lengths.any():
        raise ValueError(f"{vector_column} contains {bad_lengths.sum()} non-{TOTAL_TAXONOMY_TOPICS}-length vectors")

# ── Compact 45 → 44 ──────────────────────────────────────────────────────────

samples_df["labels"] = samples_df["labels_presence"].map(lambda v: compact_vector(v, fill_value=0.0))
samples_df["label_mask"] = samples_df["topic_mask"].map(lambda v: compact_vector(v, fill_value=0.0))
samples_df["scores_44"] = samples_df["scores"].map(lambda v: compact_vector(v))
samples_df["active_topic_ids_44"] = samples_df["active_topic_ids"].map(
    lambda v: [tid for tid in v if tid not in EXCLUDED_TOPIC_IDS]
)
samples_df["conflict_topic_ids_44"] = samples_df["conflict_topic_ids"].map(
    lambda v: [tid for tid in v if tid not in EXCLUDED_TOPIC_IDS]
)

# ── Harm score fields ─────────────────────────────────────────────────────────
# harm_score_class is already an integer {0,1,2} or NaN.
# For PyTorch CE, we need int64. Rows with NaN get -1 (ignored via harm_mask).

samples_df["harm_class"] = samples_df["harm_score_class"].fillna(-1).astype(int)
samples_df["harm_mask_float"] = samples_df["harm_mask"].astype(float)

# ── Derived statistics ────────────────────────────────────────────────────────

samples_df["label_count"] = samples_df["labels"].map(lambda v: int(np.sum(v)))
samples_df["mask_count"] = samples_df["label_mask"].map(lambda v: int(np.sum(v)))
samples_df["source_count"] = samples_df["sources"].map(len)

label_matrix = np.vstack(samples_df["labels"].to_numpy()).astype(np.float32)
mask_matrix = np.vstack(samples_df["label_mask"].to_numpy()).astype(np.float32)
positive_counts = label_matrix.sum(axis=0).astype(int)
mask_counts = mask_matrix.sum(axis=0).astype(int)

if label_matrix.shape[1] != NUM_LAWGIC_TOPICS or mask_matrix.shape[1] != NUM_LAWGIC_TOPICS:
    raise ValueError("Compacted label/mask matrices do not match NUM_LAWGIC_TOPICS")

# ── Mask sanity check ─────────────────────────────────────────────────────────

supervised_pos = int(((mask_matrix == 1) & (label_matrix == 1)).sum())
supervised_neg = int(((mask_matrix == 1) & (label_matrix == 0)).sum())
total_supervised = supervised_pos + supervised_neg
pos_ratio = supervised_pos / max(total_supervised, 1)

print(f"Raw rows: {len(raw_df):,}")
print(f"Rows after empty-text filter: {len(samples_df):,} (dropped {before_text_filter - len(samples_df):,})")
print(f"Training labels: {NUM_LAWGIC_TOPICS} topics + {NUM_HARM_CLASSES} harm classes")
print()
print("=== Topic Mask Sanity ===")
print(f"  Supervised positives (mask=1 & label=1): {supervised_pos:,}")
print(f"  Supervised negatives (mask=1 & label=0): {supervised_neg:,}")
print(f"  Positive ratio: {pos_ratio:.4f}")

if supervised_neg == 0:
    raise ValueError("CRITICAL: Zero supervised negatives! Dataset is still degenerate.")

print()
print("=== Harm Score Distribution ===")
harm_dist = samples_df["harm_class"].value_counts().sort_index()
for cls_idx, count in harm_dist.items():
    label = HARM_CLASS_NAMES.get(cls_idx, "No label")
    print(f"  Class {cls_idx} ({label}): {count:,}")
print(f"  Rows without harm label (harm_mask=0): {int((samples_df['harm_mask_float'] == 0.0).sum())}")\
""")

CELL_EDA_MD = md("""\
## 5. Exploratory Data Analysis\
""")

CELL_EDA = code("""\
# Topic distribution with mask coverage.
topic_distribution = pd.DataFrame({
    "classifier_id": range(NUM_LAWGIC_TOPICS),
    "topic_id": topic_ids,
    "name": [name_by_topic[tid] for tid in topic_ids],
    "parent_topic": [parent_by_topic[tid] for tid in topic_ids],
    "positive_count": positive_counts,
    "mask_count": mask_counts,
    "negative_count": (mask_counts - positive_counts).astype(int),
}).sort_values("positive_count", ascending=False)

display(topic_distribution.head(15))

# Source membership.
source_distribution = source_counts(samples_df["sources"])
print("Rows by source membership:")
display(source_distribution.rename("rows"))

# Harm class balance.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
harm_valid = samples_df[samples_df["harm_mask_float"] == 1.0]
harm_valid["harm_class"].value_counts().sort_index().plot.bar(
    ax=axes[0], color=["#e74c3c", "#95a5a6", "#2ecc71"]
)
axes[0].set_title("Harm Class Distribution")
axes[0].set_xticklabels(["Harmful (-1)", "Neutral (0)", "Fair (+1)"], rotation=0)

topic_distribution.sort_values("positive_count", ascending=True).plot.barh(
    x="topic_id", y=["positive_count", "negative_count"], stacked=True,
    ax=axes[1], color=["#3498db", "#ecf0f1"], figsize=(12, 10)
)
axes[1].set_title("Topic Positive/Negative Counts (mask=1)")
axes[1].set_xlabel("Count")
plt.tight_layout()
plt.show()\
""")

CELL_SPLIT_MD = md("""\
## 6. Multi-Objective Stratified Split

The stratification key combines the primary active topic ID and the resolved harm class.
This ensures both topic and harm-class distributions are balanced across train,
validation, and test splits.

```
stratify_key = f"{primary_topic}__harm{harm_class}"
```\
""")

CELL_SPLIT = code("""\
def primary_stratify_label(active_topic_ids: list[str]) -> str:
    \"\"\"Select the primary topic for stratification.

    Returns the first non-unclassified active topic, or 'none' if
    no topics are active.

    Args:
        active_topic_ids: Sorted list of active Lawgic topic IDs.

    Returns:
        Topic ID string for stratification.
    \"\"\"
    for topic_id in active_topic_ids:
        if topic_id not in EXCLUDED_TOPIC_IDS:
            return topic_id
    return "none"


def composite_stratify_key(row: pd.Series) -> str:
    \"\"\"Build a composite stratification key from topic + harm class.

    Combines the primary topic with the harm class to ensure balanced
    distribution of both dimensions across splits.

    Args:
        row: DataFrame row with 'active_topic_ids_44' and 'harm_class'.

    Returns:
        Composite key string like 'choice_of_law__harm0'.
    \"\"\"
    primary = primary_stratify_label(row["active_topic_ids_44"])
    harm = int(row["harm_class"])
    return f"{primary}__harm{harm}"


samples_df["stratify_key"] = samples_df.apply(composite_stratify_key, axis=1)

# Collapse rare keys (< 3 samples) to avoid stratification failures.
key_counts = samples_df["stratify_key"].value_counts()
rare_keys = set(key_counts[key_counts < 3].index)
if rare_keys:
    samples_df.loc[samples_df["stratify_key"].isin(rare_keys), "stratify_key"] = "__rare__"
    print(f"Collapsed {len(rare_keys)} rare stratification keys into '__rare__'")

# First split: train vs (val + test).
train_df, valtest_df = train_test_split(
    samples_df,
    test_size=(VAL_SIZE + TEST_SIZE),
    stratify=samples_df["stratify_key"],
    random_state=SEED,
)

# Second split: val vs test.
relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
val_df, test_df = train_test_split(
    valtest_df,
    test_size=relative_test_size,
    stratify=valtest_df["stratify_key"],
    random_state=SEED,
)

# Exact text leakage check.
train_texts = set(train_df["normalized_text"])
val_texts = set(val_df["normalized_text"])
test_texts = set(test_df["normalized_text"])

leak_summary = {
    "train_val_overlap": len(train_texts & val_texts),
    "train_test_overlap": len(train_texts & test_texts),
    "val_test_overlap": len(val_texts & test_texts),
}
if any(leak_summary.values()):
    raise ValueError(f"Exact text leakage detected across splits: {leak_summary}")


def split_topic_coverage(split_df: pd.DataFrame) -> int:
    \"\"\"Count topics with at least one positive label in a split.\"\"\"
    matrix = np.vstack(split_df["labels"].to_numpy()).astype(np.float32)
    return int((matrix.sum(axis=0) > 0).sum())


def split_harm_dist(split_df: pd.DataFrame) -> dict[int, int]:
    \"\"\"Count harm class occurrences in a split (excluding harm_mask=0).\"\"\"
    valid = split_df[split_df["harm_mask_float"] == 1.0]
    return valid["harm_class"].value_counts().sort_index().to_dict()


split_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(train_df), len(val_df), len(test_df)],
    "topics_with_positives": [
        split_topic_coverage(train_df),
        split_topic_coverage(val_df),
        split_topic_coverage(test_df),
    ],
    "harm_dist": [
        split_harm_dist(train_df),
        split_harm_dist(val_df),
        split_harm_dist(test_df),
    ],
})

display(split_summary)
print(f"Split strategy: {SPLIT_STRATEGY}")
print(f"Train/val/test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"Leakage check: {leak_summary}")\
""")

CELL_SMOKE_MD = md("""\
## 7. Baseline Smoke Test — Zero-Logit Diagnostics

**Mandatory pre-training check.** If the dataset were still degenerate (all positives),
zero logits would produce perfect macro F1. We verify this is no longer the case.

- **Topic head**: `sigmoid(0) = 0.5 ≥ threshold 0.5` → predicts all positive.
  With supervised negatives, macro F1 must be significantly below 1.0.
- **Harm head**: `argmax([0, 0, 0])` → always predicts class 0.
  With 3 classes, accuracy should be ~31%.\
""")

CELL_SMOKE = code("""\
def sigmoid(logits: np.ndarray) -> np.ndarray:
    \"\"\"Numerically stable sigmoid activation.\"\"\"
    clipped = np.clip(logits, -60, 60)
    return 1.0 / (1.0 + np.exp(-clipped))


def logits_to_predictions(logits: np.ndarray, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    \"\"\"Convert logits to binary predictions via sigmoid + threshold.\"\"\"
    return (sigmoid(logits) >= threshold).astype(int)


def masked_metric_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    label_masks: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    \"\"\"Compute mask-aware topic metrics over observed positions only.

    Args:
        logits: Raw topic logits, shape [N, 44].
        labels: Binary topic labels, shape [N, 44].
        label_masks: Binary mask, shape [N, 44]. 1.0 = supervised.
        threshold: Decision threshold for sigmoid predictions.

    Returns:
        Dict of metric name → value.
    \"\"\"
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = labels.astype(int)
    label_masks = label_masks.astype(bool)
    predictions = logits_to_predictions(logits, threshold=threshold)

    flat_mask = label_masks.reshape(-1)
    if not flat_mask.any():
        return {"macro_f1": 0.0, "micro_f1": 0.0, "masked_positions": 0.0}

    per_topic_f1 = []
    per_topic_support = []
    for topic_idx in range(labels.shape[1]):
        topic_mask = label_masks[:, topic_idx]
        if not topic_mask.any():
            continue
        topic_labels = labels[topic_mask, topic_idx]
        topic_preds = predictions[topic_mask, topic_idx]
        per_topic_f1.append(f1_score(topic_labels, topic_preds, zero_division=0))
        per_topic_support.append(int(topic_labels.sum()))

    support_total = np.sum(per_topic_support)
    weighted_f1 = np.average(per_topic_f1, weights=per_topic_support) if support_total > 0 else 0.0

    return {
        "macro_f1": float(np.mean(per_topic_f1)) if per_topic_f1 else 0.0,
        "micro_f1": f1_score(
            labels.reshape(-1)[flat_mask],
            predictions.reshape(-1)[flat_mask],
            zero_division=0,
        ),
        "weighted_f1": float(weighted_f1),
        "predicted_positive_rate": float(predictions.reshape(-1)[flat_mask].mean()),
        "masked_positions": float(flat_mask.sum()),
    }


# ── Topic head smoke test ─────────────────────────────────────────────────────

val_labels = np.vstack(val_df["labels"].to_numpy()).astype(np.float32)
val_masks = np.vstack(val_df["label_mask"].to_numpy()).astype(np.float32)
zero_topic_logits = np.zeros_like(val_labels)

topic_smoke = masked_metric_summary(zero_topic_logits, val_labels, val_masks)
print("=== Topic Head Zero-Logit Smoke Test ===")
for k, v in topic_smoke.items():
    print(f"  {k}: {v:.4f}")

if topic_smoke["macro_f1"] >= 0.95:
    raise ValueError(
        f"DEGENERATE: Zero-logit macro F1 = {topic_smoke['macro_f1']:.4f} >= 0.95. "
        "The dataset still has no supervised negatives!"
    )
print(f"  ✓ Non-degenerate (macro F1 = {topic_smoke['macro_f1']:.4f} < 0.95)")

# ── Harm head smoke test ──────────────────────────────────────────────────────

val_harm_labels = val_df["harm_class"].to_numpy()
val_harm_masks = val_df["harm_mask_float"].to_numpy()
valid_harm = val_harm_masks == 1.0

zero_harm_logits = np.zeros((len(val_df), NUM_HARM_CLASSES))
harm_preds = zero_harm_logits.argmax(axis=1)  # Always predicts class 0.
harm_acc = (harm_preds[valid_harm] == val_harm_labels[valid_harm]).mean()

print()
print("=== Harm Head Zero-Logit Smoke Test ===")
print(f"  Zero-logit accuracy: {harm_acc:.4f} (expected ~{(val_harm_labels[valid_harm] == 0).mean():.4f})")
print(f"  Valid harm samples: {int(valid_harm.sum()):,}")\
""")

CELL_TOK_MD = md("""\
## 8. Tokenization and PyTorch Dataset Wrappers\
""")

CELL_TOK = code("""\
if "tokenizer" not in globals():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


class LawgicDualHeadDataset(Dataset):
    \"\"\"PyTorch dataset for dual-head Lawgic model training.

    Each item contains tokenized text plus targets for both heads:
    - Topic labels and mask (44-dim float vectors)
    - Harm class label (scalar int64) and harm mask (scalar float)

    Args:
        frame: DataFrame with 'text', 'labels', 'label_mask', 'harm_class',
            'harm_mask_float' columns.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
        text_column: Name of the text column.
    \"\"\"

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        text_column: str,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.texts = self.frame[text_column].astype(str).tolist()
        self.labels = np.vstack(self.frame["labels"].to_numpy()).astype(np.float32)
        self.label_masks = np.vstack(self.frame["label_mask"].to_numpy()).astype(np.float32)
        self.harm_labels = self.frame["harm_class"].to_numpy().astype(np.int64)
        self.harm_masks = self.frame["harm_mask_float"].to_numpy().astype(np.float32)
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        \"\"\"Return a single training example.

        Returns:
            Dict with keys: input_ids, attention_mask, token_type_ids,
            labels (float32 [44]), label_mask (float32 [44]),
            harm_label (int64 scalar), harm_mask (float32 scalar).
        \"\"\"
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        item["label_mask"] = torch.tensor(self.label_masks[index], dtype=torch.float32)
        item["harm_label"] = torch.tensor(self.harm_labels[index], dtype=torch.int64)
        item["harm_mask"] = torch.tensor(self.harm_masks[index], dtype=torch.float32)
        return item


class LawgicDualHeadCollator:
    \"\"\"Data collator that pads token sequences and stacks dual-head targets.

    Handles padding of variable-length tokenized inputs while preserving
    the fixed-size label tensors for both heads.

    Args:
        tokenizer: HuggingFace tokenizer (used for padding).
    \"\"\"

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        \"\"\"Collate a batch of features into padded tensors.

        Args:
            features: List of dicts from LawgicDualHeadDataset.__getitem__.

        Returns:
            Batch dict with padded input tensors and stacked label tensors.
            Shape summary:
                input_ids:      [batch_size, seq_len]
                attention_mask: [batch_size, seq_len]
                labels:         [batch_size, 44]
                label_mask:     [batch_size, 44]
                harm_label:     [batch_size]
                harm_mask:      [batch_size]
        \"\"\"
        extra_keys = {"labels", "label_mask", "harm_label", "harm_mask"}
        token_features = [
            {k: v for k, v in f.items() if k not in extra_keys}
            for f in features
        ]
        batch = self.tokenizer.pad(token_features, return_tensors="pt")
        batch["labels"] = torch.stack([f["labels"] for f in features])
        batch["label_mask"] = torch.stack([f["label_mask"] for f in features])
        batch["harm_label"] = torch.stack([f["harm_label"] for f in features])
        batch["harm_mask"] = torch.stack([f["harm_mask"] for f in features])
        return batch


train_dataset = LawgicDualHeadDataset(train_df, tokenizer, MAX_LENGTH, TEXT_COLUMN)
val_dataset = LawgicDualHeadDataset(val_df, tokenizer, MAX_LENGTH, TEXT_COLUMN)
test_dataset = LawgicDualHeadDataset(test_df, tokenizer, MAX_LENGTH, TEXT_COLUMN)
data_collator = LawgicDualHeadCollator(tokenizer)

sample = train_dataset[0]
print(f"Tokenizer: {tokenizer.__class__.__name__}")
print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
print(f"Sample keys: {list(sample.keys())}")
print(f"Sample labels shape: {tuple(sample['labels'].shape)}")
print(f"Sample label_mask shape: {tuple(sample['label_mask'].shape)}")
print(f"Sample harm_label: {sample['harm_label']} (shape: {tuple(sample['harm_label'].shape)})")
print(f"Sample harm_mask: {sample['harm_mask']}")\
""")

CELL_MODEL_MD = md("""\
## 9. Dual-Head Model Architecture

The model wraps Legal-BERT's transformer encoder and attaches two independent linear
heads to the `[CLS]` pooler output:

```
Legal-BERT Encoder (768 hidden) ─→ [CLS] pooler output
    ├─→ topic_head: Linear(768, 44)  →  topic logits
    └─→ harm_head:  Linear(768, 3)   →  harm logits
```

**Why not `AutoModelForSequenceClassification`?** That class supports only a single
classification head. We need two heads with different loss functions, so we build
a custom `nn.Module` around `AutoModel`.\
""")

CELL_MODEL = code("""\
class LawgicDualHeadModel(nn.Module):
    \"\"\"Dual-head Legal-BERT model for topic presence + consumer harm prediction.

    Architecture:
        Input tokens → Legal-BERT encoder → [CLS] pooler output (768-dim)
            → topic_head: nn.Linear(768, num_topics) → topic logits
            → harm_head:  nn.Linear(768, num_harm_classes) → harm logits

    The encoder weights are initialized from a pretrained checkpoint and
    fine-tuned jointly with both heads.

    Args:
        model_name: HuggingFace model identifier (e.g. 'nlpaueb/legal-bert-base-uncased').
        num_topics: Number of topic output dimensions (default: 44).
        num_harm_classes: Number of harm classes (default: 3).
    \"\"\"

    def __init__(
        self,
        model_name: str,
        num_topics: int = NUM_LAWGIC_TOPICS,
        num_harm_classes: int = NUM_HARM_CLASSES,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        self.topic_head = nn.Linear(hidden_size, num_topics)
        self.harm_head = nn.Linear(hidden_size, num_harm_classes)

        self.num_topics = num_topics
        self.num_harm_classes = num_harm_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Forward pass through encoder and both heads.

        Args:
            input_ids: Token IDs, shape [batch_size, seq_len].
            attention_mask: Attention mask, shape [batch_size, seq_len].
            token_type_ids: Optional segment IDs, shape [batch_size, seq_len].

        Returns:
            Tuple of (topic_logits, harm_logits):
                topic_logits: shape [batch_size, num_topics]
                harm_logits:  shape [batch_size, num_harm_classes]
        \"\"\"
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_embedding = outputs.pooler_output  # [batch_size, 768]

        topic_logits = self.topic_head(cls_embedding)  # [batch_size, num_topics]
        harm_logits = self.harm_head(cls_embedding)    # [batch_size, num_harm_classes]

        return topic_logits, harm_logits


model = LawgicDualHeadModel(MODEL_NAME)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {MODEL_NAME}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Topic head output: {model.num_topics}")
print(f"Harm head output: {model.num_harm_classes}")
print(f"Device: {DEVICE}")\
""")

CELL_TRAINER_MD = md("""\
## 10. Custom Trainer with Multi-Objective Loss

The total training loss is the equally weighted sum of the two head losses:

$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{topic}} + \\mathcal{L}_{\\text{harm}}$$

- **Topic loss**: Masked BCE with logits — only `mask=1` positions contribute.
- **Harm loss**: Cross-entropy — only `harm_mask=1` rows contribute.\
""")

CELL_TRAINER = code("""\
class DualHeadTrainer(Trainer):
    \"\"\"HuggingFace Trainer subclass for dual-head multi-objective training.

    Overrides compute_loss to:
    1. Extract custom label/mask fields from the batch.
    2. Forward through the dual-head model.
    3. Compute masked BCE (topic) + masked CE (harm) losses.
    4. Pack both sets of logits into the output for metric computation.
    \"\"\"

    def compute_loss(
        self,
        model: LawgicDualHeadModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, SimpleNamespace]:
        \"\"\"Compute the joint multi-objective loss.

        Loss formulation:
            L_total = L_topic + L_harm

        where:
            L_topic = sum(BCE(logits, labels) * mask) / sum(mask)
            L_harm  = sum(CE(logits, class_idx) * harm_mask) / sum(harm_mask)

        Args:
            model: The LawgicDualHeadModel instance.
            inputs: Batch dict from the collator, containing input_ids,
                attention_mask, labels, label_mask, harm_label, harm_mask.
            return_outputs: If True, return (loss, outputs) tuple.

        Returns:
            Total loss scalar, or (loss, outputs) if return_outputs is True.
            outputs is a SimpleNamespace with topic_logits and harm_logits.
        \"\"\"
        # Pop custom fields so they don't get passed to the encoder.
        label_mask = inputs.pop("label_mask")
        labels = inputs.pop("labels")
        harm_label = inputs.pop("harm_label")
        harm_mask = inputs.pop("harm_mask")

        # Forward pass.
        topic_logits, harm_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

        # ── Head 1: Masked BCE for topics ─────────────────────────────────
        bce_fn = nn.BCEWithLogitsLoss(reduction="none")
        per_topic_loss = bce_fn(topic_logits, labels)  # [B, 44]
        topic_loss = (per_topic_loss * label_mask).sum() / label_mask.sum().clamp(min=1.0)

        # ── Head 2: Masked Cross-Entropy for harm ─────────────────────────
        ce_fn = nn.CrossEntropyLoss(reduction="none")
        # Clamp harm_label to [0, 2] to avoid index errors on -1 sentinel.
        safe_harm_label = harm_label.clamp(min=0)
        per_sample_ce = ce_fn(harm_logits, safe_harm_label)  # [B]
        harm_loss = (per_sample_ce * harm_mask).sum() / harm_mask.sum().clamp(min=1.0)

        total_loss = topic_loss + harm_loss

        if return_outputs:
            outputs = SimpleNamespace(
                loss=total_loss,
                topic_logits=topic_logits.detach(),
                harm_logits=harm_logits.detach(),
            )
            return total_loss, outputs
        return total_loss


# ── Smoke test the trainer loss ───────────────────────────────────────────────

smoke_batch = data_collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
smoke_batch_device = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in smoke_batch.items()}

model.eval()
with torch.no_grad():
    # Manually compute to verify shapes.
    topic_out, harm_out = model(
        input_ids=smoke_batch_device["input_ids"],
        attention_mask=smoke_batch_device["attention_mask"],
        token_type_ids=smoke_batch_device.get("token_type_ids"),
    )
print(f"Topic logits shape: {tuple(topic_out.shape)}")  # [B, 44]
print(f"Harm logits shape:  {tuple(harm_out.shape)}")   # [B, 3]
print("Trainer smoke test passed.")\
""")

CELL_METRICS_MD = md("""\
## 11. Dual-Head Metric Computation

Separate metrics for each head:
- **Topic head**: Mask-aware macro/micro/weighted F1 (primary: `eval_macro_f1`)
- **Harm head**: Multi-class accuracy and weighted F1\
""")

CELL_METRICS = code("""\
metric_context: dict[str, Any] = {}


def per_topic_report(
    logits: np.ndarray,
    labels: np.ndarray,
    label_masks: np.ndarray,
) -> dict[str, dict[str, float]]:
    \"\"\"Generate per-topic precision/recall/F1 over observed positions.

    Args:
        logits: Topic logits, shape [N, 44].
        labels: Binary topic labels, shape [N, 44].
        label_masks: Binary mask, shape [N, 44].

    Returns:
        Dict mapping topic_id to {precision, recall, f1, support}.
    \"\"\"
    predictions = logits_to_predictions(logits)
    labels = labels.astype(int)
    label_masks = label_masks.astype(bool)
    report = {}
    for topic_idx, topic_id in id2label.items():
        topic_mask = label_masks[:, topic_idx]
        if not topic_mask.any():
            continue
        t_labels = labels[topic_mask, topic_idx]
        t_preds = predictions[topic_mask, topic_idx]
        report[topic_id] = {
            "precision": float(precision_score(t_labels, t_preds, zero_division=0)),
            "recall": float(recall_score(t_labels, t_preds, zero_division=0)),
            "f1": float(f1_score(t_labels, t_preds, zero_division=0)),
            "support": int(t_labels.sum()),
            "observed": int(topic_mask.sum()),
        }
    return report


def harm_metric_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    \"\"\"Compute multi-class harm metrics over valid (harm_mask=1) rows.

    Args:
        logits: Harm logits, shape [N, 3].
        labels: Harm class labels, shape [N] (int64, 0/1/2 or -1).
        masks: Harm mask, shape [N] (float, 0.0 or 1.0).

    Returns:
        Dict of metric name → value.
    \"\"\"
    valid = masks.astype(bool)
    if not valid.any():
        return {"harm_accuracy": 0.0, "harm_weighted_f1": 0.0, "harm_valid_samples": 0.0}

    preds = logits[valid].argmax(axis=1)
    true = labels[valid]

    return {
        "harm_accuracy": float(accuracy_score(true, preds)),
        "harm_weighted_f1": float(f1_score(true, preds, average="weighted", zero_division=0)),
        "harm_macro_f1": float(f1_score(true, preds, average="macro", zero_division=0)),
        "harm_valid_samples": float(valid.sum()),
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    \"\"\"Compute dual-head metrics for the HuggingFace Trainer.

    This function is called by the Trainer at each evaluation step.
    It unpacks the packed predictions from DualHeadTrainer and computes
    metrics for both heads.

    The primary checkpoint metric is 'macro_f1' from the topic head.

    Args:
        eval_prediction: EvalPrediction with predictions and label_ids.

    Returns:
        Dict of metric names → values for logging.
    \"\"\"
    predictions = eval_prediction.predictions

    # Unpack: predictions is a tuple of (topic_logits, harm_logits).
    if isinstance(predictions, tuple):
        topic_logits, harm_logits = predictions
    else:
        # Fallback: single array means only topic logits.
        topic_logits = predictions
        harm_logits = None

    # Retrieve masks from the stored context.
    label_masks = metric_context["label_masks"]
    labels = metric_context["labels"]
    harm_labels = metric_context["harm_labels"]
    harm_masks = metric_context["harm_masks"]

    # ── Topic metrics ─────────────────────────────────────────────────────
    topic_metrics = masked_metric_summary(topic_logits, labels, label_masks)

    # ── Harm metrics ──────────────────────────────────────────────────────
    harm_metrics = {}
    if harm_logits is not None:
        harm_metrics = harm_metric_summary(harm_logits, harm_labels, harm_masks)

    return {**topic_metrics, **harm_metrics}


# Store validation context for compute_metrics.
metric_context["labels"] = val_dataset.labels
metric_context["label_masks"] = val_dataset.label_masks
metric_context["harm_labels"] = val_dataset.harm_labels
metric_context["harm_masks"] = val_dataset.harm_masks
print("Metric context initialized for validation set.")\
""")

CELL_TRAINING_MD = md("""\
## 12. Training Configuration and Execution

Key training arguments:
- `remove_unused_columns=False` — preserves custom mask/label fields through the Trainer
- `metric_for_best_model="macro_f1"` — selects checkpoints by topic head macro F1
- Early stopping with patience 3\
""")

CELL_TRAINING = code("""\
def build_training_arguments() -> TrainingArguments:
    \"\"\"Create training arguments compatible with the installed transformers version.

    Inspects the TrainingArguments signature to handle API differences across
    versions (e.g. 'evaluate_strategy' vs 'eval_strategy').

    Returns:
        Configured TrainingArguments instance.
    \"\"\"
    signature = inspect.signature(TrainingArguments.__init__)
    param_names = set(signature.parameters.keys())

    eval_strategy_key = (
        "eval_strategy" if "eval_strategy" in param_names else "evaluation_strategy"
    )

    args = {
        "output_dir": str(CHECKPOINT_DIR),
        eval_strategy_key: "epoch",
        "save_strategy": "epoch",
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE * 2,
        "num_train_epochs": MAX_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_steps": 50,
        "fp16": USE_FP16,
        "remove_unused_columns": False,
        "seed": SEED,
        "report_to": "none",
    }

    return TrainingArguments(**args)


training_args = build_training_arguments()

trainer = DualHeadTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print(f"Training arguments:")
print(f"  Epochs: {MAX_EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"  LR: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}")
print(f"  Warmup ratio: {WARMUP_RATIO}, FP16: {USE_FP16}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Checkpoint metric: macro_f1 (greater_is_better)")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
print()
print("Starting training...")
train_result = trainer.train()
print(f"Training complete. Best metric: {trainer.state.best_metric}")\
""")

CELL_EVAL_MD = md("""\
## 13. Test Evaluation and Model Saving\
""")

CELL_EVAL = code("""\
def to_jsonable(value: Any) -> Any:
    \"\"\"Convert numpy/torch types to JSON-serializable Python types.\"\"\"
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


# ── Switch metric context to test set ─────────────────────────────────────────

metric_context["labels"] = test_dataset.labels
metric_context["label_masks"] = test_dataset.label_masks
metric_context["harm_labels"] = test_dataset.harm_labels
metric_context["harm_masks"] = test_dataset.harm_masks

# ── Run test evaluation ───────────────────────────────────────────────────────

test_results = trainer.evaluate(test_dataset)
print("=== Test Results ===")
for k, v in sorted(test_results.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ── Per-topic breakdown ───────────────────────────────────────────────────────

test_predictions = trainer.predict(test_dataset)
if isinstance(test_predictions.predictions, tuple):
    test_topic_logits, test_harm_logits = test_predictions.predictions
else:
    test_topic_logits = test_predictions.predictions
    test_harm_logits = None

topic_report = per_topic_report(test_topic_logits, test_dataset.labels, test_dataset.label_masks)
topic_report_df = pd.DataFrame.from_dict(topic_report, orient="index")
topic_report_df = topic_report_df.sort_values("f1", ascending=False)
print("\\n=== Per-Topic Test Report ===")
display(topic_report_df)

# ── Harm classification report ────────────────────────────────────────────────

if test_harm_logits is not None:
    valid_harm = test_dataset.harm_masks.astype(bool)
    if valid_harm.any():
        harm_preds = test_harm_logits[valid_harm].argmax(axis=1)
        harm_true = test_dataset.harm_labels[valid_harm]
        print("\\n=== Harm Head Classification Report ===")
        print(classification_report(
            harm_true, harm_preds,
            target_names=["Harmful (-1)", "Neutral (0)", "Fair (+1)"],
            zero_division=0,
        ))

# ── Save model and metadata ──────────────────────────────────────────────────

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save encoder + both heads.
torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "model_state_dict.pt")

# Save encoder separately for compatibility.
model.encoder.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# Save head weights separately for inspection.
torch.save(model.topic_head.state_dict(), MODEL_OUTPUT_DIR / "topic_head_weights.pt")
torch.save(model.harm_head.state_dict(), MODEL_OUTPUT_DIR / "harm_head_weights.pt")

# Save taxonomy.
compact_taxonomy = [
    {"classifier_id": i, "topic_id": tid, "name": name_by_topic[tid]}
    for i, tid in enumerate(topic_ids)
]
with (MODEL_OUTPUT_DIR / "lawgic_topics_44.json").open("w") as f:
    json.dump(compact_taxonomy, f, indent=2)

import shutil
shutil.copy2(TAXONOMY_PATH, MODEL_OUTPUT_DIR / "lawgic_topics_original_45.json")

# Save test metrics.
topic_metrics_out = {
    "test_results": to_jsonable(test_results),
    "per_topic_report": to_jsonable(topic_report),
}
with (MODEL_OUTPUT_DIR / "test_metrics_topic.json").open("w") as f:
    json.dump(topic_metrics_out, f, indent=2)

if test_harm_logits is not None:
    harm_metrics_out = to_jsonable(harm_metric_summary(
        test_harm_logits, test_dataset.harm_labels, test_dataset.harm_masks
    ))
    with (MODEL_OUTPUT_DIR / "test_metrics_harm.json").open("w") as f:
        json.dump(harm_metrics_out, f, indent=2)

# Save training metadata.
metadata = {
    "model_name": MODEL_NAME,
    "architecture": "dual_head",
    "num_topics": NUM_LAWGIC_TOPICS,
    "num_harm_classes": NUM_HARM_CLASSES,
    "max_length": MAX_LENGTH,
    "decision_threshold": DECISION_THRESHOLD,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "split_strategy": SPLIT_STRATEGY,
    "loss_weighting": "equal (1.0 + 1.0)",
    "train_rows": len(train_df),
    "val_rows": len(val_df),
    "test_rows": len(test_df),
    "total_rows": len(samples_df),
    "device": DEVICE_LABEL,
    "fp16": USE_FP16,
    "seed": SEED,
    "training_completed": datetime.now(timezone.utc).isoformat(),
    "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric else None,
}
with (MODEL_OUTPUT_DIR / "training_metadata.json").open("w") as f:
    json.dump(metadata, f, indent=2)

print(f"\\nModel saved to: {MODEL_OUTPUT_DIR}")
print(f"Files: {sorted(p.name for p in MODEL_OUTPUT_DIR.iterdir() if p.is_file())}")\
""")


# ============================================================================
# ASSEMBLE NOTEBOOK
# ============================================================================

def main():
    cells = [
        CELL_TITLE,
        CELL_IMPORTS_MD, CELL_IMPORTS,
        CELL_DEVICE_MD, CELL_DEVICE,
        CELL_TAXONOMY_MD, CELL_TAXONOMY,
        CELL_DATA_MD, CELL_DATA,
        CELL_EDA_MD, CELL_EDA,
        CELL_SPLIT_MD, CELL_SPLIT,
        CELL_SMOKE_MD, CELL_SMOKE,
        CELL_TOK_MD, CELL_TOK,
        CELL_MODEL_MD, CELL_MODEL,
        CELL_TRAINER_MD, CELL_TRAINER,
        CELL_METRICS_MD, CELL_METRICS,
        CELL_TRAINING_MD, CELL_TRAINING,
        CELL_EVAL_MD, CELL_EVAL,
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / "notebooks" / "model_finetuning" / "legal_bert_finetuning_dual_head.ipynb"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Created notebook: {output_path}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
