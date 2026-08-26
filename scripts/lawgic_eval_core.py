"""Shared plumbing for the Lawgic evaluation / training-matrix notebooks.

Everything here is a faithful lift of logic that already exists in
``notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb`` (corpus
loading, 45->44 compaction, the seed-42 composite-stratified split, the masked
metric definitions) plus the TF-IDF near-duplicate machinery from
``scripts/near_duplicate_split_audit.py``. Nothing is redefined differently:
the taxonomy, masks and losses are untouched.

Why a module and not four copies in four notebooks: the whole point of the new
evaluation work is that every comparison sits on *identical* splits and
*identical* metric code. One import beats four copy-pastes that drift.

Run it directly to persist the seed-42 split assignment:

    python scripts/lawgic_eval_core.py

which writes ``generated_files/lawgic_taxonomy/splits/split_seed42.csv`` and
asserts the 21,183 / 2,648 / 2,648 row counts from training_metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── Constants mirrored from the fine-tuning notebook (do not change) ─────────

SEED = 42
BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 256
DECISION_THRESHOLD = 0.50
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.80, 0.10, 0.10
MIN_STRATUM_SIZE = 5

LEARNING_RATE = 3e-5
MAX_EPOCHS = 20
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

EXCLUDED_TOPIC_IDS = {"unclassified"}
NUM_HARM_CLASSES = 3
TEXT_COLUMN = "text"

HARM_CLASS_NAMES = {0: "Harmful", 1: "Neutral", 2: "Fair"}

# TF-IDF settings copied verbatim from scripts/near_duplicate_split_audit.py.
TFIDF_KWARGS = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2}
CONTAMINATION_THRESHOLD = 0.90


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until the Lawgic repository root is found."""
    start = (start or Path.cwd()).resolve()
    sentinel = Path("generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv")
    for candidate in (start, *start.parents):
        if (candidate / sentinel).exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root containing {sentinel}")


PROJECT_ROOT = find_project_root(Path(__file__).parent)

# ── Corpus version selector ───────────────────────────────────────────────────
# Set LAWGIC_CORPUS_VERSION=v2 (env var or change default here) to point this
# module at the 42-topic v2 corpus and v4 checkpoint. Default: v1 for backward
# compatibility with existing evaluation notebooks.
import os as _os
_CORPUS_VERSION = _os.environ.get("LAWGIC_CORPUS_VERSION", "v1")

if _CORPUS_VERSION == "v2":
    DATA_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide_v2.csv"
    TAXONOMY_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_topics_v2.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "saved_models/lawgic_classifier_legal-bert_v4"
    SPLIT_DIR = PROJECT_ROOT / "generated_files/lawgic_taxonomy/splits"
    SPLIT_PATH = SPLIT_DIR / "split_seed42_v2.csv"
    EVAL_OUT_DIR = PROJECT_ROOT / "generated_files/lawgic_taxonomy/evaluation_v2"
    EXPECTED_SPLIT_ROWS = {"train": 21243, "validation": 2655, "test": 2656}
else:
    # v1: the eval harness for the already-trained v3 checkpoint (768->44).
    DATA_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv"
    TAXONOMY_PATH = PROJECT_ROOT / "generated_files/lawgic_taxonomy/lawgic_topics.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "saved_models/lawgic_classifier_legal-bert_v3"
    SPLIT_DIR = PROJECT_ROOT / "generated_files/lawgic_taxonomy/splits"
    SPLIT_PATH = SPLIT_DIR / "split_seed42.csv"
    EVAL_OUT_DIR = PROJECT_ROOT / "generated_files/lawgic_taxonomy/evaluation"
    EXPECTED_SPLIT_ROWS = {"train": 21183, "validation": 2648, "test": 2648}


# ── Taxonomy ────────────────────────────────────────────────────────────────


def load_taxonomy() -> tuple[list[str], dict[str, str], list[int]]:
    """Return (topic_ids, name_by_topic, keep_indices) — same as notebook cell 6."""
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    all_ids = [topic["id"] for topic in payload["topics"]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate Lawgic topic IDs in taxonomy")
    if "unclassified" not in all_ids:
        raise ValueError("Taxonomy must define the `unclassified` runtime fallback")
    keep_indices = [i for i, tid in enumerate(all_ids) if tid not in EXCLUDED_TOPIC_IDS]
    topic_ids = [all_ids[i] for i in keep_indices]
    name_by_topic = {t["id"]: t.get("name", t["id"]) for t in payload["topics"]}
    return topic_ids, name_by_topic, keep_indices


# Derived from the taxonomy this module points to (see NOTE above), not
# hardcoded -- this is what let two duplicate topic columns go unnoticed.
TOPIC_IDS, TOPIC_NAME_BY_ID, KEEP_INDICES = load_taxonomy()
TOTAL_TAXONOMY_TOPICS = len(KEEP_INDICES) + len(EXCLUDED_TOPIC_IDS)
NUM_LAWGIC_TOPICS = len(TOPIC_IDS)


# ── Corpus ──────────────────────────────────────────────────────────────────

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


def _parse_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_corpus() -> pd.DataFrame:
    """Load + compact the fused corpus exactly as the fine-tuning notebook does.

    The returned frame keeps the original CSV row order and adds a stable
    ``row_id`` (the positional index in the *filtered* frame) that the split
    file keys on.
    """
    _, _, keep_indices = load_taxonomy()

    raw = pd.read_csv(DATA_PATH)
    df = raw.copy()
    for column in JSON_COLUMNS:
        df[column] = df[column].map(_parse_json_value)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype("string").str.strip()
    df["normalized_text"] = df["normalized_text"].astype("string").str.strip()
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN] != ""].copy()

    def compact(vector: list[Any], fill_value: float | None = None) -> list[Any]:
        if len(vector) != TOTAL_TAXONOMY_TOPICS:
            raise ValueError(f"Expected vector length {TOTAL_TAXONOMY_TOPICS}, found {len(vector)}")
        out = [vector[i] for i in keep_indices]
        if fill_value is not None:
            out = [fill_value if v is None else v for v in out]
        return out

    df["labels"] = df["labels_presence"].map(lambda v: compact(v, fill_value=0.0))
    df["label_mask"] = df["topic_mask"].map(lambda v: compact(v, fill_value=0.0))
    df["active_topic_ids_predicted"] = df["active_topic_ids"].map(
        lambda v: [t for t in v if t not in EXCLUDED_TOPIC_IDS]
    )
    df["harm_class"] = df["harm_score_class"].fillna(-1).astype(int)
    df["harm_mask_float"] = df["harm_mask"].astype(float)

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


# ── Split (verbatim reproduction of notebook cell 12) ───────────────────────


def _primary_stratify_label(active_topic_ids: list[str]) -> str:
    for topic_id in active_topic_ids:
        if topic_id not in EXCLUDED_TOPIC_IDS:
            return topic_id
    return "none"


def build_split_assignment(df: pd.DataFrame) -> pd.Series:
    """Recompute the seed-42 composite-stratified 80/10/10 assignment.

    Identical to the fine-tuning notebook: composite key
    ``f"{primary_topic}__harm{harm_class}"``, rare keys (<5) collapsed to
    ``__rare__``, two-stage ``train_test_split`` with ``random_state=42``.

    Returns a Series of 'train' / 'validation' / 'test' indexed like ``df``.
    """
    frame = df.copy()
    frame["stratify_key"] = frame.apply(
        lambda row: f"{_primary_stratify_label(row['active_topic_ids_predicted'])}__harm{int(row['harm_class'])}",
        axis=1,
    )

    key_counts = frame["stratify_key"].value_counts()
    rare_keys = set(key_counts[key_counts < MIN_STRATUM_SIZE].index)
    if rare_keys:
        frame.loc[frame["stratify_key"].isin(rare_keys), "stratify_key"] = "__rare__"

    train_df, valtest_df = train_test_split(
        frame,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=frame["stratify_key"],
        random_state=SEED,
    )

    vt_counts = valtest_df["stratify_key"].value_counts()
    vt_rare = set(vt_counts[vt_counts < 2].index)
    if vt_rare:
        valtest_df = valtest_df.copy()
        valtest_df.loc[valtest_df["stratify_key"].isin(vt_rare), "stratify_key"] = "__rare__"

    val_df, test_df = train_test_split(
        valtest_df,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=valtest_df["stratify_key"],
        random_state=SEED,
    )

    assignment = pd.Series("train", index=frame.index, dtype=object)
    assignment.loc[val_df.index] = "validation"
    assignment.loc[test_df.index] = "test"

    # Exact-text leakage check, same as the notebook.
    texts = {name: set(frame.loc[assignment == name, "normalized_text"]) for name in EXPECTED_SPLIT_ROWS}
    overlaps = {
        "train_val": len(texts["train"] & texts["validation"]),
        "train_test": len(texts["train"] & texts["test"]),
        "val_test": len(texts["validation"] & texts["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Exact text leakage detected across splits: {overlaps}")

    return assignment


def persist_split(force: bool = False) -> Path:
    """Write the seed-42 split assignment to disk and verify the row counts."""
    if SPLIT_PATH.exists() and not force:
        return SPLIT_PATH

    df = load_corpus()
    assignment = build_split_assignment(df)
    counts = assignment.value_counts().to_dict()
    if counts != EXPECTED_SPLIT_ROWS:
        raise AssertionError(
            f"Split row counts {counts} != expected {EXPECTED_SPLIT_ROWS}. "
            "The persisted split must match the trained checkpoint's split."
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": df["row_id"].to_numpy(),
            "split": assignment.to_numpy(),
            "normalized_text_sha": pd.util.hash_pandas_object(df["normalized_text"], index=False).to_numpy(),
        }
    ).to_csv(SPLIT_PATH, index=False)
    return SPLIT_PATH


def load_split(df: pd.DataFrame) -> pd.Series:
    """Attach the persisted split labels to ``df`` (by ``row_id``)."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} missing. Run `python scripts/lawgic_eval_core.py` once to persist it."
        )
    split_df = pd.read_csv(SPLIT_PATH)
    mapping = dict(zip(split_df["row_id"], split_df["split"]))
    assignment = df["row_id"].map(mapping)
    if assignment.isna().any():
        raise ValueError("Persisted split does not cover every corpus row — corpus changed?")
    return assignment


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    assignment = load_split(df)
    return {name: df[assignment == name].copy() for name in EXPECTED_SPLIT_ROWS}


# ── Label matrices ──────────────────────────────────────────────────────────


def label_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Stack the per-row label/mask columns into arrays for metric code."""
    return {
        "labels": np.vstack(frame["labels"].to_numpy()).astype(np.float32),
        "label_masks": np.vstack(frame["label_mask"].to_numpy()).astype(np.float32),
        "harm_labels": frame["harm_class"].to_numpy().astype(np.int64),
        "harm_masks": frame["harm_mask_float"].to_numpy().astype(np.float32),
    }


# ── Metrics (same definitions as notebook cells 14 and 22, vectorised) ──────


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def logits_to_predictions(logits: np.ndarray, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    return (sigmoid(logits) >= threshold).astype(int)


def topic_metrics(
    topic_logits: np.ndarray,
    labels: np.ndarray,
    label_masks: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Mask-aware macro/micro/weighted topic F1 over observed positions only.

    Vectorised equivalent of ``masked_metric_summary`` in the fine-tuning
    notebook (per-topic F1 with ``zero_division=0``, macro over topics that have
    at least one observed position, micro over the flattened observed cells).
    Vectorised because the bootstrap calls this ~1,000 times.
    """
    pred = logits_to_predictions(topic_logits, threshold).astype(bool)
    lab = labels.astype(bool)
    obs = label_masks.astype(bool)

    tp = (pred & lab & obs).sum(axis=0)
    fp = (pred & ~lab & obs).sum(axis=0)
    fn = (~pred & lab & obs).sum(axis=0)
    observed_topics = obs.any(axis=0)

    denom = 2 * tp + fp + fn
    per_topic_f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0)
    per_topic_f1 = per_topic_f1[observed_topics]
    support = lab[:, observed_topics].astype(int) * obs[:, observed_topics]
    support = support.sum(axis=0)

    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_denom = 2 * micro_tp + micro_fp + micro_fn

    return {
        "topic_macro_f1": float(per_topic_f1.mean()) if per_topic_f1.size else 0.0,
        "topic_micro_f1": float(2 * micro_tp / micro_denom) if micro_denom > 0 else 0.0,
        "topic_weighted_f1": float(np.average(per_topic_f1, weights=support))
        if support.sum() > 0
        else 0.0,
        "topic_observed_positions": float(obs.sum()),
    }


def harm_metrics(
    harm_logits: np.ndarray,
    harm_labels: np.ndarray,
    harm_masks: np.ndarray,
) -> dict[str, float]:
    """Risk-head accuracy / macro-F1 / weighted-F1 over harm_mask=1 rows."""
    valid = harm_masks.astype(bool)
    if not valid.any():
        return {"risk_accuracy": 0.0, "risk_macro_f1": 0.0, "risk_weighted_f1": 0.0, "risk_valid_rows": 0.0}
    preds = harm_logits[valid].argmax(axis=1)
    true = harm_labels[valid]
    return {
        "risk_accuracy": float(accuracy_score(true, preds)),
        "risk_macro_f1": float(f1_score(true, preds, average="macro", zero_division=0)),
        "risk_weighted_f1": float(f1_score(true, preds, average="weighted", zero_division=0)),
        "risk_valid_rows": float(valid.sum()),
    }


HEADLINE_METRICS = ("topic_macro_f1", "topic_micro_f1", "risk_accuracy", "risk_macro_f1")


def all_metrics(
    topic_logits: np.ndarray,
    harm_logits: np.ndarray,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Headline metrics for a (sub)set of rows, given precomputed logits."""
    idx = slice(None) if indices is None else indices
    out = topic_metrics(topic_logits[idx], arrays["labels"][idx], arrays["label_masks"][idx])
    out.update(harm_metrics(harm_logits[idx], arrays["harm_labels"][idx], arrays["harm_masks"][idx]))
    out["rows"] = float(len(topic_logits[idx]))
    return out


def per_topic_table(
    topic_logits: np.ndarray,
    arrays: dict[str, np.ndarray],
    topic_ids: list[str],
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """Per-topic precision / recall / F1 / support, plus macro and weighted rows."""
    pred = logits_to_predictions(topic_logits, threshold).astype(bool)
    lab = arrays["labels"].astype(bool)
    obs = arrays["label_masks"].astype(bool)

    rows = []
    for i, topic_id in enumerate(topic_ids):
        m = obs[:, i]
        if not m.any():
            continue
        tp = int((pred[m, i] & lab[m, i]).sum())
        fp = int((pred[m, i] & ~lab[m, i]).sum())
        fn = int((~pred[m, i] & lab[m, i]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(
            {
                "topic_id": topic_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
                "observed": int(m.sum()),
            }
        )

    table = pd.DataFrame(rows)
    support = table["support"].to_numpy()
    weights = support if support.sum() > 0 else np.ones_like(support)
    macro = {"topic_id": "macro avg", **{c: table[c].mean() for c in ("precision", "recall", "f1")}}
    weighted = {
        "topic_id": "weighted avg",
        **{c: float(np.average(table[c], weights=weights)) for c in ("precision", "recall", "f1")},
    }
    macro["support"] = weighted["support"] = int(support.sum())
    macro["observed"] = weighted["observed"] = int(table["observed"].sum())
    return pd.concat([table, pd.DataFrame([macro, weighted])], ignore_index=True)


# ── Bootstrap ───────────────────────────────────────────────────────────────


def bootstrap_ci(
    topic_logits: np.ndarray,
    harm_logits: np.ndarray,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = SEED,
    metrics: Iterable[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Percentile bootstrap 95% CIs, resampling *clauses* with replacement."""
    idx = np.arange(len(topic_logits)) if indices is None else np.asarray(indices)
    rng = np.random.default_rng(seed)
    point = all_metrics(topic_logits, harm_logits, arrays, idx)

    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        sample = rng.choice(idx, size=len(idx), replace=True)
        values = all_metrics(topic_logits, harm_logits, arrays, sample)
        for m in draws:
            draws[m].append(values[m])

    return pd.DataFrame(
        [
            {
                "metric": m,
                "point": point[m],
                "mean": float(np.mean(draws[m])),
                "ci_low": float(np.percentile(draws[m], 2.5)),
                "ci_high": float(np.percentile(draws[m], 97.5)),
                "n_rows": int(len(idx)),
            }
            for m in draws
        ]
    )


def paired_bootstrap_delta(
    metric_fn,
    idx: np.ndarray,
    n_resamples: int = 1000,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap over clauses for a metric *difference* between two models.

    ``metric_fn(sample_indices) -> float`` must return the delta (A minus B)
    computed on the same resampled clause indices for both models.
    """
    rng = np.random.default_rng(seed)
    observed = metric_fn(idx)
    deltas = np.array([metric_fn(rng.choice(idx, size=len(idx), replace=True)) for _ in range(n_resamples)])
    # Two-sided p: fraction of resamples whose centred delta exceeds the observed.
    centred = deltas - deltas.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": p_value,
        "n_resamples": n_resamples,
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact-ish McNemar test on item-level correctness (binomial, two-sided)."""
    from scipy.stats import binomtest

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    return {"b": b, "c": c, "p_value": float(binomtest(b, b + c, 0.5).pvalue)}


# ── Near-duplicate contamination (mirrors scripts/near_duplicate_split_audit) ─


def max_train_similarity(train_texts: list[str], eval_texts: list[str]) -> np.ndarray:
    """Max TF-IDF char n-gram cosine similarity of each eval text to any train text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    train_matrix = vectorizer.fit_transform(train_texts)
    eval_matrix = vectorizer.transform(eval_texts)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_matrix)
    distances, _ = nn.kneighbors(eval_matrix)
    return 1.0 - distances.ravel()


# ── Trained-checkpoint inference (saved_models/..._v3) ──────────────────────


def load_trained_checkpoint(device=None):
    """Load the existing v3 dual-head checkpoint, architecture unchanged.

    This mirrors ``api/server.py``: the trained model reads ``pooler_output``
    (BERT's dense+tanh over ``[CLS]``). The Phase 2 matrix uses raw first/last
    token pooling instead so all four encoders are treated identically — this
    loader must keep the *original* pooling or the saved weights would be read
    through a different head input.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    class TrainedDualHeadModel(nn.Module):
        def __init__(self, model_dir: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_dir)
            hidden = self.encoder.config.hidden_size
            self.topic_head = nn.Linear(hidden, NUM_LAWGIC_TOPICS)
            self.harm_head = nn.Linear(hidden, NUM_HARM_CLASSES)

        def forward(self, **inputs):
            pooled = self.encoder(**inputs).pooler_output
            return self.topic_head(pooled), self.harm_head(pooled)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model = TrainedDualHeadModel(str(CHECKPOINT_DIR))
    state_path = CHECKPOINT_DIR / "model_state_dict.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    else:
        model.topic_head.load_state_dict(
            torch.load(CHECKPOINT_DIR / "topic_head_weights.pt", map_location=device, weights_only=True)
        )
        model.harm_head.load_state_dict(
            torch.load(CHECKPOINT_DIR / "harm_head_weights.pt", map_location=device, weights_only=True)
        )
    return model.to(device).eval(), tokenizer, device


def predict_logits(model, tokenizer, texts: list[str], device, batch_size: int = 32, cache: Path | None = None):
    """Batched forward pass -> (topic_logits, harm_logits) as numpy arrays."""
    import torch

    if cache is not None and cache.exists():
        payload = np.load(cache)
        return payload["topic_logits"], payload["harm_logits"]

    topic_chunks, harm_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            topic, harm = model(**batch)
            topic_chunks.append(topic.float().cpu().numpy())
            harm_chunks.append(harm.float().cpu().numpy())

    topic_logits = np.concatenate(topic_chunks)
    harm_logits = np.concatenate(harm_chunks)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, topic_logits=topic_logits, harm_logits=harm_logits)
    return topic_logits, harm_logits


# ── Provenance / document identifiers ───────────────────────────────────────


def document_identifier(native_annotations: list[dict] | None, sources: list[str]) -> str | None:
    """Best-effort document/service identifier from the provenance columns.

    - claudette : ``source_id`` is ``"<service>:<idx>:<tag>"``  -> service
    - 100_tos   : ``source_id`` is ``"<Platform>:<idx>"``       -> platform
    - tos_dr    : ``source_id`` is a bare ToS;DR point id; the service must be
                  recovered by joining ``datasets/tos_dr/points.csv``.

    Returns ``"<source>:<doc>"`` or None when no identifier can be derived.
    """
    if not native_annotations:
        return None
    for ann in native_annotations:
        dataset = ann.get("source_dataset")
        source_id = str(ann.get("source_id", ""))
        if dataset in ("claudette", "100_tos") and ":" in source_id:
            return f"{dataset}:{source_id.split(':')[0]}"
    return None


def tosdr_service_map() -> dict[str, str]:
    """point_id -> service_id, from the raw ToS;DR export (if present)."""
    points_csv = PROJECT_ROOT / "datasets/tos_dr/points.csv"
    if not points_csv.exists():
        return {}
    points = pd.read_csv(points_csv, usecols=["id", "service_id"], low_memory=False)
    points = points.dropna(subset=["service_id"])
    return {str(int(r.id)): f"tos_dr:{int(r.service_id)}" for r in points.itertuples()}


# ── LaTeX helpers ───────────────────────────────────────────────────────────


def to_booktabs(df: pd.DataFrame, caption: str, label: str, column_format: str | None = None) -> str:
    """Render a DataFrame as a booktabs LaTeX table (no index)."""
    body = df.to_latex(
        index=False,
        escape=True,
        column_format=column_format or "l" + "r" * (df.shape[1] - 1),
        float_format="%.3f",
    )
    body = body.replace(r"\toprule", r"\toprule").replace("\\begin{tabular}", "\\begin{tabular}")
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def write_outputs(df: pd.DataFrame, stem: str, caption: str, label: str) -> tuple[Path, Path]:
    """Write ``<stem>.csv`` and ``<stem>.tex`` into the evaluation output dir."""
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_OUT_DIR / f"{stem}.csv"
    tex_path = EVAL_OUT_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(to_booktabs(df, caption, label), encoding="utf-8")
    return csv_path, tex_path


# ── Self-check / split persistence entry point ──────────────────────────────


def _self_check() -> None:
    """Assert the vectorised metrics agree with the notebook's sklearn versions."""
    rng = np.random.default_rng(0)
    n, k = 200, NUM_LAWGIC_TOPICS
    logits = rng.normal(size=(n, k))
    labels = (rng.random((n, k)) < 0.2).astype(np.float32)
    masks = (rng.random((n, k)) < 0.4).astype(np.float32)

    pred = logits_to_predictions(logits)
    bool_mask = masks.astype(bool)
    sk_per_topic = [
        f1_score(labels[bool_mask[:, i], i].astype(int), pred[bool_mask[:, i], i], zero_division=0)
        for i in range(k)
        if bool_mask[:, i].any()
    ]
    sk_macro = float(np.mean(sk_per_topic))
    sk_micro = f1_score(
        labels.reshape(-1)[bool_mask.reshape(-1)].astype(int),
        pred.reshape(-1)[bool_mask.reshape(-1)],
        zero_division=0,
    )
    ours = topic_metrics(logits, labels, masks)
    assert abs(ours["topic_macro_f1"] - sk_macro) < 1e-9, (ours["topic_macro_f1"], sk_macro)
    assert abs(ours["topic_micro_f1"] - sk_micro) < 1e-9, (ours["topic_micro_f1"], sk_micro)
    print("self-check: vectorised masked F1 matches sklearn reference ✓")


if __name__ == "__main__":
    _self_check()
    path = persist_split(force=True)
    counts = pd.read_csv(path)["split"].value_counts().to_dict()
    print(f"Persisted split -> {path}")
    print(f"Row counts: {counts}")
