"""Parameterised re-implementation of the dual-head training protocol.

Phase 2 / Phase 3 support module. The original notebook
(``notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb``) is left
untouched; this file re-expresses the *same* protocol as a function so that the
encoder, the seed and the head configuration can vary while everything else —
splits, hyperparameters, early stopping, masked losses, degenerate-model
assertion — stays fixed.

Only three things are parameterised:

    encoder_name : HuggingFace model id
    seed         : 42 / 1337 / 2024
    heads        : "dual" | "topic" | "risk"

plus (Phase 3 only) an optional ``holdout_source`` that removes one corpus
source from train/validation and restricts evaluation to that source's rows.

Nothing here changes the taxonomy, the supervision masks or the loss
definitions; the loss code is a line-for-line copy of ``DualHeadTrainer``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lawgic_eval_core import (  # noqa: E402  (same-directory import)
    BATCH_SIZE,
    DECISION_THRESHOLD,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_LENGTH,
    NUM_HARM_CLASSES,
    NUM_LAWGIC_TOPICS,
    PROJECT_ROOT,
    TEXT_COLUMN,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    all_metrics,
    harm_metrics,
    label_arrays,
    load_corpus,
    load_taxonomy,
    per_topic_table,
    split_frames,
    topic_metrics,
)

RUNS_DIR = PROJECT_ROOT / "generated_files/lawgic_taxonomy/runs"

ENCODERS = [
    "nlpaueb/legal-bert-base-uncased",
    "bert-base-uncased",
    "xlnet-base-cased",
    "roberta-base",
]
SEEDS = [42, 1337, 2024]
HEAD_MODES = ["dual", "topic", "risk"]

# Architectures whose sequence summary lives at the LAST token rather than the
# first. XLNet is trained with the summary token appended at the end of the
# sequence; taking position 0 would read a content token instead.
LAST_TOKEN_ARCHITECTURES = {"xlnet"}


# ── Device ──────────────────────────────────────────────────────────────────


def detect_device() -> tuple[str, torch.device]:
    """Same autodetection as the original notebook."""
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


# ── Encoder adapter ─────────────────────────────────────────────────────────


def pooled_representation(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    architecture: str,
) -> torch.Tensor:
    """Attention-mask-aware sequence summary, one function for all encoders.

    BERT / Legal-BERT / RoBERTa: the summary is the FIRST real token
    (``[CLS]`` / ``<s>``). XLNet: the summary is the LAST real token.

    Selection is done from the attention mask rather than by fixed index so it
    is correct under either padding side (XLNet's tokenizer pads on the left).
    """
    if architecture in LAST_TOKEN_ARCHITECTURES:
        # Index of the last position whose mask is 1.
        position = attention_mask.size(1) - 1 - attention_mask.flip(1).argmax(dim=1)
    else:
        # Index of the first position whose mask is 1.
        position = attention_mask.argmax(dim=1)
    batch_index = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_index, position]


class LawgicDualHeadModel(nn.Module):
    """Encoder-agnostic dual-head model. Heads and losses are unchanged.

    Both heads are always constructed (so checkpoints stay shape-compatible);
    ``heads`` only controls which loss terms are summed.
    """

    def __init__(
        self,
        model_name: str,
        num_topics: int = NUM_LAWGIC_TOPICS,
        num_harm_classes: int = NUM_HARM_CLASSES,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.architecture = self.encoder.config.model_type
        self.topic_head = nn.Linear(hidden_size, num_topics)
        self.harm_head = nn.Linear(hidden_size, num_harm_classes)
        self.num_topics = num_topics
        self.num_harm_classes = num_harm_classes

    def forward(self, **encoder_inputs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(**encoder_inputs)
        pooled = pooled_representation(
            outputs.last_hidden_state,
            encoder_inputs["attention_mask"],
            self.architecture,
        )
        return self.topic_head(pooled), self.harm_head(pooled)


# ── Dataset / collator ──────────────────────────────────────────────────────


class LawgicDualHeadDataset(Dataset):
    """Same fields as the original notebook dataset, tokenizer-agnostic."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.frame = frame.reset_index(drop=True).copy()
        self.texts = self.frame[TEXT_COLUMN].astype(str).tolist()
        arrays = label_arrays(self.frame)
        self.labels = arrays["labels"]
        self.label_masks = arrays["label_masks"]
        self.harm_labels = arrays["harm_labels"]
        self.harm_masks = arrays["harm_masks"]
        self.encodings = tokenizer(self.texts, truncation=True, max_length=max_length, padding=False)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        item["label_mask"] = torch.tensor(self.label_masks[index], dtype=torch.float32)
        item["harm_label"] = torch.tensor(self.harm_labels[index], dtype=torch.int64)
        item["harm_mask"] = torch.tensor(self.harm_masks[index], dtype=torch.float32)
        return item


class LawgicDualHeadCollator:
    """Pads tokens, stacks the four target tensors.

    ``tokenizer.model_input_names`` decides which encoder inputs survive, which
    is how RoBERTa (no ``token_type_ids``) is handled without special-casing.
    """

    TARGET_KEYS = ("labels", "label_mask", "harm_label", "harm_mask")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.input_names = set(tokenizer.model_input_names)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_features = [
            {k: v for k, v in f.items() if k in self.input_names} for f in features
        ]
        batch = self.tokenizer.pad(token_features, return_tensors="pt")
        for key in self.TARGET_KEYS:
            batch[key] = torch.stack([f[key] for f in features])
        return batch


# ── Trainer ─────────────────────────────────────────────────────────────────


class DualHeadTrainer(Trainer):
    """Masked BCE (topic) + masked CE (risk), identical to the original.

    ``head_mode`` drops one term from the sum for the ablation; the surviving
    term is byte-identical to the dual-head version.
    """

    def __init__(self, *args, head_mode: str = "dual", metric_context: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_mode = head_mode
        self.metric_context = metric_context if metric_context is not None else {}

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        label_mask = inputs.pop("label_mask")
        labels = inputs.pop("labels")
        harm_label = inputs.pop("harm_label")
        harm_mask = inputs.pop("harm_mask")

        topic_logits, harm_logits = model(**inputs)

        bce_fn = nn.BCEWithLogitsLoss(reduction="none")
        per_topic_loss = bce_fn(topic_logits, labels)
        topic_loss = (per_topic_loss * label_mask).sum() / label_mask.sum().clamp(min=1.0)

        ce_fn = nn.CrossEntropyLoss(reduction="none")
        per_sample_ce = ce_fn(harm_logits, harm_label.clamp(min=0))
        harm_loss = (per_sample_ce * harm_mask).sum() / harm_mask.sum().clamp(min=1.0)

        if self.head_mode == "topic":
            total_loss = topic_loss
        elif self.head_mode == "risk":
            total_loss = harm_loss
        else:
            total_loss = topic_loss + harm_loss

        if return_outputs:
            return total_loss, SimpleNamespace(
                loss=total_loss,
                topic_logits=topic_logits.detach(),
                harm_logits=harm_logits.detach(),
            )
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        topic_labels = inputs.get("labels")
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, dict(inputs), return_outputs=True)
            loss = loss.detach().mean()
            logits = (outputs.topic_logits, outputs.harm_logits)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, topic_labels)


def make_compute_metrics(metric_context: dict):
    """compute_metrics closure reading label/mask arrays from ``metric_context``."""

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        topic_logits, harm_logits = predictions if isinstance(predictions, tuple) else (predictions, None)
        out = topic_metrics(
            topic_logits, metric_context["labels"], metric_context["label_masks"]
        )
        if harm_logits is not None:
            out.update(
                harm_metrics(harm_logits, metric_context["harm_labels"], metric_context["harm_masks"])
            )
        return out

    return compute_metrics


# ── Run configuration ───────────────────────────────────────────────────────


@dataclass
class RunConfig:
    encoder_name: str = "nlpaueb/legal-bert-base-uncased"
    seed: int = 42
    heads: str = "dual"  # dual | topic | risk
    holdout_source: str | None = None  # Phase 3: "claudette" | "100_tos"
    max_epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        encoder = self.encoder_name.split("/")[-1]
        parts = [encoder, f"seed{self.seed}", self.heads]
        if self.holdout_source:
            parts.append(f"holdout-{self.holdout_source}")
        return "__".join(parts)

    @property
    def best_metric_key(self) -> str:
        """Early stopping / checkpoint selection metric.

        Dual and topic-only select on topic macro-F1 (the original protocol).
        Risk-only has an untrained topic head, so it must select on the risk
        head instead — selecting on a frozen-random head would be meaningless.
        """
        return "risk_macro_f1" if self.heads == "risk" else "topic_macro_f1"


def assert_not_degenerate(arrays: dict[str, np.ndarray]) -> float:
    """Pre-training check: a zero-logit model must not score macro-F1 >= 0.95."""
    zero_logits = np.zeros_like(arrays["labels"])
    macro = topic_metrics(zero_logits, arrays["labels"], arrays["label_masks"])["topic_macro_f1"]
    if macro >= 0.95:
        raise ValueError(
            f"DEGENERATE: zero-logit macro F1 = {macro:.4f} >= 0.95. "
            "The dataset has no supervised negatives."
        )
    return macro


def source_row_mask(frame: pd.DataFrame, source: str) -> np.ndarray:
    """Boolean mask of rows annotated by ``source``."""
    return frame["sources"].map(lambda s: source in s).to_numpy(dtype=bool)


def source_supervision_mask(frame: pd.DataFrame, source: str) -> np.ndarray:
    """[N, 44] mask of the topic cells that ``source`` actually supervised.

    Phase 3 needs this: scoring a held-out probe on cells the held-out source
    never annotated would measure mask shape, not comprehension. Reconstructed
    from ``native_annotations`` (which topics that source asserted per row),
    intersected with the row's existing supervision mask.
    """
    topic_ids, _, _ = load_taxonomy()
    topic_position = {tid: i for i, tid in enumerate(topic_ids)}

    out = np.zeros((len(frame), NUM_LAWGIC_TOPICS), dtype=np.float32)
    for row_index, annotations in enumerate(frame["native_annotations"]):
        for annotation in annotations or []:
            if annotation.get("source_dataset") != source:
                continue
            position = topic_position.get(annotation.get("lawgic_topic_id"))
            if position is not None:
                out[row_index, position] = 1.0
    return out * np.vstack(frame["label_mask"].to_numpy()).astype(np.float32)


# ── The run function ────────────────────────────────────────────────────────


def run_config(config: RunConfig, save_dir: Path | None = None, verbose: bool = True) -> dict[str, Any]:
    """Train one configuration end to end and return its test metrics.

    Everything not named in ``RunConfig`` is fixed at the original protocol's
    value. The splits come from the persisted seed-42 assignment file, so every
    run in the matrix sees exactly the same clauses.
    """
    save_dir = save_dir or (RUNS_DIR / config.run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device_label, device = detect_device()
    use_fp16 = device_label == "cuda"

    corpus = load_corpus()
    frames = split_frames(corpus)
    train_frame, val_frame, test_frame = frames["train"], frames["validation"], frames["test"]

    holdout_note = {}
    if config.holdout_source:
        source = config.holdout_source
        train_frame = train_frame[~source_row_mask(train_frame, source)].copy()
        val_frame = val_frame[~source_row_mask(val_frame, source)].copy()
        test_frame = test_frame[source_row_mask(test_frame, source)].copy()
        holdout_note = {
            "holdout_source": source,
            "train_rows_after_holdout": len(train_frame),
            "val_rows_after_holdout": len(val_frame),
            "heldout_test_rows": len(test_frame),
        }

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name, use_fast=True)
    train_dataset = LawgicDualHeadDataset(train_frame, tokenizer)
    val_dataset = LawgicDualHeadDataset(val_frame, tokenizer)
    test_dataset = LawgicDualHeadDataset(test_frame, tokenizer)
    collator = LawgicDualHeadCollator(tokenizer)

    zero_logit_macro = assert_not_degenerate(label_arrays(val_frame))

    model = LawgicDualHeadModel(config.encoder_name).to(device)

    metric_context = {
        "labels": val_dataset.labels,
        "label_masks": val_dataset.label_masks,
        "harm_labels": val_dataset.harm_labels,
        "harm_masks": val_dataset.harm_masks,
    }

    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        num_train_epochs=config.max_epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model=config.best_metric_key,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=use_fp16,
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
    )

    trainer = DualHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(metric_context),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        head_mode=config.heads,
        metric_context=metric_context,
    )

    started = time.time()
    trainer.train()
    wall_seconds = time.time() - started

    # ── Test evaluation on frozen best checkpoint ───────────────────────────
    predictions = trainer.predict(test_dataset)
    topic_logits, harm_logits = (
        predictions.predictions
        if isinstance(predictions.predictions, tuple)
        else (predictions.predictions, None)
    )

    test_arrays = {
        "labels": test_dataset.labels,
        "label_masks": test_dataset.label_masks,
        "harm_labels": test_dataset.harm_labels,
        "harm_masks": test_dataset.harm_masks,
    }

    # Phase 3: restrict topic scoring to cells the held-out source supervised.
    if config.holdout_source:
        test_arrays["label_masks"] = source_supervision_mask(test_frame, config.holdout_source)

    metrics = all_metrics(topic_logits, harm_logits, test_arrays)
    if config.heads == "topic":
        for key in ("risk_accuracy", "risk_macro_f1", "risk_weighted_f1"):
            metrics[key] = float("nan")
    if config.heads == "risk":
        for key in ("topic_macro_f1", "topic_micro_f1", "topic_weighted_f1"):
            metrics[key] = float("nan")

    topic_ids, _, _ = load_taxonomy()

    record = {
        **asdict(config),
        "run_id": config.run_id,
        "device": device_label,
        "fp16": use_fp16,
        "wall_seconds": wall_seconds,
        "epochs_run": float(trainer.state.epoch or 0),
        "best_val_metric": trainer.state.best_metric,
        "zero_logit_macro_f1": zero_logit_macro,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        **holdout_note,
        **metrics,
    }

    np.savez_compressed(
        save_dir / "test_logits.npz",
        topic_logits=topic_logits,
        harm_logits=harm_logits,
        row_id=test_frame["row_id"].to_numpy(),
        labels=test_arrays["labels"],
        label_masks=test_arrays["label_masks"],
        harm_labels=test_arrays["harm_labels"],
        harm_masks=test_arrays["harm_masks"],
    )
    (save_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    per_topic_table(topic_logits, test_arrays, topic_ids).to_csv(
        save_dir / "per_topic.csv", index=False
    )

    if verbose:
        print(f"[{config.run_id}] {wall_seconds / 60:.1f} min | " + " ".join(
            f"{k}={metrics[k]:.4f}" for k in ("topic_macro_f1", "topic_micro_f1", "risk_accuracy", "risk_macro_f1")
        ))
    return record


def build_matrix() -> list[RunConfig]:
    """12 encoder runs + 6 legal-bert head-ablation runs (Phase 2)."""
    configs = [
        RunConfig(encoder_name=encoder, seed=seed, heads="dual")
        for encoder in ENCODERS
        for seed in SEEDS
    ]
    configs += [
        RunConfig(encoder_name=ENCODERS[0], seed=seed, heads=heads)
        for heads in ("topic", "risk")
        for seed in SEEDS
    ]
    return configs


def _self_check() -> None:
    """Cheap checks that need no training: pooling indices and matrix shape."""
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])  # right-pad, left-pad
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)

    first = pooled_representation(hidden, mask, "bert")
    assert torch.equal(first[0], hidden[0, 0]) and torch.equal(first[1], hidden[1, 2]), first

    last = pooled_representation(hidden, mask, "xlnet")
    assert torch.equal(last[0], hidden[0, 2]) and torch.equal(last[1], hidden[1, 4]), last

    matrix = build_matrix()
    assert len(matrix) == 18, len(matrix)
    assert len({c.run_id for c in matrix}) == 18
    assert RunConfig(heads="risk").best_metric_key == "risk_macro_f1"
    print("self-check: mask-aware pooling + 18-run matrix ✓")


if __name__ == "__main__":
    _self_check()
