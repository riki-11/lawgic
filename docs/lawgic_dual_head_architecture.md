# Lawgic Dual-Head Architecture

This document describes the dual-head Legal-BERT model used by Lawgic to predict both
topic presence and consumer harm from Terms of Service clauses. It is a companion to
`notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb` (generated from
`notebooks/model_finetuning/create_dual_head_notebook.py`).

Training data: `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv`.

## Motivation: The Degenerate Positive-Only Trap

The original single-head pipeline trained a multi-label classifier using masked BCE loss
over 44 Lawgic topics. The data fusion step set `mask[c] = 1` **only** where
`label_presence[c] = 1`, meaning every supervised training entry was a positive example.

| Metric                           | Value       |
| -------------------------------- | ----------- |
| Supervised entries where label=1 | 49,154      |
| Supervised entries where label=0 | **0**       |
| Positive ratio among supervised  | **1.0000**  |

The model trivially minimised loss by predicting all positives, producing a degenerate
validation macro F1 of 1.00. No amount of hyperparameter tuning or regularisation can
fix a dataset that contains zero negatives.

### Root Cause

The wide-format builder (`build_wide_df`) activated `mask[c]` only for topics that
appeared in the long-format annotation table for a given text. By construction, every
such entry also had `presence[c] = 1`. The mask faithfully represented "this source
annotated this topic for this text," but that set was identical to "this topic is present
for this text." There was no mechanism to express "this source evaluated this topic and
found it absent."

## Architecture Overview

The dual-head model replaces the single-head classifier with a shared encoder feeding
two independent output heads:

```
Input text
    │
    ▼
┌──────────────────────────────────────┐
│  Legal-BERT Encoder                  │
│  (nlpaueb/legal-bert-base-uncased)   │
│  12 transformer layers, 768 hidden   │
└──────────────┬───────────────────────┘
               │
          [CLS] pooler output
          (batch_size, 768)
               │
       ┌───────┴────────┐
       │                 │
       ▼                 ▼
  ┌──────────┐    ┌───────────┐
  │Topic Head│    │ Harm Head │
  │ Linear   │    │  Linear   │
  │(768, 44) │    │ (768, 3)  │
  └────┬─────┘    └─────┬─────┘
       │                 │
       ▼                 ▼
  topic_logits      harm_logits
  (batch, 44)       (batch, 3)
       │                 │
       ▼                 ▼
  Masked BCE        Cross-Entropy
  Loss              Loss
       │                 │
       └────────┬────────┘
                │
         Total Loss = L_topic + L_harm
```

### Head 1 — Topic Presence (Multi-Label)

Predicts which of the 44 Lawgic topics (excluding `unclassified`) are present in a
clause. Uses sigmoid activation at inference and masked binary cross-entropy during
training.

### Head 2 — Consumer Harm (3-Class)

Predicts the overall consumer-harm posture of the clause as one of three classes:

| Class Index | Harm Score | Semantic Label |
| ----------- | ---------- | -------------- |
| 0           | −1         | Harmful / Bad  |
| 1           | 0          | Neutral        |
| 2           | +1         | Fair / Good    |

Uses softmax activation at inference and standard cross-entropy during training.

### Why a Custom `nn.Module`?

`AutoModelForSequenceClassification` supports only a single classification head with one
loss function. The dual-head setup needs two independent linear layers on the shared
`[CLS]` pooler output, each with a different loss (masked BCE vs cross-entropy). The
notebook therefore wraps `AutoModel` in `LawgicDualHeadModel` rather than using the
sequence-classification wrapper.

## Joint Multi-Objective Loss Function

The total training loss is an equally weighted sum of the two head losses:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{topic}} + \mathcal{L}_{\text{harm}}
$$

### Topic Loss — Masked BCE with Logits

For a batch of size $B$ with $C = 44$ topic dimensions:

$$
\mathcal{L}_{\text{topic}} = \frac{
    \sum_{b=1}^{B} \sum_{c=1}^{C} m_{b,c} \cdot \text{BCE}(\hat{y}_{b,c},\; y_{b,c})
}{
    \sum_{b=1}^{B} \sum_{c=1}^{C} m_{b,c}
}
$$

where:

- $\hat{y}_{b,c}$ is the raw logit from the topic head (pre-sigmoid).
- $y_{b,c} \in \{0, 1\}$ is the topic presence label.
- $m_{b,c} \in \{0, 1\}$ is the topic mask (1 = supervised, 0 = unknown).

Dividing by $\sum m$ normalises loss magnitude across batches with different mask
coverage.

### Harm Loss — Masked Cross-Entropy

For the same batch, but only over rows where the harm label is valid:

$$
\mathcal{L}_{\text{harm}} = \frac{
    \sum_{b=1}^{B} h_b \cdot \text{CE}(\hat{\mathbf{z}}_b,\; k_b)
}{
    \sum_{b=1}^{B} h_b
}
$$

where:

- $\hat{\mathbf{z}}_b \in \mathbb{R}^3$ is the raw logit vector from the harm head.
- $k_b \in \{0, 1, 2\}$ is the target class index.
- $h_b \in \{0, 1\}$ is the harm mask (1 = valid harm label, 0 = unresolvable).

The `harm_mask` field ($h_b$) allows the architecture to handle rows with unresolvable scores (e.g., if all sources yield `null`). Currently, pessimistic resolution resolves all rows, so $h_b = 1$ for the entire dataset, and all rows participate in the harm loss.

## Data Mapping Invariants

### Invariant 1 — Source-Aware Topic Masking (The Negative Fix)

The corrected mask generation expands `mask[c] = 1` to cover **all topics within each
contributing source's taxonomy coverage**, not just topics that are explicitly present.

**Per-source rules:**

| Source     | Mask Rule                                                                      | Coverage   |
| ---------- | ------------------------------------------------------------------------------ | ---------- |
| ToS;DR     | `mask[c] = 1` for all topics in ToS;DR's `source_mappings` in the taxonomy     | 42 topics  |
| 100 ToS    | `mask[c] = 1` for all topics in 100 ToS's `source_mappings` in the taxonomy    | 30 topics  |
| CLAUDETTE  | `mask[c] = 1` **only** for the specific topic(s) mapped by `CLAUDETTE_TOPIC_RULES` | 1–2 per annotation, 1–4 per clause |

> **Correction, 31 July 2026.** This table previously reported 37 and 26. Neither figure
> was ever measured. `build_source_coverage_mask()` (cell 20 of
> `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`) yields 42, 30, and 21 against the
> committed taxonomy, and the built corpus shows mask widths of exactly 42 and 30 on
> single-source rows.
>
> Note the asymmetry these numbers hide: the mask reads raw `source_mappings`, but ToS;DR
> labels are produced through `TOSDR_TOPIC_OVERRIDES`, which is narrower. ToS;DR reaches
> only 38 topics with positives while its mask supervises 42. See
> `docs/lawgic_coverage_discrepancy_report.md`.

**Why CLAUDETTE uses Option A (narrow masking):**

CLAUDETTE annotates only 9 native labels, which `CLAUDETTE_TOPIC_RULES` collapses onto 10
Lawgic topics. Unlike ToS;DR and
100 ToS — which have broad enough taxonomies that absence of annotation likely means
absence of the topic — CLAUDETTE's narrow scope means a missing annotation tells us
nothing about topics outside its coverage. Setting `mask[c] = 1` for topics CLAUDETTE
never evaluates would manufacture false negatives.

**Union rule:** When a text has annotations from multiple sources (e.g., both CLAUDETTE
and ToS;DR), the final mask is the **element-wise maximum** (union) of all per-source
masks. This ensures supervised coverage from any source is preserved.

**Result:** The mask matrix now contains both supervised positives (`mask=1, label=1`)
and supervised negatives (`mask=1, label=0`), eliminating the degenerate positive-only
training signal.

### Invariant 2 — Pessimistic Conflict Resolution

When the same text has different harm scores from different sources for the same topic,
the conflict is resolved using a **pessimistic consumer protection** policy:

```
If any source scores −1 → resolved score = −1
Otherwise              → resolved score = min(all non-null scores)
```

This applies at two levels:

1. **Per-topic level** (in the wide-format builder): Conflict rows that previously had
   `scores[c] = null` now receive the pessimistic minimum.
2. **Row level** (for the harm head): The single row-level `harm_score` is the minimum
   across all resolved per-topic scores.

The 47 conflict rows are no longer dropped or nullified. They participate in training
with the resolved pessimistic score.

### Invariant 3 — Harm Score Class Mapping

The model's cross-entropy head requires non-negative integer class indices:

| Raw Score | Class Index | Meaning  |
| --------- | ----------- | -------- |
| −1        | 0           | Harmful  |
| 0         | 1           | Neutral  |
| +1        | 2           | Fair     |

The `harm_mask` field is $1.0$ for all current rows, as pessimistic resolution successfully assigned a score to every text. Should future data contain unresolvable rows (where all topic scores remain null), $h_b = 0.0$ will exclude them from the harm-head loss while preserving their participation in topic-head training.

### Invariant 4 — ToS;DR Blocker Mapping

ToS;DR `blocker` classifications are treated identically to `bad`, both mapping to a
harm score of −1 (class index 0). These rows are tagged with
`parse_status="blocker_as_bad"` for auditability.

## Split Strategy

The data is split 80/10/10 using **multi-objective stratified sampling**. The
stratification key combines:

1. The primary active topic ID (first non-`unclassified` topic).
2. The resolved harm score class.

```
stratify_key = f"{primary_topic_id}__harm{harm_score_class}"
```

This ensures both topic distribution and harm-class distribution are balanced across
train, validation, and test sets. An exact-text leakage check raises an error if any
`text` string appears in more than one split.

## PyTorch Implementation

### `LawgicDualHeadDataset` and `LawgicDualHeadCollator`

Each training example exposes:

| Field          | Shape / type   | Role                                      |
| -------------- | -------------- | ----------------------------------------- |
| `input_ids`    | `[seq_len]`    | Tokenised clause text                     |
| `attention_mask` | `[seq_len]`  | Padding mask                              |
| `labels`       | `[44]` float   | Topic presence targets (0/1)              |
| `label_mask`   | `[44]` float   | Topic supervision mask (1 = observed)     |
| `harm_label`   | scalar int64   | Harm class index {0, 1, 2}              |
| `harm_mask`    | scalar float   | Harm supervision mask (1 = valid label)   |

The collator pads token sequences with the HuggingFace tokenizer and stacks the
fixed-size label tensors into batch tensors.

### `DualHeadTrainer`

A HuggingFace `Trainer` subclass wires the custom model into the training loop.

**`compute_loss`** pops `label_mask`, `labels`, `harm_label`, and `harm_mask` from
each batch (so they are not passed to `forward`), runs the encoder and both heads,
and returns the equally weighted sum of masked BCE (topic) and masked CE (harm).

**`prediction_step`** (required for evaluation) addresses two HuggingFace
constraints:

1. `LawgicDualHeadModel.forward` has no `labels` parameter, so the Trainer does
   not auto-detect `label_names` and the default `prediction_step` returns
   `labels=None`.
2. The evaluation loop only calls `compute_metrics` when **both** gathered
   predictions and labels are non-null.

The override runs `compute_loss(..., return_outputs=True)`, returns
`(topic_logits, harm_logits)` as predictions, and explicitly returns batch
`labels` so mask-aware metrics are computed. Without this override, evaluation
produces an empty metrics dict and checkpoint selection on `eval_macro_f1` fails.

### Metric Context

Topic and harm masks are not part of the Trainer's default prediction payload.
Before training, the validation set's full `labels`, `label_masks`, `harm_labels`,
and `harm_masks` arrays are stored in a module-level `metric_context` dict.
`compute_metrics` unpacks the dual-head logit tuple from `EvalPrediction` and
joins predictions with these pre-stored masks for mask-aware scoring.

## Evaluation Metrics

Logged metric names are prefixed with `eval_` by the Trainer (e.g. `macro_f1` →
`eval_macro_f1`).

### Topic Head Metrics (Mask-Aware)

All topic metrics operate only on `(mask=1)` positions:

- **Masked Macro F1** (`eval_macro_f1`, primary checkpoint metric): Per-topic F1
  over observed positions, averaged across topics.
- **Masked Micro F1** (`eval_micro_f1`): Global binary F1 over all observed topic
  positions.
- **Masked Weighted F1** (`eval_weighted_f1`): Per-topic F1 weighted by positive
  support.
- **Predicted Positive Rate** (`eval_predicted_positive_rate`): Share of
  supervised positions predicted positive.
- **Masked Positions** (`eval_masked_positions`): Count of supervised topic cells.

### Harm Head Metrics (Multi-Class)

Computed only on rows where `harm_mask = 1`:

- **Harm Accuracy** (`eval_harm_accuracy`): Fraction of correctly classified samples.
- **Harm Weighted F1** (`eval_harm_weighted_f1`): Weighted F1 across the three classes.
- **Harm Macro F1** (`eval_harm_macro_f1`): Unweighted mean F1 across classes.
- **Harm Valid Samples** (`eval_harm_valid_samples`): Row count with valid harm labels.
- **Per-Class Precision / Recall**: Breakdown by Harmful, Neutral, Fair (test-set
  classification report only).

## Training Configuration

The dual-head model inherits baseline hyperparameters from the single-head experiment:

| Parameter               | Value                                      |
| ----------------------- | ------------------------------------------ |
| Base model              | `nlpaueb/legal-bert-base-uncased`          |
| Max sequence length     | 256 tokens                                 |
| Learning rate           | 3 × 10⁻⁵                                  |
| Train batch size        | 8 per device                               |
| Eval batch size         | 16 per device (`BATCH_SIZE × 2`)           |
| Max epochs              | 20                                         |
| Early stopping          | Patience 3 (on `eval_macro_f1`)            |
| Weight decay            | 0.01                                       |
| Warmup ratio            | 0.06                                       |
| Decision threshold      | 0.50 (topic head, at inference)            |
| Loss weighting          | Equal (λ = 1.0 for both heads)             |
| FP16                    | Enabled on CUDA only; disabled on MPS/CPU  |
| Checkpoint retention    | `save_total_limit=2`, best by `macro_f1`     |
| `remove_unused_columns` | `False` (required for custom mask fields)  |
| Eval / save strategy    | Every epoch                                |

Device detection prefers CUDA, then Apple MPS, then CPU. CUDA memory is read via
`total_memory` with a fallback to the legacy `total_mem` attribute for older PyTorch
builds.

`build_training_arguments()` inspects the installed `transformers` version and sets
`eval_strategy` or `evaluation_strategy` accordingly.

## Resolved Harm Score Distribution

After pessimistic conflict resolution across all 26,479 wide rows:

| Class       | Count  | Share  |
| ----------- | ------ | ------ |
| Harmful (−1)| 8,311  | 31.4%  |
| Neutral (0) | 12,368 | 46.7%  |
| Fair (+1)   | 5,800  | 21.9%  |
| No label    | 0      | 0.0%   |

## Output Artefacts

The trained model and metadata are saved under
`saved_models/lawgic_classifier_legal-bert_v3/`:

```
saved_models/lawgic_classifier_legal-bert_v3/
├── model_state_dict.pt        # Full dual-head state dict (encoder + both heads)
├── topic_head_weights.pt      # Topic head Linear weights only
├── harm_head_weights.pt       # Harm head Linear weights only
├── config.json                # Encoder config (from save_pretrained)
├── model.safetensors          # Encoder weights (from save_pretrained)
├── tokenizer files
├── lawgic_topics_44.json      # Compact 44-topic classifier mapping
├── lawgic_topics_original_45.json
├── test_metrics_topic.json    # Test metrics + per-topic report
├── test_metrics_harm.json     # Harm head test metrics
├── training_metadata.json     # Hyperparameters, split sizes, best metric, timing
└── checkpoints/               # HuggingFace Trainer epoch checkpoints
```

`training_metadata.json` records `architecture: "dual_head"`, split row counts,
device, FP16 flag, and the best validation `macro_f1` observed during training.
