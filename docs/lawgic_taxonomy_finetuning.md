# Lawgic Taxonomy Fine-Tuning

This document describes `notebooks/model_finetuning/legal_bert_finetuning_lawgic_taxonomy.ipynb`, which fine-tunes Legal-BERT on the fused Lawgic taxonomy dataset.

## Purpose

The notebook trains `nlpaueb/legal-bert-base-uncased` as a multi-label classifier over the curated Lawgic taxonomy. It uses the fused 3-source dataset from CLAUDETTE, ToS;DR, and 100 ToS, rather than the older ToS;DR-only training export.

The model predicts Lawgic topics for terms-of-service clauses. It does not yet predict harm scores; score-head training remains future work.

## Relationship to the Prior Notebook

The previous notebook, `notebooks/model_finetuning/legal_bert_finetuning.ipynb`, trained on ToS;DR points only and used standard multi-label BCE over observed ToS;DR topics.

The taxonomy notebook changes three core pieces:

- It trains from `generated_files/lawgic_taxonomy/lawgic_combined_wide.csv`, a fused 3-source dataset with 26,479 wide rows.
- It uses masked Binary Cross-Entropy so unknown labels do not become false negatives.
- It trains on 44 curated Lawgic topics, excluding `unclassified`.

## Inputs

Primary inputs:

- `generated_files/lawgic_taxonomy/lawgic_combined_wide.csv`
- `generated_files/lawgic_taxonomy/lawgic_topics.json`
- `generated_files/lawgic_taxonomy/lawgic_fusion_summary.json`

Important wide CSV columns:

- `text`: raw clause text used as model input.
- `normalized_text`: NFKC-cleaned grouping key, used for diagnostics and leakage checks.
- `sources`: JSON list of source datasets contributing to the row.
- `labels_presence`: JSON 45-float multi-hot vector.
- `mask`: JSON 45-float vector indicating observed/evaluable label positions.
- `scores`: JSON 45-value score vector, retained for diagnostics but not used as training target.
- `active_topic_ids`: JSON list of active topic IDs.
- `conflict_topic_ids`: JSON list of topics with source score disagreement.
- `has_score_conflict`: row-level conflict flag.
- `native_annotations`: JSON annotation provenance.

## Label Space

The source taxonomy has 45 topics. The notebook drops `unclassified` before training, so the classifier head has 44 outputs.

`unclassified` is excluded because it is a fallback retention bucket, not a stable legal concept. Keeping it would teach the model noisy residual behavior. The notebook still validates original 45-length vectors, then slices out taxonomy index 44 to create compact 44-length `labels`, `label_mask`, and `scores_44` fields.

The saved compact taxonomy is written to:

- `saved_models/lawgic_classifier_legal-bert_v2/lawgic_topics_44.json`

The original 45-topic taxonomy is copied for traceability:

- `saved_models/lawgic_classifier_legal-bert_v2/lawgic_topics_original_45.json`

## Masked BCE

The fused dataset does not mean every unmarked topic is negative. A source may only annotate a subset of topics. Standard BCE would punish the model for predicting positives in unknown positions.

The notebook uses masked BCE:

```python
loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
per_topic_loss = loss_fn(logits, labels)
masked_loss = (per_topic_loss * label_mask).sum()
masked_loss = masked_loss / label_mask.sum().clamp(min=1.0)
```

Only `label_mask == 1` positions contribute to loss. Normalizing by the mask sum keeps loss magnitude comparable across batches with different coverage.

`remove_unused_columns=False` is required in `TrainingArguments` so Hugging Face `Trainer` does not discard `label_mask` before `compute_loss()`.

## Split Strategy

The baseline split is 80/10/10.

`SPLIT_STRATEGY = "stratified"` stratifies by the first active non-`unclassified` topic in each row. This is a practical multi-label baseline that keeps rare topics represented across train, validation, and test splits.

The notebook also checks exact `text` overlap across splits and raises an error if leakage is detected.

A stricter group split is deferred because the fused wide CSV does not yet expose one stable service/company key. A future version can derive this from `native_annotations` once the grouping rule is verified.

## Metrics

Validation and test metrics are mask-aware:

- Masked macro F1: per-topic F1 over observed positions only, then averaged.
- Masked micro F1: global binary F1 over all observed topic positions.
- Masked weighted F1: per-topic F1 weighted by positive support.
- Masked subset accuracy: row-level exact match over observed positions.
- Predicted positive rate: share of observed positions predicted positive.
- Per-source metrics: separate masked metrics for rows containing `claudette`, `tos_dr`, or `100_tos`.

The primary checkpoint metric is `eval_macro_f1`.

## Hyperparameters

The first v2 run intentionally keeps the prior baseline:

- Model: `nlpaueb/legal-bert-base-uncased`
- Max length: `256`
- Learning rate: `3e-5`
- Batch size: `8`
- Max epochs: `20`
- Early stopping patience: `3`
- Weight decay: `0.01`
- Warmup ratio: `0.06`
- Decision threshold: `0.50`

Future experiments can try batch size `16` on CUDA, learning rate `2e-5`, threshold calibration, or a score-prediction head.

## Outputs

Runtime outputs are saved under:

- `saved_models/lawgic_classifier_legal-bert_v2/`

Expected artifacts:

- Fine-tuned model weights and config.
- Tokenizer files.
- `test_metrics.json`.
- `classification_thresholds.json`.
- `training_metadata.json`.
- `lawgic_topics_44.json`.
- `lawgic_topics_original_45.json`.
- Checkpoints under `saved_models/lawgic_classifier_legal-bert_v2/checkpoints/`.

## Re-Run Steps

1. Open `notebooks/model_finetuning/legal_bert_finetuning_lawgic_taxonomy.ipynb`.
2. Run sections through data validation and EDA first.
3. Confirm row counts, topic coverage, and mask coverage look plausible.
4. Run training on CUDA when possible; MPS works but will be slower and stays full precision.
5. Run final test evaluation and model saving.
6. Inspect `test_metrics.json` and per-topic report.

## Loading the Saved Model

```python
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = Path("saved_models/lawgic_classifier_legal-bert_v2")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

text = "We may terminate your account at any time for any reason."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    probabilities = torch.sigmoid(model(**inputs).logits)[0]

top_indices = probabilities.argsort(descending=True)[:5]
for index in top_indices:
    label = model.config.id2label[int(index)]
    print(label, float(probabilities[index]))
```

## Known Limitations

- No score head yet; the notebook trains topic presence only.
- The fusion summary reports 60 score-conflict entries; the wide CSV row flag currently marks 47 unique wide rows. They are kept for topic training because conflict concerns scores, not topic presence.
- Group split is not implemented until a stable service/company key is added or derived.
- `unclassified` predictions are unavailable by design.
- Thresholds are fixed at `0.50`; validation-based calibration is a follow-up experiment.

## Dual-Machine Notes

CUDA uses fp16 and should be the preferred training path. Apple Silicon MPS is detected automatically, but mixed precision is disabled for stability. CPU execution is supported for correctness checks but is not practical for full training.
