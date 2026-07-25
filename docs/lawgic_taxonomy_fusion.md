# Lawgic Taxonomy and Dataset Fusion

This document explains how the Lawgic taxonomy and fused training dataset are intended to work. It is written as a companion to `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`.

The notebook contains the executable pipeline. This document explains the reasoning behind the pipeline so the decisions remain understandable later.

## Goal

Lawgic combines three Terms of Service annotation sources:

- 100 ToS
- ToS;DR
- CLAUDETTE Cross Market

Each source has a different taxonomy, annotation style, and scoring scheme. The project needs one shared Lawgic topic taxonomy and one combined dataset for model training.

The hard part is not concatenating CSV files. The hard part is avoiding bad labels. If one source never annotated privacy topics, those missing privacy labels must not become negative labels. The fusion pipeline therefore builds a dataset with both labels and masks.

## Master Taxonomy

The master taxonomy lives at:

`generated_files/lawgic_taxonomy/lawgic_topics.json`

It defines 45 Lawgic topic IDs grouped under 12 parent topics:

- `dispute_resolution`
- `limitation_of_remedies`
- `unilateral_modification`
- `enforcement_actions`
- `consent_contract_formation`
- `user_content_ip`
- `assignment_corporate_changes`
- `privacy_data_governance`
- `user_rights_consumer_protection`
- `interpretation_legal_structure`
- `service_governance`
- `unclassified`

Each topic has:

- `id`
- `name`
- `parent_topic`
- `description`
- `source_mappings`
- `scores`
- `examples`

The important field for fusion is `source_mappings`. It records which native source labels point at a Lawgic topic. For example, `choice_of_law` maps to:

- 100 ToS `c_law`
- ToS;DR `Jurisdiction and governing laws`
- CLAUDETTE `law`

The mapping file is the source of valid topic IDs. The fusion notebook never invents new topic IDs.

## Why Source Mappings Are Not Enough

Some native labels are broad. A blind mapping from source label to every Lawgic topic in `source_mappings` would create false positives.

Example: CLAUDETTE has one broad `ltd` label for limitation of liability. In Lawgic, the limitation-remedy family has narrower topics:

- `limitation_of_liability`
- `liability_cap`
- `warranty_disclaimer`
- `indemnification`

If every CLAUDETTE `ltd` row activated all four topics, the model would learn that every generic limitation-of-liability clause is also a liability cap, warranty disclaimer, and indemnity clause. That is noisy.

The fusion notebook therefore adds explicit mapping rules on top of `source_mappings`.

## Explicit Coarse-to-Fine Mapping Rules

### CLAUDETTE

CLAUDETTE rows are already parsed in:

`generated_files/claudette_cross_market/claudette_cross_market_clauses.csv`

The fusion notebook uses `quoted_text`, `claudette_topic_code`, `claudette_tag`, and `mapped_score`.

It does not use `binary_label` for harm scoring. CLAUDETTE fairness levels already map to Lawgic-style scores:

- `1` clearly fair -> `1`
- `2` potentially unfair -> `0`
- `3` clearly unfair -> `-1`

Fusion rules:

| CLAUDETTE label | Lawgic topic IDs | Reason |
| --- | --- | --- |
| `law` | `choice_of_law` | Direct concept match. |
| `j` | `choice_of_forum` | Direct concept match. |
| `a` | `mandatory_arbitration`, `class_action_waiver` | CLAUDETTE groups arbitration and class-action waivers together. Both heads should receive the same score. |
| `ltd` | `limitation_of_liability` | Broad limitation label. Do not invent subtype labels for caps, warranty disclaimers, or indemnity. |
| `ch` | `contract_changes` | Broad unilateral-change label. Do not activate all change subtypes. |
| `ter` | `account_termination` | Closest match to unilateral termination. |
| `cr` | `content_removal` | Content removal/censorship, not all content rules. |
| `use` | `contract_by_use` | Direct concept match. |
| `pinc` | `privacy_incorporation` | Direct concept match. |

The main conservative choices are `ltd` and `ch`. They stay on one representative head instead of being fanned out to every child topic.

### ToS;DR

ToS;DR rows come from:

`generated_files/tos_dr/tos_dr_points.csv`

The fusion notebook uses:

- `point_quote_text`
- `point_id`
- `service_name`
- `cases.case_topic`
- `cases.case_classification`
- `score_rubric`

Classification scores map as:

| ToS;DR classification | Lawgic score |
| --- | --- |
| `bad` | `-1` |
| `blocker` | `-1` |
| `neutral` | `0` |
| `good` | `1` |

`blocker` is mapped to `-1` because it represents a severe obstacle in the ToS;DR model. The notebook marks these rows with `parse_status="blocker_as_bad"` so they can be audited.

Some ToS;DR rows have a topic and classification but no quote text. These rows cannot become training examples because the model would receive empty input. The notebook skips them and logs them to:

`generated_files/lawgic_taxonomy/lawgic_tosdr_parse_diagnostics.csv`

Most ToS;DR topic mappings come from `lawgic_topics.json`. Some broad topics are overridden:

| ToS;DR topic | Lawgic topic IDs | Reason |
| --- | --- | --- |
| `Jurisdiction and governing laws` | `choice_of_law`, `choice_of_forum` | ToS;DR combines law and forum. |
| `Dispute Resolution` | `mandatory_arbitration`, `class_action_waiver` | Broad dispute topic usually covers both arbitration and collective-action restrictions. |
| `Guarantee` | `warranty_disclaimer` | Best narrow fit. Avoid activating every limitation-remedy head. |
| `Waivers` | `indemnification` | Best narrow fit for waiver/hold-harmless language. |
| `Changes` | `contract_changes` | Broad changes topic. More specific ToS;DR topics handle notice and user involvement. |

This keeps the ToS;DR signal useful without over-labeling fine Lawgic heads.

### 100 ToS

100 ToS is the most fine-grained source and the hardest to parse.

Preferred input:

`generated_files/100_tos/annotated_tos_comments.csv`

Legacy/fallback input:

`generated_files/100_tos/cleaned_tos_comments.csv`

The notebook prefers `annotated_tos_comments.csv` because it preserves semicolon-separated multi-annotations like:

- `acc_sus 0; acc_del 0`
- `cnt_del 0; cnt_modr 0`
- `ip -1; shad_ip`

The 100 ToS eval-variable definitions live at:

`generated_files/100_tos/100_tos_eval_variables.json`

Only structured `code score` annotations are used for training. The score must be one of:

- `-1`
- `0`
- `1`

Accepted examples:

- `ltd 0`
- `acc_del -1`
- `arb -1`
- `acc_sus 0; acc_del 0`

Alias handling:

| Raw code | Canonical code |
| --- | --- |
| `ip` | `IP` |
| `tran` | `transfer` |

Excluded examples:

| Pattern | Reason |
| --- | --- |
| `docu 1-3` | Documentary note, not an eval-variable label. |
| `uncle`, `uncle 2` | Unclear/legal-savings fragment, not a topic label. |
| `virtu`, `pato`, `shad_ip` | Unknown custom code outside the 24 eval variables. |
| `against Art. 16...` | Free-text explanatory tail, not a structured score. |

Every excluded segment is logged to:

`generated_files/lawgic_taxonomy/lawgic_100_tos_parse_diagnostics.csv`

Some 100 ToS rows have a valid `code score` comment but an empty `referenced_text`. Those rows cannot become training examples because the model would receive empty input. The notebook skips them with diagnostic reason `empty_referenced_text`.

This is important. The parser is conservative, but it should never silently drop hard-to-parse information.

## Text Normalization

Fusion groups examples by normalized text. The notebook keeps both original text and normalized text.

Normalization steps:

1. Convert value to string.
2. Apply Unicode NFKC normalization.
3. Collapse whitespace runs to one space.
4. Strip leading and trailing whitespace.

The normalized text is the grouping key. The original text remains available for display and audit.

Why NFKC matters:

- It normalizes compatibility glyphs and ligatures.
- It reduces superficial source differences.
- It helps align clauses from `annotated_tos_comments.csv` and other generated CSVs.

The notebook does not perform aggressive semantic normalization. It does not lowercase, remove punctuation, stem words, or fuzzy match. Exact normalized text is safer for v1.

## Grouping Strategy

The long table keeps every explicit source annotation. The wide table groups by global `normalized_text`.

100 ToS has an extra contextual concern: the same referenced text can appear under multiple companies. The parser preserves `company` in metadata and source IDs. Cross-source fusion still groups globally by normalized text because the model input is the clause text itself.

This means:

- Long format keeps source and company provenance.
- Wide format merges all evidence for the same normalized text.
- Score conflicts are flagged instead of hidden.

## Long Format

Output:

`generated_files/lawgic_taxonomy/lawgic_combined_long.csv`

One row means:

> One source explicitly annotated this text with this Lawgic topic and score.

Core columns:

| Column | Meaning |
| --- | --- |
| `text` | Original clause text. |
| `normalized_text` | Normalized grouping key. |
| `source_dataset` | `claudette`, `tos_dr`, or `100_tos`. |
| `source_id` | Source-specific stable ID. |
| `service_name` | ToS;DR service, where available. |
| `platform` | CLAUDETTE platform, where available. |
| `company` | 100 ToS company, where available. |
| `lawgic_topic_id` | One of the 45 Lawgic topic IDs. |
| `topic_index` | Index of topic in taxonomy order. |
| `mapped_score` | Harm score in `{-1, 0, 1}`. |
| `presence_label` | Always `1.0` for parsed topic rows. |
| `native_label` | Native source label, such as `ltd`, `Dispute Resolution`, or `acc_del`. |
| `native_tag` | More specific native tag/comment where available. |
| `native_score` | Source-native score or classification. |
| `parse_status` | `parsed`, `blocker_as_bad`, etc. |
| `mapping_rule` | Which explicit mapping rule produced the row. |
| `metadata_json` | Source-specific audit metadata. |

The long table is the source of truth for debugging.

## Wide Format

Output:

`generated_files/lawgic_taxonomy/lawgic_combined_wide.csv`

One row means:

> One normalized clause text with vectors over all 45 Lawgic topics.

Core columns:

| Column | Meaning |
| --- | --- |
| `text` | Display text, chosen from the longest original version. |
| `normalized_text` | Grouping key. |
| `sources` | JSON list of contributing datasets. |
| `labels_presence` | JSON list of 45 floats. |
| `mask` | JSON list of 45 floats. |
| `scores` | JSON list of 45 scores or nulls. |
| `topic_scores` | JSON dict of active topic ID to score/null. |
| `active_topic_ids` | JSON list of active Lawgic topics. |
| `conflict_topic_ids` | JSON list of topics with score conflicts. |
| `has_score_conflict` | Boolean conflict flag. |
| `native_annotations` | JSON audit trail from long rows. |

## Masked Multi-Label Training

The wide table contains both `labels_presence` and `mask`.

For each topic dimension `c`:

- `mask[c] = 1` means the source explicitly annotated that topic for this text.
- `mask[c] = 0` means unknown.
- `labels_presence[c] = 1` means the topic is present.
- `labels_presence[c] = 0` with `mask[c] = 0` must not be read as negative.

Masked BCE:

```python
loss_per_topic = BCEWithLogitsLoss(reduction="none")(logits, labels_presence)
masked_loss = (loss_per_topic * mask).sum() / mask.sum().clamp_min(1.0)
```

The mask avoids the missing-label trap.

Example:

CLAUDETTE does not annotate `personal_data`. If a CLAUDETTE clause contains data-sharing language, standard BCE would treat `personal_data=0` as negative and punish the model for predicting it. With masked BCE, `mask[personal_data]=0`, so that dimension contributes no loss.

## Presence vs Score

Lawgic ultimately cares about consumer-harm scores. The fused dataset carries those scores from day one.

However, the recommended first training phase is topic presence:

- Train topic presence with masked BCE.
- Preserve score vectors and `topic_scores`.
- Add a score head later after the fusion outputs and conflict review are stable.

Reason:

- The existing finetuning notebook is binary multi-label.
- Masked presence training is easier to debug.
- Score prediction is sparse and needs a separate loss design.

Suggested later model:

1. Shared Legal-BERT encoder.
2. Topic-presence head trained with masked BCE.
3. Harm-score head trained only where `mask=1` and score is non-conflicting.

For the score head, use 3-class cross entropy or an ordinal loss rather than plain BCE.

## Conflict Detection

Output:

`generated_files/lawgic_taxonomy/lawgic_combined_conflicts.csv`

A conflict is:

> Same `normalized_text`, same `lawgic_topic_id`, multiple `mapped_score` values.

The notebook keeps all long rows. In wide format:

- `mask` remains `1`.
- `labels_presence` remains `1`.
- `scores[index]` becomes `null`.
- `topic_scores[topic_id]` becomes `null`.
- `conflict_topic_ids` includes the topic.

This preserves presence training while preventing bad score supervision.

Conflicts should be reviewed before final score-head training.

## Output Summary

The notebook also writes:

`generated_files/lawgic_taxonomy/lawgic_fusion_summary.json`

It records:

- input paths
- output paths
- topic IDs
- mapping notes
- score policies
- row counts
- source counts
- conflict counts
- 100 ToS parse diagnostic counts
- ToS;DR parse diagnostic counts

This gives a quick machine-readable audit trail for each run.

## Known Limitations

The fusion is conservative by design.

Known limitations:

- No fuzzy matching across near-duplicate text.
- No manual conflict resolution yet.
- CLAUDETTE `ltd` and `ch` lose subtype detail because the native labels are broad.
- ToS;DR broad topics are mapped to stable representative heads, not every possible fine topic.
- 100 ToS free-text comments are excluded instead of interpreted.
- `blocker` is treated as `-1`, but this remains a policy choice to revisit.
- The notebook does not update model training code.

These limitations are acceptable for a first reliable fused dataset. The goal is to avoid false precision.

## How To Inspect Results After Running

Start with:

1. `lawgic_fusion_summary.json`
2. `lawgic_100_tos_parse_diagnostics.csv`
3. `lawgic_combined_conflicts.csv`
4. `lawgic_combined_long.csv`
5. `lawgic_combined_wide.csv`

Recommended checks:

- Count rows by `source_dataset`.
- Count rows by `lawgic_topic_id`.
- Inspect excluded 100 ToS segments by `reason`.
- Inspect conflicts before score-head training.
- Pick a few `normalized_text` examples and compare long annotations to wide vectors.

## Design Principle

The central design choice is:

> Keep labels only where a source explicitly provides them. Keep everything else masked.

That one rule prevents the combined dataset from manufacturing negatives across incompatible source taxonomies.
