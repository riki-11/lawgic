# Preparation of the Lawgic Taxonomy Dataset

**Artifact under study:** `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv`
**Producing pipeline:** `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`
**Master taxonomy:** `generated_files/lawgic_taxonomy/lawgic_topics.json`

This report documents how the Lawgic training corpus was assembled from three independently constructed Terms of Service (ToS) annotation datasets, and why the fusion was carried out in the specific way it was. It is intended as a methodological reference for the thesis. The final dataset is the supervision signal for a fine-tuned Legal-BERT model; that model is treated as downstream work and is not covered here except where its requirements dictated a dataset design decision.

The account proceeds in the order the decisions were forced: first the three heterogeneous sources and the taxonomy that had to reconcile them, then the mapping and scoring policies that make heterogeneous labels commensurable, then the failure of the naïve single-head fusion, and finally the architectural pivot to a multi-head dataset that resolves that failure. The central methodological claim is that the multi-head design is not a stylistic preference but the direct consequence of a defect that a single-head fusion cannot express away.

---

## 1. Motivation: Three Formats, One Supervision Target

Consumer-facing ToS assessment has no single canonical labelled corpus. Three public efforts each annotate ToS clauses, but each was built for a different purpose, under a different taxonomy, with a different scoring convention:

| Source | Origin and purpose | Native unit | Native labels | Native scoring |
| --- | --- | --- | --- | --- |
| **CLAUDETTE Cross Market** | Automated detection of potentially unfair contractual clauses (Lippi et al.). | Sentence / clause | 9 unfairness categories (`a`, `ch`, `cr`, `ltd`, `ter`, `use`, `pinc`, `j`, `law`) | 3-level fairness: `1` clearly fair, `2` potentially unfair, `3` clearly unfair |
| **ToS;DR** | Community-curated consumer transparency project ("Terms of Service; Didn't Read"). | Quoted point | ~30 free-text case topics (e.g. `Trackers`, `Governance`, `Guarantee`) | Case classification: `good`, `neutral`, `bad`, `blocker` |
| **100 ToS** | Fine-grained academic annotation of 100 ToS documents. | Annotator comment on a referenced clause | 24 structured eval-variable codes (e.g. `c_law`, `arb`, `ltd_cap`, `as_is`, `sever`) | Per-code integer score in `{-1, 0, 1}` |

The three are complementary rather than redundant. CLAUDETTE is broad and shallow: it covers a small set of high-frequency unfairness categories with high annotation consistency. 100 ToS is narrow and deep: it distinguishes fine subtypes — liability caps versus warranty disclaimers versus indemnity — that CLAUDETTE collapses. ToS;DR is broad on the privacy and governance axis that the other two barely touch — trackers, third parties, government requests, anonymity, logs — but its community topics are coarse and its quotes are sometimes absent.

No single source spans the label space a consumer risk tool must cover. Merging is therefore not an optimisation; it is a precondition for adequate coverage. The engineering problem is that the three cannot be concatenated: they disagree on what the label set is, on what a score means, and on what a missing label implies.

The prevailing literature on ToS clause classification trains **single-head** models — one classifier over one dataset's native taxonomy (CLAUDETTE's unfairness categories, or ToS;DR's topics). This inherits each dataset's coverage ceiling and cannot express the two-part judgment a consumer needs: *which* contractual mechanisms a clause invokes, and *how harmful* the clause's treatment of those mechanisms is. Lawgic instead fuses all three under one reconciled taxonomy and trains a **multi-head** model that predicts topic presence and consumer harm jointly. The dataset described here is what makes that possible.

---

## 2. The Master Taxonomy

The reconciliation layer is `lawgic_topics.json`: **45 topic IDs** (44 substantive topics plus an `unclassified` fallback) organised under **12 parent topics**:

`dispute_resolution`, `limitation_of_remedies`, `unilateral_modification`, `enforcement_actions`, `consent_contract_formation`, `user_content_ip`, `assignment_corporate_changes`, `privacy_data_governance`, `user_rights_consumer_protection`, `interpretation_legal_structure`, `service_governance`, `unclassified`.

Each topic carries an `id`, `name`, `parent_topic`, `description`, a `source_mappings` block recording which native labels from each source point at it, and a `scores` block giving a three-point harm rubric (`-1` bad, `0` neutral, `1` good) with a written legal explanation per level. The rubric is anchored in consumer-protection reasoning: for `choice_of_law`, for instance, `-1` is assigned when a law other than that of the user's habitual residence governs, `0` when foreign law applies only to businesses or mandatory consumer protections are preserved, and `1` when no adverse choice-of-law clause is imposed.

The taxonomy was designed to be **finer than any single source**. It splits CLAUDETTE's single `ltd` category into `limitation_of_liability`, `liability_cap`, `warranty_disclaimer`, `indemnification`, and `limitation_period`; it splits generic "changes" into `contract_changes`, `service_changes`, `price_changes`, `notice_of_changes`, and `user_participation_in_changes`. This granularity is what lets the fused corpus carry 100 ToS's fine distinctions without discarding CLAUDETTE's coarse ones — but it also creates the coarse-to-fine mapping problem addressed in §3.

Two invariants govern the taxonomy's use. First, **the taxonomy is closed**: the fusion pipeline never invents a topic ID. Every mapped label must resolve to one of the 45 IDs, and the notebook asserts this at load time (duplicate-ID check, count-equals-45 check, and validation that every hand-written mapping points at an existing ID). Second, `source_mappings` is treated as necessary but **not sufficient** — the reason for the explicit override layer in §3.

---

## 3. Mapping Heterogeneous Labels: The Coarse-to-Fine Problem

`source_mappings` records that, e.g., CLAUDETTE `ltd` is *related to* the limitation-of-remedies family. A naïve fusion would activate every Lawgic topic listed under a native label. This manufactures false positives: because CLAUDETTE has one broad `ltd` label and Lawgic has four limitation-remedy children, blindly fanning `ltd` out would teach the model that *every* generic limitation-of-liability clause is simultaneously a liability cap, a warranty disclaimer, and an indemnity clause. That is factually wrong for most clauses and injects systematic label noise.

The pipeline therefore adds an **explicit resolution layer** on top of `source_mappings`. The governing principle is *map a coarse native label only to the narrowest Lawgic topic it reliably entails*, and let a finer source supply the finer heads.

### 3.1 CLAUDETTE

| Native label | Lawgic topic(s) | Rationale |
| --- | --- | --- |
| `law` | `choice_of_law` | Direct concept match. |
| `j` | `choice_of_forum` | Direct concept match. |
| `a` | `mandatory_arbitration`, `class_action_waiver` | CLAUDETTE's arbitration category jointly covers both; both heads receive the same score. |
| `ltd` | `limitation_of_liability` **only** | Too coarse to safely assert cap, warranty, or indemnity subtypes. |
| `ch` | `contract_changes` **only** | Broad unilateral-change label; 100 ToS supplies the change subtypes. |
| `ter` | `account_termination` | Closest fit to unilateral termination. |
| `cr` | `content_removal` | Denotes removal/censorship, not all content rules. |
| `use` | `contract_by_use` | Direct concept match. |
| `pinc` | `privacy_incorporation` | Direct concept match. |

`a` is the sole deliberate one-to-many expansion, and it is justified by the native semantics: the CLAUDETTE category explicitly groups arbitration and class-action waivers. `ltd` and `ch` are the conservative one-to-one collapses that prevent subtype fabrication.

### 3.2 ToS;DR

ToS;DR mappings default to `source_mappings`, with explicit overrides for broad topics that would otherwise fan out:

| Native topic | Lawgic topic(s) | Rationale |
| --- | --- | --- |
| `Jurisdiction and governing laws` | `choice_of_law`, `choice_of_forum` | ToS;DR combines law and forum; both are reliably entailed. |
| `Dispute Resolution` | `mandatory_arbitration`, `class_action_waiver` | Broad dispute topic covers both restrictions. |
| `Guarantee` | `warranty_disclaimer` | Narrowest stable limitation-remedy fit. |
| `Waivers` | `indemnification` | Narrowest stable fit for waiver / hold-harmless language. |
| `Changes` | `contract_changes` | Broad; dedicated ToS;DR topics handle notice and user involvement. |

### 3.3 100 ToS

100 ToS codes map **directly**, because the source is already fine-grained and rubric-scored — it is the intended supplier of narrow heads (`liability_cap`, `warranty_disclaimer`, `severability`, `interpretation_clause`). Only two normalisations are applied: spelling/case aliases (`ip → IP`, `tran → transfer`). Where the taxonomy itself maps a code to more than one topic (`IP → copyright_license, ownership`), that expansion is honoured because it is taxonomy-sanctioned rather than invented at fusion time.

The net effect: **coarse sources supply representative heads; the fine source supplies subtype heads; no source is allowed to assert a distinction it does not actually annotate.**

---

## 4. Score Harmonisation

The three native scoring schemes are mapped onto the common `{-1, 0, 1}` harm scale:

- **CLAUDETTE** — the pre-computed `mapped_score` is used (fairness level `1→1`, `2→0`, `3→-1`). The binary `binary_label` is deliberately ignored for harm, because it discards the neutral/severity distinction that the fairness level preserves.
- **ToS;DR** — `good→1`, `neutral→0`, `bad→-1`, and **`blocker→-1`**. `blocker` is not a rubric grade but a severity flag marking a severe obstacle to user rights; it is treated as harmful and tagged `parse_status="blocker_as_bad"` so the policy is auditable rather than silent.
- **100 ToS** — the integer score parsed from the annotator comment is used directly.

Every harmonisation choice is recorded in `lawgic_fusion_summary.json` and in per-row provenance, so no score transformation is opaque after the fact.

---

## 5. Long-Format Construction: The Audit Layer

All three parsers emit a single canonical **long-record** schema — one row per `(text, source annotation, Lawgic topic, score)`. The long table is the source of truth; the model-facing wide table is derived from it and nothing else.

Key columns: `text`, `normalized_text`, `source_dataset`, `source_id`, provenance (`service_name`, `platform`, `company`), `lawgic_topic_id`, `topic_index`, `mapped_score`, `presence_label` (always `1.0` for a parsed positive), `native_label`, `native_tag`, `native_score`, `parse_status`, `mapping_rule`, and `metadata_json`.

The defining semantics of a long row:

> A row exists **only** where a source explicitly annotated that text with that topic. Absence of a row is not negative evidence.

This is the invariant that later governs masking, and it is why the long table is deliberately not deduplicated across sources: multiple rows for the same text/topic represent multiple sources agreeing, which is signal, not redundancy.

### 5.1 Conservative parsing and full diagnostics

100 ToS is the messiest source: its `comment` field mixes structured `code score` annotations with documentary notes, unclear/legal-savings fragments, unknown custom codes, and free-text legal reasoning. The parser accepts **only** segments matching `code score` with `score ∈ {-1, 0, 1}` and a code validated against `100_tos_eval_variables.json`; everything else is excluded. Semicolon- and comma-separated comments are split so multi-annotations (`acc_sus 0; acc_del 0`) are preserved. Rows with a valid comment but empty `referenced_text` are skipped, because empty model input cannot be a training example.

Nothing is dropped silently. Every excluded segment is written to a diagnostics table with a typed reason. The exclusion profile:

| 100 ToS exclusion reason | Count |
| --- | --- |
| `unknown_code_or_free_text` | 912 |
| `unclear_or_legal_savings_fragment` | 594 |
| `documentary_note` | 291 |
| `known_code_without_valid_score` | 12 |
| `empty_referenced_text` | 1 |

ToS;DR contributes an analogous diagnostic: **1,464** rows carry a topic and classification but no quote text, and are logged (`empty_point_quote_text`) rather than admitted. This conservatism is a stated design stance: *it is better to withhold an ambiguous row than to manufacture a noisy label.*

### 5.2 Long-table statistics

| Quantity | Value |
| --- | --- |
| Total long rows | **50,086** |
| — from ToS;DR | 44,317 |
| — from CLAUDETTE | 3,721 |
| — from 100 ToS | 2,048 |
| Score distribution | `-1`: 13,410 · `0`: 24,282 · `1`: 12,394 |
| Unique normalised texts | 26,479 |

---

## 6. Text Normalisation and Grouping

Cross-source fusion requires a join key on clause text. Normalisation is deliberately shallow: convert to string, apply Unicode **NFKC**, collapse whitespace runs to a single space, strip ends. NFKC folds compatibility glyphs and ligatures introduced by different export pipelines; whitespace collapse makes embedded newlines comparable across CSVs. The pipeline does **not** lowercase, strip punctuation, stem, or fuzzy-match — exact normalised text is a safer v1 key than aggressive semantic normalisation, which would silently merge distinct clauses.

The normalised text is the grouping key for the wide table; the original text is retained for display and audit. The wide table groups **globally** by normalised text, because the model's input is the clause string itself and identical strings must receive one merged supervision vector. 100 ToS's company-level provenance is preserved in long-row metadata but does not fragment the global grouping.

---

## 7. Conflict Detection

Merging independent annotators produces disagreement. A **conflict** is defined precisely:

> the same `normalized_text` and same `lawgic_topic_id` receiving more than one distinct `mapped_score`.

The pipeline never collapses conflicts silently. It emits a review table (`lawgic_combined_conflicts.csv`, **60** flagged text/topic pairs) carrying the display text, the conflicting scores, the contributing datasets, and the full native-annotation audit trail. How conflicts are *resolved* differs between the original and the multi-head wide format, and that difference is part of the architectural pivot in §9.

---

## 8. The Single-Head Failure: The Positive-Only Trap

The first wide-format builder (`build_wide_df`) produced, per normalised text, three length-45 vectors: `labels_presence`, `mask`, and `scores`. Its masking rule was the obvious one: set `mask[c] = 1` exactly where topic `c` was annotated for that text — i.e. exactly where `labels_presence[c] = 1`.

This rule is **degenerate**. By construction, the supervised set (`mask = 1`) is identical to the positive set (`label = 1`). There is no cell that says "a source evaluated this topic and found it absent." A masked BCE loss trained on this data sees only positives:

| Metric | Value |
| --- | --- |
| Supervised cells with label = 1 | 49,154 |
| Supervised cells with label = 0 | **0** |
| Positive ratio among supervised cells | **1.0000** |

A classifier minimises this loss trivially by predicting every topic present, yielding a **validation macro-F1 of 1.00** that reflects a broken objective, not a competent model. No hyperparameter, regulariser, or threshold repairs a dataset containing zero negatives. The defect is in the supervision, not the optimiser.

The root cause is conceptual, not a coding slip. The mask faithfully encoded "this source annotated this topic here," but that set coincides exactly with "this topic is present here." The single-head fusion had **no mechanism to represent a supervised negative** — an observed absence — and therefore could not train a discriminative classifier at all.

---

## 9. The Multi-Head Pivot: Manufacturing Honest Negatives

The fix is the reason the dataset — and the model it feeds — became multi-head. It rests on a distinction the single-head mask could not draw: the difference between *this text was not annotated for topic c* and *a source that covers topic c annotated this text and did not flag c*. The first is genuinely unknown; the second is an observed negative. Recovering the second requires knowing **what each source is capable of annotating**.

### 9.1 Source-aware masking

The taxonomy already encodes each source's coverage in `source_mappings`. The corrected builder computes, per source, a coverage vector over the 45 topics and applies source-specific masking:

| Source | Mask rule | Coverage |
| --- | --- | --- |
| **ToS;DR** | `mask[c] = 1` for **all** topics in ToS;DR's taxonomy coverage | 42 topics |
| **100 ToS** | `mask[c] = 1` for **all** topics in 100 ToS's taxonomy coverage | 30 topics |
| **CLAUDETTE** | `mask[c] = 1` **only** for the directly mapped topic(s) of the row's native label | 1–2 topics per annotation, 1–4 per clause |

The reasoning is asymmetric by design. ToS;DR and 100 ToS have broad enough taxonomies that a topic within their coverage, left unannotated on a clause they *did* annotate, is credible evidence of absence — a supervised negative. CLAUDETTE annotates only 9 native labels, which the resolution layer in §3.1 collapses onto 10 topics; a missing CLAUDETTE annotation tells us nothing about the 34 topics it never evaluates. Applying broad masking to CLAUDETTE would **manufacture false negatives**, the mirror of the false positives §3 avoided. CLAUDETTE therefore keeps a narrow mask (Option A). When multiple sources annotate the same text, the final mask is their element-wise **union**, so supervision from any capable source is retained.

> **Correction, 31 July 2026.** Earlier revisions of this table reported 37 and 26. Those
> figures were never measured; the only committed version of `lawgic_topics.json` yields
> 42 and 30, and the built corpus confirms it (mask widths of exactly 42 and 30 on
> single-source rows). The verifying script is `build_source_coverage_mask()` in cell 20
> of `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`, whose printed output had been
> cleared before commit.
>
> **The 42 and 30 figures describe the mask, not the label space.** The mask is built from
> raw `source_mappings`, while ToS;DR labels are built from `TOSDR_TOPIC_OVERRIDES`, which
> is narrower. ToS;DR can only produce positives on **38** of the 42 topics its mask
> supervises. The four unreachable topics (`limitation_of_liability`, `liability_cap`,
> `service_changes`, `price_changes`) therefore receive a supervised negative on every
> ToS;DR row and can never receive a ToS;DR positive. See
> `docs/lawgic_coverage_discrepancy_report.md` for the counts and the remediation plan.

This single change is what breaks the trap: the mask matrix now contains cells that are supervised (`mask = 1`) yet negative (`label = 0`), giving the classifier the contrast it needs. The pipeline asserts `supervised_negative > 0` and fails loudly if the negative-generation ever regresses.

### 9.2 Pessimistic conflict resolution

Because the multi-head harm objective must assign a score to every clause, conflicts can no longer be nullified as in §7. They are resolved by a **pessimistic consumer-protection** policy: the resolved score is the minimum over non-null source scores — if any source judged a topic harmful (`-1`), the fused topic is harmful. This applies at the per-topic level and again at the row level, where the single `harm_score` is the minimum over resolved per-topic scores. The policy is defensible for a *risk* tool: a false "safe" is costlier to a consumer than a false "harmful," so disagreement resolves toward the protective label. Conflicted topics are still tagged in `conflict_topic_ids` for audit.

### 9.3 The dual-head supervision schema

The resulting `lawgic_multihead_wide.csv` carries, per normalised clause, the supervision for two heads on a shared encoder:

- **Topic-presence head** (44 substantive topics, `unclassified` excluded) — `labels_presence` and the source-aware `topic_mask`, trained with masked BCE. The mask makes unknown cells contribute zero loss, so a source's silence on a topic it never covers is never punished.
- **Harm head** (3 classes) — a row-level `harm_score_class` mapping `{-1→0, 0→1, 1→2}`, with `harm_mask` gating rows whose score is unresolvable. Trained with (masked) cross-entropy.

Additional columns (`scores`, `topic_scores`, `active_topic_ids`, `conflict_topic_ids`, `has_score_conflict`, `sources`, `native_annotations`) preserve the full derivation, so any wide row is traceable back to its long-format evidence.

---

## 10. Final Dataset Profile

`lawgic_multihead_wide.csv` — one row per unique normalised clause:

| Quantity | Value |
| --- | --- |
| Wide rows (unique normalised texts) | **26,479** |
| Topic-presence dimensions | 44 (+ `unclassified` retained upstream) |
| Harm classes | 3 |
| Harm distribution — Harmful (`-1`) | 8,311 (31.4%) |
| Harm distribution — Neutral (`0`) | 12,368 (46.7%) |
| Harm distribution — Fair (`+1`) | 5,800 (21.9%) |
| Rows without a harm label | 0 (pessimistic resolution scores every row) |
| Flagged score conflicts (review table) | 60 |
| Excluded 100 ToS segments (logged) | 1,810 |
| Skipped empty-quote ToS;DR rows (logged) | 1,464 |

The topic-presence mask now yields both supervised positives and supervised negatives, and the harm labels form a non-degenerate three-class distribution. The corpus is directly trainable.

---

## 11. Limitations

The fusion is conservative by construction, and its limitations follow from that stance:

1. **No fuzzy text matching.** Near-duplicate clauses differing after NFKC normalisation are not merged; some cross-source agreement is missed rather than risk incorrect merges.
2. **Coarse-label detail loss.** CLAUDETTE `ltd` and `ch` collapse to single representative heads; their subtype detail is supplied only where 100 ToS covers the clause.
3. **Coverage-heuristic negatives.** Source-aware masking treats "within taxonomy coverage but unannotated" as a negative for ToS;DR and 100 ToS. This is a defensible heuristic, not ground truth, and can introduce occasional false negatives where an annotator missed a present topic.
4. **Policy choices carried as data.** `blocker → -1` and pessimistic conflict resolution are protective policy decisions, not neutral facts; both are tagged and auditable so they can be revisited.
5. **Manual conflict review outstanding.** The 60 flagged conflicts are resolved algorithmically (pessimistically) but not yet adjudicated by hand.

Each limitation is a deliberate trade favouring label reliability over label volume — the correct trade for a first fused corpus, where false precision is more damaging than conservative omission.

---

## 12. Methodological Contribution

The preparation pipeline makes two claims relevant to the thesis.

First, **fusion under a finer-than-any-source taxonomy with explicit coarse-to-fine mapping** lets three incompatible datasets contribute their respective strengths — CLAUDETTE's consistency, 100 ToS's granularity, ToS;DR's privacy/governance breadth — without any source asserting distinctions it does not annotate.

Second, and more consequentially, **the multi-head design is forced by the data, not chosen for novelty.** The single-head fusion is provably degenerate: its mask cannot represent an observed absence, so it contains zero negatives and admits only a trivial classifier. Source-aware masking, grounded in each source's declared taxonomy coverage, is what recovers honest negatives — and once negatives and a separate harm signal both exist, a shared encoder with a masked-BCE topic head and a cross-entropy harm head is the natural architecture. This departs from the single-head, single-taxonomy models that dominate the ToS-classification literature, and the departure is motivated by a concrete, reproducible failure of the single-head alternative rather than by preference.
</content>
</invoke>
