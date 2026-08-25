# Duplicate topic columns — resolved

> This document originally held the task brief for this fix. It now records what was actually
> done. Full context for all four fixes from this pass (not just this one) is in
> [`docs/lawgic_taxonomy_revisions.md`](lawgic_taxonomy_revisions.md); this file keeps the
> duplicate-pair-specific detail the brief asked for.

**Status: done.** Corpus rebuilt (`_v2` generation). Retraining not run — out of scope for this
pass.

## The defect

Two topic pairs drew from identical native source labels, so their label and mask columns were
bit-identical across all 26,479 rows:

| Pair | Members | Identical positives |
|---|---|---|
| A | `transfer_of_contract` / `business_transfer` | 346 |
| B | `transparency` / `recommender_system_transparency` | 1,368 |

Confirmed programmatically (`np.array_equal` on label and mask columns) before any change was
made — this is now a permanent guard (below), not a one-time check.

## The decision — deleted, not retained as "unlearnable"

The original brief's default was to keep both topics defined in the schema with the dropped
member marked `"predicted": false`. **The owner's actual decision, made after the brief was
written, was to delete the dropped member from the taxonomy outright.** `business_transfer` and
`recommender_transparency` no longer exist in `lawgic_topics_v2.json`. The taxonomy goes 45 → 43
identifiers; the topic head's prediction space goes 44 → 42.

## Pair B — as the brief already decided

Supervision is 1,341 rows from ToS;DR against 27 from 100 ToS, and the ToS;DR label is general
transparency. Kept `transparency`. Dropped `recommender_transparency`.

## Pair A — determined (the brief left this open)

The brief flagged this as undecided and warned the answer might invert the intuitive choice. It
doesn't. Both members map to the same two native labels:

| Source | Native label | Maps to (v1) |
|---|---|---|
| 100 ToS | `transfer` | both `transfer_of_contract` and `business_transfer` |
| ToS;DR | `Business Transfers` | both `transfer_of_contract` and `business_transfer` |

The 100 ToS variable definition for `transfer`:

> *"Transfer of contractual rights to another subject"* — general category "Unilateral alteration
> clauses", legal ground Annex 1(p) Directive 93/13.

The definition names **contractual rights**, not corporate reorganization. `transfer_of_contract`
is the topic the native label actually names; `business_transfer` is the inferred one. This
matches the entailment rule the thesis already commits to (a native label licenses only the topic
it reliably entails) and matches the intuitive choice — the brief's concern about an inversion did
not materialize.

**Decision: kept `transfer_of_contract`, deleted `business_transfer`.**

## Indemnification — untouched, as instructed

`indemnification` was explicitly out of scope for this fix and was not swept up with the
duplicates. (It was, separately, fixed by a different item in this pass — the `ind` alias recovery
— see the revisions doc. That is a distinct defect: a duplicate column carries no information its
partner doesn't already carry; indemnification's zero-positives problem was a parser bug losing
real annotations, not redundant supervision.)

## What was actually done, against the brief's phases

- **Phase 0 (diagnosis):** done, folded into implementation rather than written to a separate
  `duplicate_topic_columns_diagnosis.md` — the pair confirmation and Pair A determination above
  cover it.
- **Phase 1 (eval-time exclusion on the existing checkpoint):** **not run.** The brief's expected
  numbers (macro F1 ≈0.766, support 4,759) were never measured; this pass rebuilt the corpus
  directly rather than validating against the live v3 checkpoint first. If those numbers are
  still wanted for the manuscript, they're a follow-up.
- **Phase 2 (taxonomy and corpus rebuild):** done, with the deletion decision above (not the
  `"predicted": false` field the brief proposed). The permanent duplicate-column guard the brief
  asked for is in place (notebook cell 22 of `lawgic_taxonomy.ipynb`) — it asserts after every
  build that no two label/mask column pairs are element-wise identical, and raises naming the
  offending pair if any are found. The label-ordering file the brief asked for
  (`lawgic_topic_order_v2.json`, 42 entries) is written alongside the corpus.
- **The split hazard:** the brief's requirement — that the persisted split must not change — was
  superseded by a separate decision in this pass (see the revisions doc, §"Split rebuild"): the
  `ind` alias fix added 75 new clauses, so the split was rebuilt from scratch rather than held
  fixed. This means the split hazard this brief worried about (duplicate removal shifting a
  clause's primary-stratify topic) is moot — the split changed anyway, for an unrelated reason.
- **Phase 3 (retraining prep):** **not done.** No v3-successor checkpoint exists. Out of scope for
  this pass per explicit instruction.

## Actual measured numbers

| Quantity | v1 | v2 (measured) |
|---|---|---|
| Taxonomy topics | 45 | 43 |
| Predicted topics | 44 | 42 |
| `transfer_of_contract` positives | 346 | 346 (unchanged) |
| `transparency` positives | 1,368 | 1,368 (unchanged) |
| `business_transfer`, `recommender_transparency` | 346 / 1,368 | removed |

No macro-F1 number is reported here — that requires the retraining pass this document explicitly
did not run.

## Manuscript edits (unchanged from the original brief, still to apply)

Same locations as originally identified, values updated:

- `chapter_4.tex`, redundancy-disclosure paragraph (Section 4.2, "The mapping tables carry one
  redundancy…"): rewrite to state the duplication was found and **resolved by deleting** the two
  dropped topics from the taxonomy (not by excluding them from prediction while keeping them
  defined), name which member of each pair was dropped and on what rule (Pair A evidence above),
  and that the correction lowers reported topic macro-F1 once retrained.
- `chapter_4.tex`, `tab:corpusprofile`: topic-presence dimensions 44 → 42; taxonomy topic count
  45 → 43 wherever cited.
- `chapter_4.tex`, Section 4.3 task formulation: topic vector and mask in $\{0,1\}^{42}$.
- `chapter_4.tex`, dual-head architecture: topic head is 768 → 42 (once retrained; currently the
  serving checkpoint is still 768 → 44).
- `chapter_4.tex`, Section 4.4 evaluation: single reported macro over 42, not a 44-vs-42
  comparison.
- `appendix_C.tex`: per-topic support and results tables lose one row per pair; macro/weighted
  rows recomputed once retrained.
- `fig:corpussupport` and `figures/corpus_topic_support.png`: one bar per pair removed.
- Grep `.tex` files for `44`/`45` and check each occurrence by hand (some are page numbers).

## Related

- [`lawgic_taxonomy_revisions.md`](lawgic_taxonomy_revisions.md) — full record of all four fixes
  from this pass, including this one.
- [[Lawgic - Dataset Duplicate Topics Issue]] — the original task brief (Obsidian vault), superseded
  by this document for what was actually implemented.
