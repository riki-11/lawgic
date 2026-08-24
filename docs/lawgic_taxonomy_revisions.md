# Lawgic Taxonomy Revisions — 2026-08-24

> What was fixed in the fused corpus and taxonomy following the panel's Recommendation 3
> definition audit, what changed as a result, and what it invalidates. This is the record to copy
> manuscript values from — not a plan, a report of what was actually done.

All four fixes below produced a new corpus generation under the `_v2` suffix
(`generated_files/lawgic_taxonomy/*_v2.*`). The original 44-topic artifacts are untouched on disk
for comparison. **No model was retrained.** This document covers the data pipeline and reporting
only.

---

## 1. What was wrong

**Issue 1 — a missing parser alias silenced 77 annotations.** The 100 ToS annotator wrote `ind 0`
as shorthand for the published variable `indemn` in 77 clause comments. The parser's
`HUNDRED_TOS_CODE_ALIASES` table didn't recognize `ind`, so all 77 were silently discarded and
`indemnification` reported zero positives — not because no source ever discussed indemnification,
but because a lookup table was missing one entry.

**Issue 2 — two topic pairs had bit-identical supervision.** `transfer_of_contract`/
`business_transfer` and `transparency`/`recommender_transparency` each drew from the exact same
native source labels, so their label and mask columns were identical across all 26,479 rows.
Nothing in the training signal had ever distinguished the members of either pair — the topic head
emitted both, always, on every clause.

**Issue 3 — one 100 ToS code was mapped to two topics that don't entail each other.** `price_chg`
appeared in the `source_mappings` of both `price_changes` and `payments`, so every `price_changes`
positive was also forced positive for `payments` (100% one-directional containment).

**Issue 4 — the supervision report credited the wrong source.** `scripts/corpus_report.py`'s
`positive_sources` column credited every source present on a multi-source row for every topic on
that row, rather than only the source whose own annotation actually asserted that topic. This
inflated every source's "topics reached" count on the 109 multi-source rows.

---

## 2. What was changed

### Issue 4 fixed first (independent of the rest; the only fix verifiable against the exact corpus every previously quoted number came from)

`scripts/corpus_report.py`: the per-topic source-composition loop now walks each row's
`native_annotations` (the audit trail already written by the fusion pipeline) instead of the row's
`sources` list, and is gated on the supervision mask. A source is credited for a topic only where
its own native annotation maps to that topic.

**Correction found while implementing:** the number this fix converges on is **37 / 26 / 10**
(ToS;DR / 100 ToS / CLAUDETTE), not the 38/30 the earlier issue brief expected. That 38/30 figure
comes from a *different, deeper* proposed fix in `docs/lawgic_coverage_discrepancy_report.md` — a
rewrite of `build_source_coverage_mask()` that would exclude topics whose raw `source_mappings`
entry is redirected away by an override table before it can ever fire. That is a masking-level
change and stays explicitly out of scope here. This fix answers a narrower question — how many
topics does each source's own native annotation actually reach in the built corpus — and 37/26/10
is confirmed independently against `lawgic_combined_long.csv`
(`groupby("source_dataset")["lawgic_topic_id"].nunique()` returns the identical figures).

### Issue 1 — `ind` alias added, `sugg`/`inter` deliberately excluded

`HUNDRED_TOS_CODE_ALIASES` in `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` (cell 4) gained one
entry: `"ind": "indemn"`. `sugg`→`suggest` and `inter`→`interpret` were considered and **not**
added, by decision: recovering them would mean accepting annotation codes whose abbreviated form
isn't documented in the source publication, even though the codes they expand to (`suggest`,
`interpret`) are among Pałka et al.'s 24 published variables. This is recorded as an explicit
`MAPPING_NOTES` entry (`100_tos:excluded_aliases`) rather than left as a silent gap.

### Issue 2 — duplicate topics deleted, not retained as "unlearnable"

`business_transfer` and `recommender_transparency` were removed outright from
`lawgic_topics_v2.json`. The taxonomy goes 45 → 43 identifiers (42 predicted + `unclassified`).
This differs from the original task brief, which proposed keeping both members in the schema
marked `"predicted": false`; the owner's decision was to delete the dropped members entirely.

**Pair A determination** (the brief left this open): both members map to 100 ToS `transfer` and
ToS;DR `Business Transfers`. The 100 ToS variable definition for `transfer` reads *"Transfer of
contractual rights to another subject"* — it names contractual rights, not corporate
reorganization. `transfer_of_contract` is therefore the topic the native label actually names;
`business_transfer` is the inferred one. The entailment rule and the owner's intuitive choice agree
here; the brief's warning that the answer might invert did not apply.

### Issue 3 — `price_chg` removed from `payments`

The 100 ToS variable definition for `price_chg` is *"Unilateral change of future prices"*, general
category *"Unilateral alteration clauses"* (partially Annex 1(j) Directive 93/13). Lawgic's
`payments` topic is defined as payment obligations, recurring charges, automatic renewal, payment
credentials, cancellation, and refunds. A clause reserving the right to change future prices does
not entail any of those — the mapping was a defect under the entailment rule, not a disclosure
item. `100_tos: ["price_chg"]` was removed from `payments`'s `source_mappings` in
`lawgic_topics_v2.json`; the `tos_dr: ["Payments"]` mapping is untouched.

Containment was confirmed one-directional before the fix: `payments` held 353 long-format rows,
only 50 from `price_chg`; the other 303 came from ToS;DR's `Payments` topic. Removing the mapping
does not empty the topic — **no**, the reverse containment does not hold: 300 of `payments`'s 300
positives are not `price_changes` positives after the fix (they never were).

### Standing regression guard added

Notebook cell 22 now asserts, after every corpus build, that no two label/mask column pairs are
element-wise identical — the exact defect that let Issue 2 go unnoticed. It raises and names the
offending pair rather than passing silently.

### Constants derived from the taxonomy instead of hardcoded

- `scripts/lawgic_eval_core.py`: `TOTAL_TAXONOMY_TOPICS`/`NUM_LAWGIC_TOPICS` are now computed from
  `load_taxonomy()` rather than literals `45`/`44`. **This module was deliberately left pointed at
  the v1 (44-topic) corpus and the v3 checkpoint** — it's the eval harness backing the already
  -reported multiseed and source-heldout results, and repointing it to v2 would shape-mismatch
  every existing evaluation against that 768→44 checkpoint. The fix removes the literal, not the
  v1 binding; a future v2-trained checkpoint gets its own path constants.
- `api/server.py`: `NUM_TOPICS = 44` deleted. The topic count is now read from the serving
  checkpoint's own label map (`lawgic_topics_44.json`, or `lawgic_topic_order.json` for a future
  non-44-topic checkpoint) before the model is constructed, so the `Linear` layer is sized from
  the checkpoint rather than a module constant. The running v3 checkpoint is unaffected.
- A persisted label-order file (`lawgic_topic_order_v2.json`, 42 entries) is written alongside the
  v2 corpus — the contract a future model trained on it must read instead of assuming an order.

### Split and contamination audit rebuilt for v2

The corpus grew by 75 clauses (unevenly distributed — see §3), so the positional `row_id` key in
the persisted v1 split no longer addresses the same clauses. Two new standalone scripts —
`scripts/build_split_v2.py` and `scripts/near_duplicate_split_audit_v2.py` — reuse the existing
pure stratification/audit functions from `lawgic_eval_core.py` and
`near_duplicate_split_audit.py` respectively against the v2 corpus, writing `_v2`-suffixed outputs.
Neither v1 script nor its outputs were touched.

### Explicitly not touched

The entailment rule and override tables (`CLAUDETTE_TOPIC_RULES`, `TOSDR_TOPIC_OVERRIDES`), the
−1/0/+1 score mapping, `build_source_coverage_mask`/`compute_source_aware_mask`, and model training.
No new topics were added. The `a → class_action_waiver` mapping defect (CLAUDETTE's arbitration
label supplies 29% of that topic's positives despite its codebook being silent on class actions)
stays a disclosed limitation, not a code change — that was the owner's explicit instruction.

---

## 3. Before / after

Measured directly from the produced artifacts (`lawgic_fusion_summary{,_v2}.json`,
`per_topic_supervision{,_v2}.csv`, `split_seed42{,_v2}.csv`).

| Quantity | v1 | v2 |
|---|---|---|
| Taxonomy topics (incl. `unclassified`) | 45 | **43** |
| Predicted topics | 44 | **42** |
| Long-format rows | 50,086 | **48,354** |
| Unique clauses | 26,479 | **26,554** (+75) |
| Score conflicts | 60 | 60 (unchanged) |
| Split (train / val / test) | 21,183 / 2,648 / 2,648 | **21,243 / 2,655 / 2,656** |
| 100 ToS `unknown_code_or_free_text` diagnostic rows | 912 | **833** |

Per-topic support, topics whose numbers moved:

| Topic | v1 positives | v2 positives | Note |
|---|---|---|---|
| `indemnification` | 0 | **76** | all from 100 ToS `indemn` (via `ind` alias), all score 0 |
| `payments` | 350 | **300** | now 100% ToS;DR; `price_chg` no longer contributes |
| `price_changes` | 50 | 50 (unchanged) | sole supplier remains 100 ToS `price_chg` |
| `business_transfer` | 346 | *(removed)* | duplicate of `transfer_of_contract`, deleted |
| `recommender_transparency` | 1,368 | *(removed)* | duplicate of `transparency`, deleted |
| `transfer_of_contract` | 346 | 346 (unchanged) | kept per Pair A determination, §2 |
| `transparency` | 1,368 | 1,368 (unchanged) | kept per original Pair B decision |

The 76-vs-77 note: the long-format table gained exactly 77 `indemnification` rows (matching the
77 newly-recovered `ind` clauses), but 2 of those rows share a clause+topic pair with another
`ind` annotation on the same text, collapsing to 1 wide-format positive — so the wide corpus shows
76 supervised positives against 77 long-format annotations. Both numbers are internally consistent
with the pessimistic-resolution / grouping logic already in the pipeline; this is not a defect.

---

## 4. Source attribution correction (Issue 4)

| Source | Old (wrong) count | New (verified) count, v1 corpus | New count, v2 corpus |
|---|---|---|---|
| ToS;DR | 40 | **37** | 35 |
| 100 ToS | 31 | **26** | 24 |
| CLAUDETTE | 24 | **10** | 10 (unchanged) |

The v1→v2 drop for ToS;DR and 100 ToS (37→35, 26→24) is expected and mechanical: both dropped
duplicate topics (`business_transfer`, `recommender_transparency`) were independently reached by
both ToS;DR and 100 ToS (their native labels fan out to both members of a duplicate pair), so
deleting the two topics removes two entries from each source's reach count. CLAUDETTE never reached
either dropped topic, so its count is unaffected.

**Use 37/26/10 (v1) or 35/24/10 (v2) as the verified figures going forward — not 40/31/24, and not
the 38/30 the earlier issue brief predicted (see §2).**

---

## 5. What this invalidates

- **Every previously reported topic-head metric.** The v3 checkpoint is a 768→44 head trained on
  the v1 corpus and v1 split. It cannot be evaluated against the v2 corpus (42 columns) without a
  shape mismatch, and no v2-trained checkpoint exists yet — that's a retraining pass, out of scope
  here.
- **The near-duplicate contamination audit.** The v1 figures (`near_duplicate_audit.json`) were
  computed against the v1 80/10/10 split. The v2 split is a different partition of a different-size
  corpus; `near_duplicate_audit_v2.json` is the current figure and is not comparable clause-for
  -clause to the v1 one, only structurally (same TF-IDF method, same thresholds).
- **The corpus profile table and any manuscript prose citing 44 topics, 26,479 clauses, or the
  40/31/24 source-reach figures.** These need regeneration from the v2 artifacts, not in-place
  editing.
- **Nothing about the v3 checkpoint's current serving behavior in `api/server.py`.** It still loads
  its own 44-entry label map and serves unchanged.

---

## 6. Numbers to transfer into the manuscript

| Value | Manuscript location |
|---|---|
| Taxonomy topics 45 → 43 | `chapter_4.tex`, taxonomy structure prose |
| Predicted topics 44 → 42 | `chapter_4.tex` §4.3 task formulation, dual-head architecture (§ around "768→44") |
| `tab:corpusprofile` topic-presence dimension 44 → 42 | `chapter_4.tex` |
| Unique clauses 26,479 → 26,554 | `chapter_4.tex`, all corpus-size citations |
| Long-format rows 50,086 → 48,354 | `chapter_4.tex`, corpus profile |
| Split 21,183/2,648/2,648 → 21,243/2,655/2,656 | `chapter_4.tex` §4.3, `appendix_C.tex` split table |
| `indemnification`: 0 → 76 positives, moves bucket 3 (unlearnable) → bucket 2 (low-support) | `chapter_4.tex` §9.3-equivalent low-support discussion, `appendix_C.tex` per-topic table |
| `payments`: 350 → 300 positives, containment with `price_changes` resolved to 0% | wherever the containment finding is disclosed |
| `business_transfer`, `recommender_transparency` removed from taxonomy and all tables | redundancy-disclosure paragraph (§4.2, "The mapping tables carry one redundancy…"), `appendix_C.tex` per-topic support and results tables, `fig:corpussupport` |
| Pair A determination: keep `transfer_of_contract`, drop `business_transfer`, on the entailment rule reading of 100 ToS `transfer` | same redundancy-disclosure paragraph |
| ToS;DR reach corrected 40 → 37 (v1) / 35 (v2); 100 ToS 31 → 26 (v1) / 24 (v2); CLAUDETTE 24 → 10 | wherever `positive_sources`/coverage-by-source is cited |
| `ind → indemn` alias recovery reported as a definition-audit correction; `sugg`/`inter` explicitly left unaliased | Section 4.2.2/4.2.3 lost-label accounting |

---

## 7. Verification record

All gates from the implementation plan passed:

- Issue 4 fix verified against the live v1 corpus before any rebuild: 37/26/10 confirmed
  independently via `lawgic_combined_long.csv` groupby.
- Notebook executed top-to-bottom with no errors; duplicate-column guard printed "passed"; harm
  mask sanity check passed (zero unresolved rows); mask sanity check confirms nonzero supervised
  negatives.
- `lawgic_eval_core.py` imports cleanly under `thesis-env` and resolves to (44, 45) — unchanged
  from before, confirming the v1 binding was preserved through the literal-removal refactor.
- `build_split_v2.py` ran with no leakage exception; 21,243/2,655/2,656 rows, sum = 26,554.
- `near_duplicate_split_audit_v2.py` ran to completion, writing comparable-methodology contamination
  figures for the v2 split.
- `corpus_report.py v2` confirms `topics_with_zero_positives` is empty (was `["indemnification"]`),
  `payments` credits only `tos_dr`, and both dropped topics are absent from the 43-topic report.

---

## Related

- [[Lawgic - Data Preprocessing Masterfile]] — needs its own pass to update §2, §4.1, §8, §9, §11
  with the numbers in §3–4 above.
- `docs/lawgic_coverage_discrepancy_report.md` — the 42/30/21 raw taxonomy-coverage table is
  unaffected by this pass (it counts `source_mappings` capacity, not this document's fixes); its
  proposed masking-level rewrite (38/29/21) remains a separate, unimplemented item.
- `docs/lawgic_duplicate_topics_fix.md` — currently a verbatim copy of the original task brief;
  should be replaced with the Pair A determination and decision recorded in §2 above.
- `CHANGES.md` — add one entry per fix, cross-linking here.
