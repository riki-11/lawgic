# Report: Data Preparation Defects in the Lawgic Fusion Pipeline

Language: ASD-STE100 Simplified Technical English.
Created: 31 July 2026. Revised: 31 July 2026 after the CLAUDETTE source-paper audit.
Scope: the taxonomy, the fusion notebook, the raw source corpora, the built corpus, and
the documents.
Audience: the author, and any coding agent that implements the fixes in section 11.

---

## 1. Summary and status

The investigation started with a wrong number in two documents. It found six defects. Two
are fixed. Four remain.

| # | Defect | Severity | Status |
| --- | --- | --- | --- |
| 1 | The topic mask and the topic labels are built from different tables | High | Open |
| 2 | The notebook outputs were cleared, so no number was ever measured | Medium | **Fixed** |
| 3 | Two masking regimes are applied without a complete justification | Medium | Open |
| 4 | Five ToS;DR labels expand to three or four topics each | High | Open |
| 5 | The CLAUDETTE extraction discards 91.5 percent of the source corpus | **Highest** | Open |
| 6 | Three mapping rules contradict the published category definitions | High | Open |

Defect 5 is the largest. It was found last. It changes the plan for defects 3 and 6.

**Nothing in this report invalidates the trained model.** The corpus rebuilds byte-for-byte
from the committed notebook. The defects are limits on what the corpus can teach, not
errors in what it recorded.

---

## 2. Verified reference numbers

Every number below was measured on 31 July 2026. Use these, not the numbers in older
document revisions.

### 2.1 Source coverage

| Source | Method A, raw `source_mappings` | Method B, after the override tables |
| --- | --- | --- |
| CLAUDETTE | 21 | 10 |
| ToS;DR | 42 | 38 |
| 100 ToS | 30 | 30 |

The code uses Method A for ToS;DR and 100 ToS. It uses Method B for CLAUDETTE, and applies
it per row rather than per source.

**Do not use 37 and 26.** Those numbers appear in older revisions of
`docs/lawgic_dataset_report.md` and `docs/lawgic_dual_head_architecture.md`. They were
never measured. Git holds only one version of `lawgic_topics.json`, and it yields 42 and
30. Both documents are now corrected.

### 2.2 Mask width per clause, measured on the built corpus

| Sources for the clause | Clauses | Mask width |
| --- | --- | --- |
| ToS;DR only | 21,876 | 42 |
| 100 ToS only | 1,402 | 30 |
| CLAUDETTE only | 3,092 | 1 to 4, mean 1.16 |
| CLAUDETTE and ToS;DR | 51 | 42 |
| 100 ToS and CLAUDETTE | 36 | 30 |
| 100 ToS and ToS;DR | 19 | 44 |
| All three | 3 | 44 |

A CLAUDETTE mask width of 21 never occurs. A width of 10 never occurs. Both numbers
describe the mapping table, not the mask.

### 2.3 Supervision totals

| Cell type | Count |
| --- | --- |
| Supervised positive | 49,154 |
| Supervised negative | 919,461 |
| Unsupervised | 222,940 |
| Total | 1,191,555 |
| Positive ratio among supervised | 0.0507 |

The total equals 26,479 rows multiplied by 45 topics.

### 2.4 The CLAUDETTE supervision gap

| Quantity | Value |
| --- | --- |
| CLAUDETTE-only clauses | 3,092, which is 11.7% of the corpus |
| Supervised cells they carry | 3,573, which is 0.37% of all supervised cells |
| Supervised **negatives** they carry | **0** |

### 2.5 The raw CLAUDETTE corpus, which is already in the repository

Path: `datasets/claudette_cross_market/`. It holds 142 documents in parallel directories
named `sentences/`, `labels/`, `tags/`, `tags_unfair/`, `xml/`, and `txt/`.

| Quantity | Value |
| --- | --- |
| Sentences in the corpus | 37,862 |
| Sentences carrying at least one tag | 3,221 |
| Sentences with two or more distinct categories | 310, which is 9.6% of tagged |
| Untagged sentences | 34,641 |
| Untagged and 5 words or longer | **27,206** |
| Tagged and 5 words or longer | 3,221 |

Tag counts per category: `ltd` 1,072, `ter` 624, `ch` 506, `use` 370, `cr` 261, `law` 225,
`j` 218, `a` 165, `pinc` 115.

---

## 3. Defect 1: the mask and the labels use different tables

This is the primary code defect.

**The labels.** `map_tosdr_topic()` in cell 10 of
`notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` reads `TOSDR_TOPIC_OVERRIDES` first. The
overrides make the label set narrow.

**The mask.** `build_source_coverage_mask()` in cell 20 reads `source_mappings` from the
taxonomy file. It does not read the overrides. The mask is wide.

The mask is wider than the labels. Four topics sit inside the ToS;DR mask but cannot
receive a ToS;DR positive. Every ToS;DR row therefore supervises them as negative.

| Topic | ToS;DR positives | Total positives | Supervised negatives | Ratio |
| --- | --- | --- | --- | --- |
| `limitation_of_liability` | 0 | 1,186 | 23,243 | 1 : 20 |
| `service_changes` | 0 | 115 | 23,272 | 1 : 202 |
| `liability_cap` | 0 | 57 | 23,330 | 1 : 409 |
| `price_changes` | 0 | 50 | 23,337 | 1 : 467 |

Causes:

- `Guarantee` maps only to `warranty_disclaimer`, so ToS;DR cannot reach
  `limitation_of_liability` or `liability_cap`.
- `Changes` maps only to `contract_changes`, so ToS;DR cannot reach `price_changes` or
  `service_changes`.

The label-count chart on slide 55 confirms this. None of these four bars has a ToS;DR
segment.

### 3.1 The dead `Waivers` override

`Waivers` maps to `indemnification`. The ToS;DR data never uses the label `Waivers`. There
are 0 rows. The 100 ToS data never uses the code `indemn`. There are 0 rows.

Two results follow:

- `indemnification` has 0 positives and 23,387 negatives. It is unlearnable.
- The override removed `limitation_of_liability` and `liability_cap` from the ToS;DR label
  set and gained nothing, because `Waivers` does not occur.

---

## 4. Defect 2: the notebook outputs were cleared. Fixed

Cell 20 prints the coverage. Before 31 July 2026 the saved notebook had no output for that
cell, and `execution_count` was `None` for every code cell. Therefore no document quoted a
measured number.

**Status: fixed on 31 July 2026.** The notebook was executed end to end. Every code cell
now carries an execution count from 1 to 13 and its stored output. Cell 6 has no output
because it only defines functions.

Cell 20 printed:

```
  tos_dr taxonomy coverage: 42 / 45 topics
  100_tos taxonomy coverage: 30 / 45 topics
  claudette taxonomy coverage: 21 / 45 topics
```

Cell 22 printed:

```
  Supervised positives  (mask=1 & label=1): 49,154
  Supervised negatives  (mask=1 & label=0): 919,461
  Unsupervised/ignored  (mask=0):           222,940
  Positive ratio among supervised:          0.0507
```

### 4.1 The corpus is reproducible

Seven output files were compared with a backup taken before the run. All seven are
byte-identical, and a `shasum -a 256 -c` check passed on all six CSV files.

```
lawgic_combined_long.csv              IDENTICAL
lawgic_combined_wide.csv              IDENTICAL
lawgic_multihead_wide.csv             IDENTICAL
lawgic_combined_conflicts.csv         IDENTICAL
lawgic_100_tos_parse_diagnostics.csv  IDENTICAL
lawgic_tosdr_parse_diagnostics.csv    IDENTICAL
lawgic_fusion_summary.json            IDENTICAL
```

State this in the reproducibility section of the manuscript.

---

## 5. Defect 3: two masking regimes

`compute_source_aware_mask()` in cell 21 applies two different rules.

- ToS;DR and 100 ToS get a **source-level** mask. Every row from that source gets the same
  mask, 42 or 30 wide. Silence inside that width becomes a negative.
- CLAUDETTE gets a **per-row** mask. The mask covers only the topics named by the label on
  that row, 1 to 4 wide. Silence is never turned into a negative.

The vector `source_coverage["claudette"]` is computed, printed, and then never read.

**The stated justification is incomplete.** The report says a missing CLAUDETTE annotation
tells us nothing about the 34 topics CLAUDETTE never evaluates. That is correct and rules
out a 44-wide mask. It does not explain why CLAUDETTE cannot speak about the other 9
topics inside its own list of 10.

**The source-paper audit did not settle this.** Lippi et al. (2019) allow multi-label
annotation through nested tags:

> "Nested tags were used to annotate text segments relevant to more than one type of
> clause." (Section 3.1, page 5)

They do not state that annotators checked all nine categories on every sentence. The
papers are silent on that point.

**Therefore do not infer negatives from tagged clauses.** Defect 5 provides real negatives
instead, and they are better evidence. See section 11, Fix E.

---

## 6. Defect 4: five ToS;DR labels expand to three or four topics

Section 3.2 of the dataset report says the overrides stop the fan-out problem. The override
table has five entries. Five other labels still fan out.

| ToS;DR label | Points | Long rows | Topics produced |
| --- | --- | --- | --- |
| `Governance` | 2,476 | 9,904 | `contract_by_use`, `complaint_system`, `discretionary_interpretation`, `governance` |
| `Suspension and Censorship` | 1,149 | 4,596 | `account_suspension`, `account_termination`, `complaint_system`, `content_removal` |
| `Transparency` | 1,381 | 4,143 | `interpretation_clause`, `recommender_transparency`, `transparency` |
| `User Choice` | 886 | 2,658 | `contract_by_use`, `governance`, `privacy_incorporation` |
| `Ownership` | 411 | 1,233 | `content_retrieval`, `feedback_reuse`, `ownership` |

ToS;DR has 22,404 unique points. The pipeline made 44,317 long rows from them. The fan-out
factor is 1.98. The pipeline made 21,913 extra rows, and these five labels made 16,231 of
them.

This contradicts the rule in section 3 of the dataset report. That rule says a source must
not assert a distinction it does not annotate. The pipeline applies the rule to CLAUDETTE
`ltd`. It does not apply the rule to ToS;DR `Governance`.

---

## 7. Defect 5: the CLAUDETTE extraction discards 91.5 percent of the corpus

This is the largest defect. It was found last.

`generated_files/claudette_cross_market/claudette_cross_market_clauses.csv` holds 3,556
rows. Every row carries a category tag. The file holds no untagged sentence.

The raw corpus in `datasets/claudette_cross_market/` holds 37,862 sentences. Only 3,221
carry a tag. **34,641 untagged sentences were dropped during extraction.** After the
5-word filter used by the original authors, 27,206 remain usable.

### 7.1 The source papers define those sentences as the negative class

This is not an inference. It is the corpus authors' own method.

Jablonowska et al. (2021), Section 2, page 63:

> "We consider a binary classification task: the positive class is made by all potentially
> or clearly unfair clauses, for all categories, indiscriminately, and the negative class
> by all the remaining clauses."

Lippi et al. (2019), Section 5, page 12:

> "The problem is formulated as a binary classification task, where the positive class is
> either the union of all potentially unfair sentences, or the set of potentially unfair
> clauses of a single category ... and the negative class otherwise."

The papers also confirm the annotators read whole documents, and that the released corpus
keeps untagged sentences. Lippi et al. (2019), Section 3.2, page 9:

> "The final corpus contains 12,011 sentences overall, 1,032 of which (8.6%) were labeled
> as positive, thus containing a potentially unfair clause."

The papers nowhere warn against treating an unmarked sentence as a negative example.

### 7.2 What recovery would give

| Item | Now | After recovery |
| --- | --- | --- |
| CLAUDETTE clauses in the corpus | 3,092 | about 30,400 |
| Total corpus rows | 26,479 | about 53,700 |
| Negative cells from CLAUDETTE | 0 | about 272,000 |

Compare this with the alternative of inferring negatives on tagged clauses, which would add
27,419 cells. Recovery adds ten times more, and they are grounded in documents.

### 7.3 The cost

This is a corpus overhaul, not a patch.

- The corpus roughly doubles.
- The 80/10/10 split must be rebuilt. The current persisted split no longer covers all rows.
- Untagged sentences carry no harm score. They need `harm_mask = 0`. At present all 26,479
  rows carry a harm label. That property ends.
- The harm head trains on a much smaller share of the corpus.
- Every number in the results chapter is replaced.

Do not start this before the defense.

---

## 8. Defect 6: three mapping rules contradict the published definitions

The verbatim category definitions were obtained from Lippi et al. (2019), Section 3.1.
Three rules in `CLAUDETTE_TOPIC_RULES` disagree with them.

### 8.1 The `a` expansion is not supported. Remove it

Current rule: `a` maps to `mandatory_arbitration` **and** `class_action_waiver`. The
dataset report calls this the one deliberate one-to-many expansion.

The definition, page 9:

> "The arbitration clause requires or allows the parties to resolve their disputes through
> an arbitration process, before the case could go to court. It is therefore considered a
> kind of forum selection clause."

The definition says nothing about class actions. The source-paper audit marked the topic as
completely absent.

Effect: 164 of the 559 `class_action_waiver` positives, which is 29 percent, come from an
unsupported expansion.

This breaks the pipeline's own rule that a native label licenses only the topic it reliably
entails.

**Action: change `a` to map to `mandatory_arbitration` only.**

### 8.2 The `ter` collapse may be too narrow

The definition, page 7:

> "The unilateral termination clause gives provider the right to **suspend** and/or
> **terminate** the service and/or the contract ..."

The definition names suspension directly. The current rule maps `ter` to
`account_termination` only, and drops `account_suspension`.

**Action: decide and record the reasoning.** The phrase "and/or" means an individual clause
may be one, the other, or both. A defensible reading is that neither child is reliably
entailed alone, so the current collapse stands. An equally defensible reading is that the
definition names both, as with the arbitration case the pipeline wrongly assumed. Write the
choice down either way.

### 8.3 The `ch` collapse may be too narrow

The definition, page 7:

> "The unilateral change clause specifies the conditions under which the service provider
> could amend and modify the **terms of service and/or the service itself**."

The definition names the service directly. The current rule maps `ch` to `contract_changes`
only, and drops `service_changes`.

**Action: same as section 8.2.** Decide and record.

The definition supports neither `price_changes` nor `notice_of_changes`. The slide 53
example remains correct.

### 8.4 The six rules that are correct

| Label | Rule | Definition support |
| --- | --- | --- |
| `law` | `choice_of_law` | Direct match |
| `j` | `choice_of_forum` | "what courts will have the competence to adjudicate disputes" |
| `ltd` | `limitation_of_liability` | Confirmed undivided. No sub-codes for cap, warranty, or indemnity |
| `cr` | `content_removal` | "right to modify/delete user's content". Does not support `content_rules` |
| `use` | `contract_by_use` | "bound by the terms of use ... simply by using the service". Does not support `governance` |
| `pinc` | `privacy_incorporation` | "the scope of consent granted to the ToS incorporates also the privacy policy" |

### 8.5 Provenance facts recovered from the papers

- Inter-annotator agreement on CLAUDETTE is **Cohen's kappa 0.871**, reported by Galassi
  et al. (2025), footnote 9, page 646.
- The 42 contracts added for the cross-market set were marked by **two independent
  annotators** (Jablonowska et al., 2021, Section 2, page 62).
- The original corpus states only "legal experts". The papers do not give the annotator
  count, and are silent on the disagreement-resolution protocol. Report this gap.
- The original authors discarded sentences shorter than 5 words. This reduced their working
  set from 12,011 to 9,414 sentences.
- A category tag marks the topic. The appended digit marks fairness: 1 clearly fair, 2
  potentially unfair, 3 clearly unfair. A tagged clause can be clearly fair. This confirms
  the score-mapping rule.
- Some categories have no level 1 at all. `use`, `cr`, and `pinc` are always at least
  potentially unfair, so the tag itself implies unfairness for those three.

---

## 9. Duplicate and near-duplicate label columns

A pairwise scan compared all 44 label columns and all 44 mask columns on all 26,479 rows.

**Identical. Difference is 0 rows.**

- `transfer_of_contract` and `business_transfer`
- `recommender_transparency` and `transparency`

**Near-identical. Difference is 27 rows, which is 0.10 percent.**

- `transparency`, 1,368 positives
- `recommender_transparency`, 1,368 positives
- `interpretation_clause`, 1,341 positives

All three agree on 26,452 of 26,479 rows. The cause is the ToS;DR label `Transparency`,
which produces all three. Only 27 rows from 100 ToS separate them.

`docs/lawgic_duplicate_topics_fix.md` plans to drop `recommender_transparency`. After that
change `interpretation_clause` remains a near-copy of `transparency`. The brief does not
cover this. Update it.

**Nested, not identical.** `contract_by_use` and `governance` have identical
`source_mappings` but differ on 366 rows, because CLAUDETTE `use` maps only to
`contract_by_use`. `governance` is a strict subset of `contract_by_use`.

Fix C in section 11 also removes the near-identical group, because restricting
`Transparency` separates the three columns.

---

## 10. File inventory: what causes what

| File | Part | Effect |
| --- | --- | --- |
| `generated_files/lawgic_taxonomy/lawgic_topics.json` | `source_mappings` | Gives raw coverage 42, 30, 21. Three topic groups share identical lists |
| `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` cell 4 | `CLAUDETTE_TOPIC_RULES` | Holds the unsupported `a` expansion. Defect 6 |
| Same, cell 4 | `TOSDR_TOPIC_OVERRIDES` | Only 5 entries. Five more labels still fan out. Defect 4 |
| Same, cell 8 | `parse_claudette()` | Reads only the tagged-clause CSV. Defect 5 |
| Same, cell 10 | `map_tosdr_topic()` | Applies the overrides to labels only |
| Same, cell 20 | `build_source_coverage_mask()` | Ignores the overrides. Builds the wide mask. Defect 1 |
| Same, cell 21 | `compute_source_aware_mask()` | Mixes the source-level and per-row regimes. Defect 3 |
| `generated_files/claudette_cross_market/claudette_cross_market_clauses.csv` | whole file | Holds only 3,556 tagged rows. Missing 34,641 untagged sentences |
| `datasets/claudette_cross_market/sentences/`, `labels/`, `tags/` | 142 files each | The untagged sentences are here. Not yet used |
| `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv` | `topic_mask` | Contains the false negatives |
| `docs/lawgic_dataset_report.md` | §9.1 table | Reported 37 and 26. Corrected 31 July 2026 |
| `docs/lawgic_dual_head_architecture.md` | mask table | Reported 37 and 26. Corrected 31 July 2026 |
| `docs/lawgic_duplicate_topics_fix.md` | defect section | Lists two pairs. Does not list the near-identical group of three |

---

## 11. Implementation brief

Six fixes. They are ordered by cost. Read the whole section before starting.

**Rule for every fix: back up first, run the notebook, save the outputs, compare the
checksums, commit the notebook with its outputs.**

```
cd "/Users/riki/Coding Projects/Thesis/lawgic"
mkdir -p ~/lawgic_backup_$(date +%Y%m%d_%H%M)
cp generated_files/lawgic_taxonomy/*.csv generated_files/lawgic_taxonomy/*.json \
   ~/lawgic_backup_$(date +%Y%m%d_%H%M)/
```

### Fix 0: correct the `a` mapping rule

Cost: minutes. Data change: small. Retrain: required.

In cell 4, change:

```python
"a": ["mandatory_arbitration", "class_action_waiver"],
```

to:

```python
"a": ["mandatory_arbitration"],
```

Update `MAPPING_NOTES` to record the reason and quote the definition.

Expected effect: `class_action_waiver` loses 164 positives, falling from 559 to about 395.
No other topic changes.

### Fix A: make the mask agree with the labels

Cost: half a day. Data change: mask only. Retrain: required.

Replace `build_source_coverage_mask()` in cell 20:

```python
def build_source_coverage_mask(taxonomy, topic_id_to_index):
    """Coverage = the topics a source can actually label positive."""
    coverage = {src: [0.0] * len(topic_id_to_index)
                for src in ["tos_dr", "100_tos", "claudette"]}

    # tos_dr: honour TOSDR_TOPIC_OVERRIDES, exactly as map_tosdr_topic() does.
    for native_label in raw_source_to_topics["tos_dr"]:
        for topic_id in map_tosdr_topic(native_label):
            coverage["tos_dr"][topic_id_to_index[topic_id]] = 1.0

    # 100_tos: no override layer, so the raw mapping is the effective mapping.
    for native_label, topic_list in raw_source_to_topics["100_tos"].items():
        for topic_id in topic_list:
            coverage["100_tos"][topic_id_to_index[topic_id]] = 1.0

    # claudette: the union of the narrow per-label rules.
    for topic_list in CLAUDETTE_TOPIC_RULES.values():
        for topic_id in topic_list:
            coverage["claudette"][topic_id_to_index[topic_id]] = 1.0

    for src in coverage:
        print(f"  {src} effective coverage: {int(sum(coverage[src]))} "
              f"/ {len(topic_id_to_index)} topics")
    return coverage
```

ToS;DR coverage falls from 42 to 38. The other two do not change.

Expected effect, measured by simulation on the current corpus:

| Quantity | Before | After |
| --- | --- | --- |
| Supervised positives | 49,154 | 49,154 |
| Supervised negatives | 919,461 | 831,764 |
| Positive ratio | 0.0507 | 0.0558 |

Fix A removes 87,697 false negative cells, which is 9.5 percent of all negatives. No
positive is lost.

Per-topic negatives after Fix A: `limitation_of_liability` 1,327, `service_changes` 1,345,
`liability_cap` 1,403, `price_changes` 1,410.

### Fix B: add the permanent guards

Cost: one hour. Data change: none. Retrain: not required.

Add three assertions to the corpus builder. Each must fail the build and name the offender.

1. No two label columns may be element-wise identical.
2. Every masked topic must be reachable by at least one contributing source. This makes
   Defect 1 impossible to repeat.
3. Keep the existing `supervised_negative > 0` assertion.

Consider a fourth warning, not an error: report any two label columns that differ on fewer
than 1 percent of rows. This would have caught the transparency group.

### Fix C: restrict the five fan-out ToS;DR labels

Cost: one day, plus the legal reasoning. Data change: large. Retrain: required.

Add entries to `TOSDR_TOPIC_OVERRIDES`. Three have an obvious narrowest topic:

| Label | Proposed target | Confidence |
| --- | --- | --- |
| `Governance` | `governance` | High |
| `Transparency` | `transparency` | High |
| `Ownership` | `ownership` | High |
| `Suspension and Censorship` | needs a legal judgment | Low |
| `User Choice` | needs a legal judgment | Low |

Write the reasoning for each before writing the code.

Fix C also removes the near-identical group in section 9, because `Transparency` is the
only reason `interpretation_clause` tracks `transparency`.

### Fix D: keep the dead `Waivers` override

Cost: minutes. Data change: none.

Keep the override. ToS;DR declares `Waivers` in its own case-topic list, so it is inside
the source vocabulary even though this snapshot does not use it. Deleting it would let
`Waivers` fall back to the raw mapping, reach six topics, and undo part of Fix A.

Record the reason in `MAPPING_NOTES`.

### Fix E: recover the untagged CLAUDETTE sentences

Cost: several days plus a full retrain. Data change: the corpus roughly doubles. **Do not
start before the defense.**

This replaces the earlier idea of inferring negatives on tagged clauses. Real negatives are
better evidence than inferred negatives, and there are ten times more of them.

Steps:

1. Write a new extractor that reads `datasets/claudette_cross_market/sentences/`,
   `labels/`, and `tags/` in parallel, one file per document, aligned by line index.
2. Apply the original authors' filter. Discard sentences shorter than 5 words. This leaves
   27,206 untagged and 3,221 tagged sentences.
3. Emit tagged sentences as now, through `CLAUDETTE_TOPIC_RULES`.
4. Emit untagged sentences as new rows with all 10 CLAUDETTE topics masked and every label
   set to 0.
5. Set `harm_mask = 0` on the untagged rows. They carry no fairness judgment.
6. Rebuild the split. The current persisted split does not cover the new rows.
7. Retrain and rerun evaluation phases 1, 2, and 3. Phase 4 does not depend on the corpus.

Expected effect: about 30,400 CLAUDETTE clauses, a corpus of about 53,700 rows, and about
272,000 new negative cells.

**Watch these risks.**

- The harm head loses its property that every row has a label. Report the new share.
- The corpus becomes CLAUDETTE-heavy. At present ToS;DR is 88 percent of rows. Check the
  new balance and report it.
- Untagged sentences include headings and boilerplate. The 5-word filter removes many, but
  inspect a sample before trusting the rest.
- Deduplicate against the existing corpus. Some CLAUDETTE sentences may already be present
  from another source.

### What does not need to change

The split logic is stratified on the primary active topic and the harm class, both derived
from labels. For Fix A, Fix B, and Fix D the labels do not change, so the existing
persisted split still applies and the leakage check still passes. Fix 0, Fix C, and Fix E
change labels or rows, so they need a new split.

---

## 12. Verification checklist

After each fix, confirm these numbers before committing.

| After | Check | Expected |
| --- | --- | --- |
| Any run | Cell 22 total | positives + negatives + unsupervised = rows × 45 |
| Any run | `supervised_negative` | greater than 0 |
| Fix 0 | `class_action_waiver` positives | about 395, down from 559 |
| Fix A | Cell 20 print for `tos_dr` | 38 |
| Fix A | Supervised negatives | 831,764 |
| Fix A | Positive ratio | 0.0558 |
| Fix A | `price_changes` negatives | 1,410 |
| Fix B | Duplicate-column assertion | fails on the two known pairs until Fix C runs |
| Fix E | Corpus rows | about 53,700 |
| Fix E | Rows with `harm_mask = 0` | about 27,200 |

---

## 13. What to change in the manuscript

The manuscript is in the other repository.

1. Correct the coverage table to 42 and 30, or to the new numbers after a rebuild.
2. State the basis of every coverage number in the same sentence as the number.
3. State the mask width per clause. CLAUDETTE gives 1 to 2 cells per annotation and 1 to 4
   per clause.
4. Add the CLAUDETTE provenance: Cohen's kappa 0.871, two independent annotators for the 42
   new contracts, and the gap in the published disagreement protocol.
5. Add a limitation for Defect 1. Four topics receive negatives from a source that cannot
   give them a positive. The cost is 87,697 cells.
6. Add a limitation for Defect 4. Five ToS;DR labels expand to three or four topics, and
   the fan-out factor is 1.98.
7. Add a limitation for Defect 5. The extraction uses 8.5 percent of the CLAUDETTE corpus,
   and 27,206 usable negative sentences were discarded.
8. Correct the mapping section. Remove the claim that CLAUDETTE `a` covers class action
   waivers. Quote the definition.
9. Correct the duplicate section. Report two identical pairs and one near-identical group
   of three. Report `contract_by_use` and `governance` as nested, not identical.
10. Keep the `indemnification` disclosure. Add the cause: neither source emitted the label.
11. Add the reproducibility statement. The corpus rebuilds byte-for-byte.

Do not remove any disclosure. The defects support the argument. The first fusion attempt
failed and you reported it. These findings are the same kind of evidence.

---

## 14. What to change in the slides

1. **Slide 50.** It says 100 ToS has 9 categories. The correct number is 24. The 9 was
   copied from the CLAUDETTE card.
2. **Slide 53 footer.** Do not list "CLAUDETTE 1 to 2, ToS;DR 42, 100 ToS 30" as one set.
   They are different kinds of number. Use two lines:
   - "CLAUDETTE: the mask covers only the topics named by the label on the row. Width 1 to 2."
   - "ToS;DR and 100 ToS: the mask covers every topic the source can express. Width 42 and 30."
3. Do not say "contributed". Say "can answer" or "covers".
4. Slide 53 stays correct otherwise. Its example clause carries the label `ch`, its mask
   width is 1, and the definition of `ch` supports neither `price_changes` nor
   `notice_of_changes`.
5. Put the fan-out data and the CLAUDETTE recovery on the limitations slide, not on slide 53.

---

## 15. Order of work

| # | Task | Status |
| --- | --- | --- |
| 1 | Correct the two documents reporting 37 and 26 | Done, 31 July 2026 |
| 2 | Correct the slide footer and slide 50 | With the author |
| 3 | Run the notebook and save the outputs | Done, 31 July 2026 |
| 4 | Confirm the corpus is reproducible | Done. All files identical |
| 5 | Write the new limitations into the manuscript | Open |
| 6 | Fix 0, the `a` mapping | Open. Minutes, plus a retrain |
| 7 | Fix A, the mask correction | Open. Half a day, plus a retrain |
| 8 | Fix B, the guards | Open. One hour |
| 9 | Fix D, document the `Waivers` decision | Open. Minutes |
| 10 | Fix C, the fan-out labels | After the defense |
| 11 | Fix E, recover the untagged sentences | After the defense |

### If you have limited time

Do task 5 only. Report the defects with their measured effects and the plan. You do not
need working code to defend a known and quantified defect.

### If you have one working day

Do Fix 0, Fix A, Fix B, and Fix D together. They are all small. Then retrain Legal-BERT
with one seed and compare only the affected topics. Do not rebuild the whole results
chapter on one run.

### Warning about a partial retrain

Any fix that changes the corpus invalidates every number in the current results chapter.
Do not put old numbers and new numbers in one table. Either report the current corpus
everywhere and describe the fixes as future work, or rerun the whole protocol.
