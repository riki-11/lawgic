# Defense Numbers Rundown

Language: ASD-STE100 Simplified Technical English.
Date: 31 July 2026.
Purpose: every number you must know for the proposal defense, and where each one comes from.

---

## 0. How to use this document

Numbers fall into three groups.

- **Group A. Fixed.** These describe the data. A code fix does not change them.
- **Group B. Will change.** Fix A changes these. Say them, but say that a correction is planned.
- **Group C. Do not say.** These are wrong, stale, or unmeasured.

Every section marks its numbers with A, B, or C.

---

## 1. The three sources (Group A)

| Source | Native labels | Long rows | Share |
| --- | --- | --- | --- |
| ToS;DR | 28 declared, 25 used | 44,317 | 88.5% |
| CLAUDETTE Cross Market | 9 | 3,721 | 7.4% |
| 100 ToS | 24 | 2,048 | 4.1% |
| **Total** | | **50,086** | 100% |

ToS;DR has 22,404 unique points. The pipeline made 44,317 long rows from them. The
fan-out factor is 1.98.

**Where these come from.** The row counts are printed by cell 14 of
`notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb`. They are stored in
`lawgic_fusion_summary.json` under `long_summary.rows_by_source`.

**Slide error to fix (Group C).** Slide 50 of `thesis ppt 2026.pdf` says 100 ToS has
**9 categories**. The correct number is **24**. The 9 was copied from the CLAUDETTE card.

---

## 2. The taxonomy (Group A)

| Quantity | Value |
| --- | --- |
| Topic identifiers | 45 |
| Substantive topics | 44 |
| Fallback topic | 1, named `unclassified` |
| Parent topics | 12 |
| Independently identifiable columns | 42 |

**Why 44 and not 42.** Two pairs of topics draw on identical native labels, so their
columns are equal on all 26,479 rows:

- `transfer_of_contract` and `business_transfer`
- `recommender_transparency` and `transparency`

The model trains all 44. Report macro-F1 over both 44 and 42.

**A near-identical group of three (Group A, newly found).** These three columns agree on
26,452 of 26,479 rows, which is 99.90 percent:

- `transparency`, 1,368 positives
- `recommender_transparency`, 1,368 positives
- `interpretation_clause`, 1,341 positives

The cause is the ToS;DR label `Transparency`. It produces all three topics. Only 27 rows
from 100 ToS separate them. Disclose this. Your current fix brief covers only two pairs.

**Not a duplicate.** `contract_by_use` and `governance` have identical `source_mappings`
but differ on 366 rows, because CLAUDETTE `use` goes only to `contract_by_use`. They are
nested, not identical.

**Where this comes from.** A pairwise scan of all 44 label columns in
`lawgic_multihead_wide.csv`.

---

## 3. Source coverage (Group B)

| Source | Method A, raw file | Method B, after overrides |
| --- | --- | --- |
| CLAUDETTE | 21 | 10 |
| ToS;DR | 42 | 38 |
| 100 ToS | 30 | 30 |

**Which to say now.** Say 42 for ToS;DR and 30 for 100 ToS. That is what the code does
today. The printed output of cell 20 confirms it:

```
  tos_dr taxonomy coverage: 42 / 45 topics
  100_tos taxonomy coverage: 30 / 45 topics
  claudette taxonomy coverage: 21 / 45 topics
```

The denominator 45 includes `unclassified`. Quote 42 of 44, not 42 of 45.

### 3.1 The CLAUDETTE number 21 is never applied

This is important. Cell 20 computes and prints 21 for CLAUDETTE. Cell 21 then ignores it.

Read `compute_source_aware_mask()`. It reads `source_coverage` for `tos_dr` and for
`100_tos` only. For CLAUDETTE it reads `CLAUDETTE_TOPIC_RULES` and applies the rule for
the native label on that row. The vector `source_coverage["claudette"]` is never used.

Therefore CLAUDETTE has no source-level mask width. It has a per-row mask width of 1 to 4.
The values 21 and 10 never appear as a mask width on any clause. This was verified on all
3,092 CLAUDETTE-only clauses.

**How to say it.** Do not put 21, 42, and 30 in one list. They are not the same kind of
number. Say this instead:

- "ToS;DR rows are graded on all 42 topics ToS;DR can express."
- "100 ToS rows are graded on all 30 topics 100 ToS can express."
- "CLAUDETTE rows are graded only on the topics named by the label on that row, which is
  1 to 4 topics."

The numbers 21 and 10 describe the mapping table, not the mask. Use them only when you
explain the mapping table.

### 3.2 Where the CLAUDETTE 21 comes from

CLAUDETTE has 9 native labels. The taxonomy file lists each label under every topic it
relates to. The counts add up to 21.

| Native label | Topics in the JSON file | Topics after the rule |
| --- | --- | --- |
| `ch` | 5 | 1 |
| `ltd` | 4 | 1 |
| `a` | 2 | 2 |
| `ter` | 2 | 1 |
| `cr` | 2 | 1 |
| `use` | 2 | 1 |
| `pinc` | 2 | 1 |
| `law` | 1 | 1 |
| `j` | 1 | 1 |
| **Total** | **21** | **10** |

No topic appears under two different CLAUDETTE labels. The sum and the union are both 21.

`CLAUDETTE_TOPIC_RULES` cuts 21 down to 10. It keeps only the topic each label reliably
entails. `ltd` keeps `limitation_of_liability` and drops the cap, the warranty, and the
indemnity. `ch` keeps `contract_changes` and drops the four change subtypes. `a` keeps both
arbitration topics, because the native category names both.

**Do not say (Group C): 37 and 26.** Those numbers appear in older revisions of
`docs/lawgic_dataset_report.md` and `docs/lawgic_dual_head_architecture.md`. They were
never measured and match no version of the taxonomy file. Both documents are corrected.

---

## 4. Score harmonisation and exclusions (Group A)

Score distribution across the 50,086 long rows:

| Score | Rows |
| --- | --- |
| -1, harmful | 13,410 |
| 0, neutral | 24,282 |
| +1, fair | 12,394 |

Excluded and logged:

| Reason | Count |
| --- | --- |
| 100 ToS, unknown code or free text | 912 |
| 100 ToS, unclear or legal-savings fragment | 594 |
| 100 ToS, documentary note | 291 |
| 100 ToS, known code without valid score | 12 |
| 100 ToS, empty referenced text | 1 |
| **100 ToS total** | **1,810** |
| ToS;DR, empty quote text | 1,464 |

Nothing is dropped silently. Every exclusion carries a typed reason.

---

## 5. The fused corpus (Group A)

| Quantity | Value |
| --- | --- |
| Unique normalised clauses | 26,479 |
| Harmful, -1 | 8,311, 31.4% |
| Neutral, 0 | 12,368, 46.7% |
| Fair, +1 | 5,800, 21.9% |
| Rows with no harm label | 0 |
| Score conflicts, flagged | 60 |
| Clauses annotated by more than one source | 109, 0.4% |

Multi-source breakdown:

| Combination | Clauses |
| --- | --- |
| CLAUDETTE and ToS;DR | 51 |
| 100 ToS and CLAUDETTE | 36 |
| 100 ToS and ToS;DR | 19 |
| All three | 3 |

Conflict rate is 60 of 50,086 long rows, which is 0.12 percent. Conflicts resolve by the
minimum score, which is the pessimistic consumer-protection rule.

Token lengths of the clause text:

| Measure | Tokens |
| --- | --- |
| Median | 38 |
| 90th percentile | 101 |
| 95th percentile | 137 |
| 99th percentile | 268 |
| Maximum | 3,579 |
| Over the 256 limit | 298 clauses, 1.13% |

---

## 6. Supervision counts (Group B)

Printed by cell 22:

| Cell type | Count |
| --- | --- |
| Supervised positive, mask 1 and label 1 | 49,154 |
| Supervised negative, mask 1 and label 0 | 919,461 |
| Unsupervised, mask 0 | 222,940 |
| **Total** | **1,191,555** |
| Positive ratio among supervised | 0.0507 |

The total equals 26,479 rows multiplied by 45 topics. The arithmetic closes.

**This is the number that proves the fix worked.** The first fusion attempt had 49,154
positives and **0** negatives. A model that answered yes to everything scored a perfect
macro-F1 of 1.00. Source-aware masking produced 919,461 negatives.

Mask width per clause:

| Sources | Clauses | Mask width |
| --- | --- | --- |
| ToS;DR only | 21,876 | 42 |
| 100 ToS only | 1,402 | 30 |
| CLAUDETTE only | 3,092 | 1 to 4, mean 1.16 |
| CLAUDETTE and ToS;DR | 51 | 42 |
| 100 ToS and CLAUDETTE | 36 | 30 |
| 100 ToS and ToS;DR | 19 | 44 |
| All three | 3 | 44 |

**Say 1 to 2 per annotation, 1 to 4 per clause.** One clause can carry more than one
CLAUDETTE label.

Topic supervision problems:

| Problem | Topics |
| --- | --- |
| Zero positives | `indemnification` |
| Under 50 positives | `indemnification` 0, `limitation_period` 14, `user_participation_in_changes` 44 |
| Zero negatives | none |

`indemnification` has 0 positives because neither source emitted it. ToS;DR never uses the
label `Waivers`. 100 ToS never uses the code `indemn`. The topic is unlearnable because of
the data, not because of the model.

---

## 7. The known defect (Group B)

Four topics receive a supervised negative on every ToS;DR row. ToS;DR can never give them
a positive.

| Topic | Positives | Negatives now | Ratio now |
| --- | --- | --- | --- |
| `limitation_of_liability` | 1,186 | 23,243 | 1 : 20 |
| `service_changes` | 115 | 23,272 | 1 : 202 |
| `liability_cap` | 57 | 23,330 | 1 : 409 |
| `price_changes` | 50 | 23,337 | 1 : 467 |

The cause is in the code. Cell 20 builds the mask from the raw taxonomy file. Cell 10
builds the labels from an override table. The mask is wider than the labels.

Measured effect of the planned fix:

| Quantity | Now | After Fix A |
| --- | --- | --- |
| Supervised positives | 49,154 | 49,154 |
| Supervised negatives | 919,461 | 831,764 |
| Positive ratio | 0.0507 | 0.0558 |

Fix A removes **87,697** false negative cells. That is 9.5 percent of all supervised
negatives. No positive label is lost.

**Say this on stage.** "I found the defect, I measured it at 87,697 cells, and the fix is
scheduled. The current results stand on the current corpus."

---

## 8. The split (Group A)

| Split | Rows |
| --- | --- |
| Train | 21,183 |
| Validation | 2,648 |
| Test | 2,648 |
| **Total** | **26,479** |

The split is 80/10/10. Sampling is stratified on a key that joins the primary active topic
and the harm class. The seed is 42. The assignment is saved to a file, and all runs load
it. An exact-text leakage check stops training if one clause string appears in two splits.

---

## 9. The model (Group A)

| Item | Value |
| --- | --- |
| Encoder | `nlpaueb/legal-bert-base-uncased` |
| Layers | 12 |
| Parameters | about 110 million |
| Topic head | `Linear(768, 44)`, sigmoid, masked BCE |
| Risk head | `Linear(768, 3)`, softmax, cross-entropy |
| Decision threshold | 0.5 |
| Maximum sequence length | 256 tokens |
| Learning rate | 3e-5 |
| Maximum epochs | 20 |
| Early stopping patience | 3, on validation topic macro-F1 |
| Seeds | 42, 1337, 2024 |

Training stopped at **epoch 17**. The best validation masked macro-F1 was **0.761**.

**The degeneracy guard.** A model with all-zero logits predicts every topic present,
because sigmoid(0) equals 0.5. On the old broken corpus that scored a perfect 1.00. The
notebook asserts that this score is below 0.95 and raises `DEGENERATE` if not. Quote this
guard. It shows the failure cannot return unnoticed.

**Say "transformer encoder", not "LLM".** Legal-BERT is an encoder with 110 million
parameters. Reserve "LLM" for the generative layer.

---

## 10. Main results (Group B)

Legal-BERT, dual head, mean and standard deviation over 3 seeds, test set of 2,648 rows:

| Metric | Value |
| --- | --- |
| Topic macro-F1 | 0.771 ± 0.003 |
| Topic micro-F1 | 0.834 ± 0.002 |
| Risk accuracy | 0.838 ± 0.002 |
| Risk macro-F1 | 0.833 ± 0.002 |

Macro-F1 over the 42 independent columns is about **0.766**. Report both.

**Warning.** 0.766 is the 44-head model scored over 42 columns. It is not the score of a
42-head model. Never call it "the 42-head result".

The gap between micro-F1 0.834 and macro-F1 0.771 is a support effect. Rare topics drag
the macro average down.

Best topics and worst topics from the seed 2024 run:

| Topic | F1 | Test support |
| --- | --- | --- |
| `warranty_disclaimer` | 0.900 | 164 |
| `severability` | 0.889 | 8 |
| `security` | 0.889 | 71 |
| `class_action_waiver` | 0.885 | 57 |
| `choice_of_law` | 0.881 | 119 |
| `governance` | 0.881 | 337 |
| `transparency` | 0.745 | 136 |
| `interpretation_clause` | 0.727 | 133 |
| `price_changes` | 0.53 | 5 |
| `liability_cap` | 0.500 | 7 |
| `indemnification` | 0.000 | 0 |

Total test support across the 44 topics is 4,930 positive cells.

**Report support beside every F1.** A topic with 5 test examples cannot carry a claim.

---

## 11. Encoder comparison (Group B)

Dual head, mean over 3 seeds, topic macro-F1:

| Encoder | Topic macro-F1 | Risk macro-F1 |
| --- | --- | --- |
| RoBERTa | 0.776 ± 0.006 | 0.836 ± 0.005 |
| BERT | 0.775 ± 0.006 | 0.824 ± 0.006 |
| Legal-BERT | 0.771 ± 0.003 | 0.833 ± 0.002 |
| XLNet | 0.762 ± 0.031 | 0.830 ± 0.008 |

Paired tests against Legal-BERT, seed 2024:

| Comparison | Delta | 95% CI | p |
| --- | --- | --- | --- |
| against BERT | -0.010 | [-0.031, +0.010] | 0.329 |
| against RoBERTa | +0.002 | [-0.019, +0.021] | 0.832 |
| against XLNet | -0.019 | [-0.037, -0.001] | 0.033 |

**The finding.** Legal-domain pre-training gave no measurable advantage on the topic head.
All four encoders sit within 0.014 of each other. Three of the four are statistically
indistinguishable.

XLNet has a standard deviation of 0.031. That is ten times the standard deviation of
Legal-BERT. XLNet produced both the best single run and the worst single run.

**Say this as a finding, not a failure.** A defended null result is worth more than an
undefended claim. Legal-BERT is retained on the risk head and on the domain-adequacy
argument, not on a measured topic gain.

---

## 12. Head ablation (Group B)

The question: does one shared encoder with two heads beat two separate models?

Risk head, dual minus risk-only, risk macro-F1:

| Encoder | Mean delta | 95% CI | All seeds positive |
| --- | --- | --- | --- |
| RoBERTa | +0.023 | [-0.015, +0.061] | Yes |
| Legal-BERT | +0.014 | [+0.008, +0.020] | Yes |
| XLNet | +0.012 | [-0.014, +0.038] | Yes |
| BERT | +0.012 | [-0.018, +0.041] | No |

Topic head, dual minus topic-only, topic macro-F1:

| Encoder | Mean delta | All seeds positive |
| --- | --- | --- |
| BERT | +0.028 | Yes |
| Legal-BERT | -0.006 | No |
| RoBERTa | -0.003 | No |
| XLNet | -0.013 | No |

**The honest reading.** The second head helps the risk head. Legal-BERT gains 0.014 with a
confidence interval that excludes zero, and all three seeds agree. The second head does
not help the topic head. Three of four encoders lose a small amount.

Do not claim the dual head improves both tasks. Claim that it improves the risk task and
costs nothing on the topic task.

A control confirms the heads are separate. A risk-only model scores about 0.08 to 0.09
topic macro-F1. A topic-only model scores about 0.30 to 0.48 risk accuracy. Each head only
learns its own task.

---

## 13. Contamination check (Group B)

Near-duplicate clause text can appear in both the training split and the test split. The
audit measures this with cosine similarity.

Test split similarity to the nearest training clause:

| Threshold | Fraction of test rows |
| --- | --- |
| Over 0.95 | 3.66% |
| Over 0.90 | 6.84% |
| Over 0.80 | 15.48% |
| Median maximum similarity | 0.552 |

Scores on the full test set and on the cleaned test set:

| Metric | Full, n = 2,648 | Cleaned, n = 2,440 | Change |
| --- | --- | --- | --- |
| Topic macro-F1 | 0.770 | 0.762 | -0.008 |
| Topic micro-F1 | 0.831 | 0.828 | -0.003 |
| Risk accuracy | 0.836 | 0.838 | +0.002 |
| Risk macro-F1 | 0.831 | 0.833 | +0.002 |

The cleaning removed 208 rows. The largest change is 0.008. **Contamination does not
explain your results.** This is a strong number. Say it.

**Keep this separate from the duplicate-column problem.** This audit compares rows of
text. The duplicate topics are columns of labels. They are different checks.

---

## 14. Source hold-out probes (Group B)

The question: does the model learn the text, or does it learn which source a clause came
from?

Method: remove one source from training, then test on that source only.

| Held-out source | Metric | In distribution | Held out | Retained |
| --- | --- | --- | --- | --- |
| CLAUDETTE | Topic macro-F1 | 0.286 | 0.217 | 76% |
| CLAUDETTE | Topic micro-F1 | 0.877 | 0.624 | 71% |
| CLAUDETTE | Micro precision | 0.961 | 0.953 | 99% |
| CLAUDETTE | Micro recall | 0.806 | 0.463 | 58% |
| 100 ToS | Topic macro-F1 | 0.389 | 0.194 | 50% |
| 100 ToS | Topic micro-F1 | 0.656 | 0.286 | 44% |
| 100 ToS | Micro precision | 0.598 | 0.203 | 34% |
| 100 ToS | Micro recall | 0.727 | 0.483 | 66% |

CLAUDETTE test rows: 324. 100 ToS test rows: 141.

**How to read this.** Performance falls when a source is removed, but it does not fall to
zero. The model keeps 76 percent of macro-F1 on CLAUDETTE and 50 percent on 100 ToS.

CLAUDETTE precision stays at 99 percent of its original value. The model still gets the
right answer when it commits. Recall falls to 58 percent. The model becomes cautious, not
wrong.

**Be careful with this claim.** The probe shows partial transfer. It does not prove the
model ignores source identity. State the limitation.

---

## 15. Explanation readability (Group B)

| Metric | n | Clause | Explanation | Change | p | Improved |
| --- | --- | --- | --- | --- | --- | --- |
| Flesch Reading Ease | 87 | 37.97 | 34.54 | -3.43 | 0.040 | 36.8% |
| Flesch-Kincaid Grade | 87 | 13.69 | 14.26 | +0.57 | 0.143 | 41.4% |

Test: Wilcoxon signed-rank on 87 paired differences. Effect size for Reading Ease is
-0.25 rank-biserial.

**This is a negative result. Report it as one.** Reading Ease falls. Grade level rises,
but not significantly. Only 37 percent of explanations read easier than the clause they
explain.

Two facts must go with it.

1. **The Pravasi and Das instrument failed.** `py-readability-metrics` needs 100 words.
   The median change quotes 35 words and produces 56 words. Zero pairs were scorable. A
   strict replication is impossible at this unit of analysis.
2. **Legal-BERT does not write this text.** The generator is `gemma4:31b-cloud`. The
   classifier only supplies advisory context. The readability measured here is a property
   of the generator's prose.

---

## 16. What changes after Fix A, and what does not

**Does not change:**

- 3 sources, 50,086 long rows, 26,479 clauses
- 45 topic identifiers, 44 substantive, 42 identifiable
- Harm distribution 8,311 / 12,368 / 5,800
- 60 conflicts, 1,810 and 1,464 exclusions
- 109 multi-source clauses
- Token length statistics
- 49,154 supervised positives
- The 80/10/10 split
- All hyperparameters

**Changes:**

- Supervised negatives, 919,461 to 831,764
- Positive ratio, 0.0507 to 0.0558
- ToS;DR mask coverage, 42 to 38
- Negative counts for the four repaired topics
- Every model score, after a retrain

**Never put old scores and new scores in the same table.** Either report the current
corpus everywhere and call Fix A future work, or rerun the whole protocol.

---

## 17. Numbers you must not say

| Wrong number | Correct number | Where it appears |
| --- | --- | --- |
| ToS;DR coverage 37 | 42 | Two documents, now corrected |
| 100 ToS coverage 26 | 30 | Two documents, now corrected |
| 100 ToS has 9 categories | 24 | Slide 50 of the PDF deck |
| "42-head model scores 0.766" | "44-head model, macro over 42 columns" | Say it correctly |
| "CLAUDETTE contributed 10 topics" | "CLAUDETTE can answer 10 of 44" | Coverage, not contribution |
| "Two duplicate pairs" | Two identical pairs and one near-identical group of three | Disclose the group |

---

## 18. The five sentences to memorise

1. "The corpus is 26,479 clauses fused from three sources, carrying 44 topic labels and a
   three-class risk label."
2. "The first fusion had 49,154 positives and zero negatives, and a model that answered
   yes to everything scored a perfect 1.00. Source-aware masking produced 919,461
   negatives."
3. "Legal-BERT reaches topic macro-F1 0.771 plus or minus 0.003 over three seeds, and the
   four encoders are within 0.014 of each other, so legal pre-training gave no measurable
   topic advantage."
4. "Removing near-duplicate clauses from the test set moves macro-F1 by 0.008, so
   contamination does not explain the result."
5. "The corpus rebuilds byte-for-byte from the committed notebook, and I found and measured
   a masking defect worth 87,697 cells that the next iteration corrects."
