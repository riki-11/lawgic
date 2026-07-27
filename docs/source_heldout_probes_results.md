# Source-Held-Out Probe Results — Interpretation and Manuscript Guidance

**Date:** 2026-07-27
**Scope:** Phase 3 of the statistical-rigor upgrade (`CHANGES.md`). Executed on the Windows machine; artifacts synced.
**Companion:** `docs/multiseed_evaluation_results.md` (Phases 1–2). Read Part I of that document first — the mental model there (seeds, bootstrap, pairing) applies unchanged here.
**Artifacts:** `generated_files/lawgic_taxonomy/evaluation/phase3_source_holdout.{csv,tex}`, `runs/legal-bert-base-uncased__seed42__dual__holdout-{claudette,100_tos}/`

> ### ⏸ Manuscript status (2026-07-27): deferred until after the proposal defense
>
> **Part V of this document is on hold. Do not apply it yet.** The probe has been removed from the manuscript in its entirety — the experiment, the results, and the motivation — on the grounds that the finding cannot be understood, written, and defended in the time remaining before the proposal defense. Reinstatement is planned for after it.
>
> What that means for this document: Parts I–IV and Part VI stand as the analysis of record. **Part V is the reinstatement plan, not a to-do list for now.** The removal is recorded, with the exact edits and a reinstatement checklist, in the Appendix of `misc/das_defense_prep.md` in the manuscript repository. Two `TODO (post-defense)` blocks in `chapter_4.tex` mark where the text comes back.
>
> Nothing in the code was reverted: the notebook, both trained probe runs, and the persisted logits all remain, so reinstatement is a writing task.

---

## 0. The one-paragraph version

The probe asks whether the classifier reads clauses or recognises datasets. Two retrains, each removing one source from training entirely, then evaluating only on that source's clauses. **Both probes show a large, statistically unambiguous drop.** On the notebook's evaluation surface the model retains 65% of its topic macro-F1 for both sources. On a corrected surface that restores observed negatives, CLAUDETTE retains 71% of micro-F1 and 100 ToS retains only 44%. The risk head collapses below a constant-predictor baseline in both probes. **Cross-source generalisation is partial, not intact, and the honest reading is that a meaningful share of the headline score depends on having seen the annotation project during training.**

But the collapse is *not* uniform, and the pattern is the finding. Formulaic clause families (arbitration, choice of law, account termination) transfer nearly intact. The topics that collapse split into two mechanisms — topics whose supervision the holdout *removed*, and topics that keep 95%+ of their supervision yet still go to zero. Only the second is evidence of source-trace matching. Separating them is what turns this from a bad number into a defensible contribution.

---

# Part I — What was run, and why in exactly this way

## I.1 The question the probe answers

The corpus is fused from three annotation projects. Section \ref{sec:masking} of the manuscript already names the risk this creates: each source leaves recognisable traces on its rows — drafting register, clause segmentation, label mix, and above all **mask shape** (a ToS;DR row is observed on 42 topics, a 100 ToS row on 30, a CLAUDETTE row on 1–4). A model could learn "this row shape plus this register means ToS;DR, and ToS;DR rows about trackers are usually harmful," score well on a test set drawn from the same three sources, and have read nothing.

A random test split cannot detect this, because the test split contains the same three sources in the same proportions. That is the entire reason the probe exists.

## I.2 Why *this* design and not a cheaper one

You already had a cheaper diagnostic: the linear source-classifier probe (`outputs/source_probe_cache/source_probe_results.csv`), which decodes source identity from the fine-tuned encoder at 0.872 macro-F1 versus 0.575 from raw Legal-BERT. **That result is necessary but not sufficient**, and knowing why is the methodological point.

An encoder can *carry* source information without the heads *using* it. Fine-tuning on a fused corpus will inevitably make source linearly decodable, because register correlates with label distribution and the encoder has no reason to discard a correlated feature. The linear probe measures what the representation contains. It cannot measure what the classifier depends on.

The source-held-out probe measures dependence directly, by ablating the thing in question:

- Remove one source's rows from **train and validation**.
- Retrain the whole model from scratch under the identical protocol.
- Evaluate **only on that source's rows in the test split**.

Now the traces the model could have exploited genuinely do not exist in its training experience. If it still classifies those clauses, it learned the clause. If it does not, it was leaning on the traces. This is a counterfactual test, not a correlational one, which is why it costs 2 GPU-hours and the linear probe cost minutes.

## I.3 Why only two of three sources

| Probe | Held out | Wide rows | Train rows removed | Test rows scored |
|---|---|---|---|---|
| A | CLAUDETTE | 3,182 | 2,528 (11.9%) | 324 |
| B | 100 ToS | 1,460 | 1,175 (5.5%) | 141 |
| — | ToS;DR | 21,949 | 17,559 (82.9%) | *not run* |

Removing ToS;DR would leave roughly 3,600 training clauses. A collapse under that condition is uninterpretable: it is equally consistent with "the model only recognised ToS;DR" and with "nobody learns 44-way multi-label legal topic detection from 3,600 examples." **The experiment would be confounded with data starvation and would answer neither question.** The two smaller sources can be removed while leaving the training regime broadly intact, which is what makes their results readable at all.

Keep that sentence in the manuscript. It is the kind of deliberate non-experiment a panelist respects when it is explained and punishes when it looks like an omission.

## I.4 Why the in-distribution reference must be recomputed, not reused

The obvious mistake here would be comparing the probe's score to the headline 0.771 topic macro-F1. That comparison is meaningless. The probe scores 324 clauses from one source under one source's mask; the headline scores 2,648 clauses from three sources under their union mask. Different rows, different cells, different denominators.

So the reference is built to match exactly: the **Phase 2 legal-bert/seed-42 run's persisted test logits**, re-scored on *the identical rows* and *the identical cells* the probe faces. Same clauses, same mask, same metric code, only the training experience differs. That is what makes "retained ratio" a legitimate quantity.

This is the single most important design detail in Phase 3, and it is only possible because Phase 2 persisted raw logits rather than scalar metrics. The reference cost zero GPU time.

## I.5 Why the mask restriction, and where it went wrong

The masking decision decides whether the probe means anything, and the notebook gets the *direction* right and the *width* wrong.

Right: a held-out CLAUDETTE row must not be scored on all 44 topics, because CLAUDETTE's nine labels reach only ten of them. Grading the other 34 would grade the shape of the mask rather than comprehension of the clause.

Wrong: `source_supervision_mask()` in `scripts/lawgic_train_matrix.py` rebuilds the mask from `native_annotations`, and `native_annotations` records **only asserted annotations** — that is, only positives. The intersection therefore contains **only positive cells**:

```
CLAUDETTE probe: 375 observed cells, 375 positive, 0 negative
100 ToS  probe: 197 observed cells, 197 positive, 0 negative
```

**Consequence: false positives are structurally impossible on this surface.** Verified in the persisted logits — `fp = 0` in all four conditions, and micro-precision is exactly 1.000 everywhere, including in-distribution. Per topic, F1 collapses to `2r/(1+r)`, a monotone function of recall alone. The probe as scored is a **recall probe**, and a model that predicted every topic present for every clause would score a perfect 1.000 on it.

This is the same degenerate-surface failure that motivated source-aware masking in the first place (manuscript §4.2: a positive-only corpus "can reward nothing except answering yes"), reappearing one layer up in the evaluation. The manuscript's own §4.2 states the correct rule: a source's coverage is "the set of topics its annotation scheme is capable of expressing… **not narrowed to the topics that happen to receive a positive**." The probe narrowed it to exactly that.

**The fix costs no GPU time.** The probe logits are persisted, so re-scoring under the row's own corpus supervision mask — which already encodes each source's coverage correctly, and which for these rows is that source's mask, since 96.3% of CLAUDETTE test rows and 95.7% of 100 ToS test rows are single-source — restores the negatives:

```
CLAUDETTE probe: 842 observed cells, 397 positive, 445 negative
100 ToS  probe: 4,286 observed cells, 205 positive, 4,081 negative
```

Both surfaces are reported below. They tell partly different stories, and the difference between them is itself a result.

---

# Part II — The numbers

## II.1 As produced by the notebook (positives-only surface)

Ready to paste from `phase3_source_holdout.tex`. Read this as a **recall** result.

| | CLAUDETTE | 100 ToS |
|---|---|---|
| Test rows scored | 324 | 141 |
| Observed cells (all positive) | 375 | 197 |
| Topics with any observed cell | 10 | 26 |
| **Topic macro-F1, in-distribution** | **0.894** | **0.791** |
| **Topic macro-F1, held out** | **0.581** | **0.520** |
| Retained ratio | 0.649 | 0.657 |
| Δ macro-F1, bootstrap 95% CI | [−0.355, −0.276] | [−0.340, −0.218] |
| Macro recall, in-distribution | 0.812 | 0.729 |
| Macro recall, held out | 0.495 | 0.467 |
| Micro-precision (both conditions) | 1.000 | 1.000 |
| Topic micro-F1, in-dist → held out | 0.899 → 0.626 | 0.845 → 0.641 |
| **Risk accuracy, in-distribution** | **0.772** | **0.560** |
| **Risk accuracy, held out** | **0.460** | **0.411** |
| Δ risk accuracy, bootstrap 95% CI | [−0.364, −0.256] | [−0.241, −0.057] |
| Risk macro-F1, in-dist → held out | 0.696 → 0.361 | 0.489 → 0.358 |
| **Majority-class risk accuracy on the same rows** | **0.744** | **0.610** |

The bootstrap CIs on the deltas are mine, computed from the persisted logits over 2,000 clause-level resamples; the notebook did not produce them. **Both topic deltas and both risk deltas exclude zero.** The collapse is not seed noise or sampling noise. (One seed only, so strictly this is one draw from the training-noise distribution — see the caveats in Part IV.)

## II.2 On the corrected surface (negatives restored)

Macro-F1 is not the right headline here, because averaging over all 44 topics on 141 rows sends most topics to an undefined-coerced-to-zero F1 and drags the mean down even in-distribution (0.386 for 100 ToS). **Micro-F1 is the honest aggregate on this surface** — it averages over predictions, and every observed cell contributes.

| | CLAUDETTE | 100 ToS |
|---|---|---|
| Observed cells | 842 (397 pos / 445 neg) | 4,286 (205 pos / 4,081 neg) |
| **Topic micro-F1, in-distribution** | **0.877** [0.841, 0.909] | **0.656** [0.585, 0.727] |
| **Topic micro-F1, held out** | **0.624** [0.570, 0.677] | **0.286** [0.245, 0.327] |
| Retained ratio | 0.711 | 0.435 |
| Δ micro-F1, bootstrap 95% CI | [−0.301, −0.207] | [−0.444, −0.304] |
| Micro-precision, in-dist → held out | 0.961 → 0.953 | 0.598 → **0.203** |
| Micro-recall, in-dist → held out | 0.806 → 0.464 | 0.727 → 0.483 |
| False positives, in-dist → held out | 13 → 9 | 100 → **389** |

**This is the table that changes the interpretation.** The two probes fail in different ways:

- **CLAUDETTE degrades by silence.** Recall halves (0.806 → 0.464); precision is untouched (0.961 → 0.953); false positives actually go *down* (13 → 9). The model, deprived of CLAUDETTE, simply stops firing on CLAUDETTE-style clauses. It does not start guessing.
- **100 ToS degrades by silence *and* noise.** Recall drops as before (0.727 → 0.483), but precision collapses (0.598 → 0.203) and false positives quadruple (100 → 389). Here the model does start guessing.

The positives-only surface reported micro-precision of exactly 1.000 for both. It hid the entire second failure mode. **This is the concrete cost of the mask defect, and it is why the corrected surface has to be reported.**

Note also that 100 ToS in-distribution micro-precision is already only 0.598 — the model over-predicts on 100 ToS rows *even when trained on them*, because those rows carry wide 30-topic masks and its ToS;DR-shaped priors fire into them.

## II.3 Per-source performance in-distribution — context you need before reading any ratio

A retained ratio is only meaningful against its denominator. From the Phase 2 legal-bert/seed-42 run, scored per source on each row's own mask:

| Test rows | n | Topic micro-F1 | Micro-P | Micro-R | Risk accuracy | Majority-class risk accuracy |
|---|---|---|---|---|---|---|
| All | 2,648 | 0.836 | 0.855 | 0.818 | **0.841** | 0.469 |
| ToS;DR | 2,199 | 0.841 | 0.860 | 0.822 | **0.868** | 0.435 |
| CLAUDETTE | 324 | 0.877 | 0.961 | 0.806 | **0.772** | 0.744 |
| 100 ToS | 141 | 0.656 | 0.598 | 0.727 | **0.560** | 0.610 |

Two things fall out of this table, both of which belong in the manuscript independently of Phase 3:

1. **The majority-class baseline you still owed is 0.469** (predict *neutral* for every clause). It is a property of the labels, so it holds for every run. The manuscript's selected checkpoint (0.836, seed 2024) clears it by 0.367; the seed-42 run above scores 0.841 and clears it by 0.372. That is one of the two trivial floors §4.4.1 promises and it now exists at zero GPU cost.
2. **The risk head does not work on the two smaller sources.** On CLAUDETTE test rows it beats a constant predictor by 0.028 (0.772 vs 0.744). On 100 ToS test rows it is 0.050 *below* a constant predictor (0.560 vs 0.610). The headline 0.841 is carried almost entirely by ToS;DR rows, where it beats the floor by 0.433.

That second point is uncomfortable and it should be reported anyway. It is also the correct frame for the probe's risk numbers: held-out risk accuracy of 0.460 and 0.411 is not a collapse from a working system, it is further degradation of a head that was already marginal on those sources.

## II.4 Which topics survive, which collapse, and why — the actual finding

Per-topic results joined against how much training supervision each topic *keeps* after the holdout (positive annotations in the train split from the remaining sources):

**CLAUDETTE holdout** — all ten scored topics retain some supervision, so there is no zero-supervision confound here.

| Topic | Held-out F1 | Recall | Support | Train positives from CLAUDETTE | Remaining from other sources | CLAUDETTE's share |
|---|---|---|---|---|---|---|
| account_termination | 0.944 | 0.894 | 66 | 502 | 1,118 | 31% |
| content_removal | 0.870 | 0.769 | 26 | 213 | 1,973 | 10% |
| mandatory_arbitration | 0.865 | 0.762 | 21 | 124 | 336 | 27% |
| choice_of_law | 0.842 | 0.727 | 22 | 178 | 788 | 18% |
| choice_of_forum | 0.837 | 0.720 | 25 | 170 | 776 | 18% |
| class_action_waiver | 0.833 | 0.714 | 21 | 124 | 329 | 27% |
| contract_changes | 0.308 | 0.182 | 44 | 412 | 827 | 33% |
| limitation_of_liability | 0.306 | 0.181 | 105 | **856** | 103 | **89%** |
| contract_by_use | **0.000** | 0.000 | 31 | 291 | **2,698** | 10% |
| privacy_incorporation | **0.000** | 0.000 | 14 | 87 | **1,970** | 4% |

**100 ToS holdout** — five topics lose *all* supervision, because 100 ToS is their only supplier.

| Topic | Held-out F1 | Support | Remaining supervision | Note |
|---|---|---|---|---|
| liability_cap | 0.000 | 7 | **0** | unlearnable in probe condition |
| severability | 0.000 | 8 | **0** | unlearnable |
| service_changes | 0.000 | 12 | **0** | unlearnable |
| price_changes | 0.000 | 5 | **0** | unlearnable |
| limitation_period | 1.000 | 1 | **0** | unlearnable; scored on one clause |
| complaint_system | 0.000 | 3 | 2,904 | keeps 99% of supervision |
| transparency | 0.000 | 3 | 1,110 | keeps 98% |
| recommender_transparency | 0.000 | 3 | 1,110 | keeps 98% |
| content_rules | 0.000 | 10 | 975 | keeps 92% |
| ownership | 0.143 | 13 | 327 | keeps 79% |
| transfer_of_contract / business_transfer | 0.222 | 8 | 241 | keeps 86% |
| warranty_disclaimer | 1.000 | 7 | 1,280 | keeps 95% |
| account_termination | 1.000 | 16 | 1,414 | keeps 87% |
| choice_of_forum, arbitration, class_action_waiver | 1.000 | 8, 2, 2 | 878, 431, 431 | keeps 93–95% |
| choice_of_law | 0.947 | 10 | 886 | keeps 92% |
| content_removal | 0.947 | 10 | 2,100 | keeps 96% |
| copyright_license | 0.933 | 8 | 452 | keeps 89% |
| limitation_of_liability | 0.800 | 12 | 856 | keeps 89% |

Excluding the five topics with zero remaining supervision raises the 100 ToS probe from macro-F1 0.520 (26 topics) to **0.596** (21 topics).

**Three mechanisms, and the manuscript should name all three:**

1. **Supervision withdrawal.** The holdout removed the topic's main or only supplier, so the topic became unlearnable in the probe condition. `limitation_of_liability` (89% CLAUDETTE-supplied, recall 0.18) and the five 100-ToS-exclusive topics are this. **This is not a reading failure.** It is the per-topic version of exactly the data-starvation confound that made a ToS;DR probe uninterpretable — the confound was avoided at corpus level and reappeared at topic level. Anything in this class must be excluded from a claim about source recognition.
2. **Register transfer failure.** The topic keeps 92–99% of its supervision and still goes to zero recall: `contract_by_use` (2,698 remaining, F1 0.000), `privacy_incorporation` (1,970, 0.000), `complaint_system` (2,904, 0.000), `content_rules` (975, 0.000), `transparency` and `recommender_transparency` (1,110, 0.000). **This is the real evidence of source dependence.** The model learned what a ToS;DR *point* about contract-by-use looks like, not what contract formation by use *is*, and cannot recognise the same mechanism in a CLAUDETTE sentence.
3. **Genuine cross-source generalisation.** Arbitration, class-action waiver, choice of law, choice of forum, account termination, content removal, warranty disclaimer, copyright licence — F1 0.83 to 1.00 with the source entirely absent from training. These are the formulaic clause families whose wording repeats across services, and they are exactly the topics §4.4.5 already identifies as the model's strongest. **The concept transferred.** This is a positive result and it is the only part of Phase 3 that is good news, so do not bury it.

The three-way split is the contribution. "The model retains 65% of its performance" is a number. "Cross-source generalisation holds for formulaic clause families, fails for broad discretionary ones, and cannot be assessed at all for topics a single source supplies" is a finding, and it points at a corpus problem (single-supplier topics) rather than only a model problem.

## II.5 The risk head's failure mode is over-prediction of harm

From the confusion matrices in the persisted logits (rows = true harmful / neutral / fair):

**CLAUDETTE probe.** True distribution 66 / 241 / 17. In-distribution the model predicts 92 / 212 / 20. Held out it predicts **162** / 135 / 27 — it calls 119 of the 241 truly-neutral clauses harmful. Accuracy 0.460 against a majority-class floor of 0.744.

**100 ToS probe.** True distribution 86 / 47 / 8. In-distribution the model predicts 53 / 77 / 11 (already under-calling harm on a harm-heavy subset). Held out it predicts 36 / 83 / 22 — it drifts *further* toward neutral on a subset where harmful is the majority class. Accuracy 0.411 against a floor of 0.610.

Note the two probes fail in **opposite directions** on risk: over-calling harm on CLAUDETTE clauses, under-calling it on 100 ToS clauses. That rules out a single global bias and points at the scoring conventions instead. CLAUDETTE's fairness levels map to a mostly-neutral distribution (74% neutral) while 100 ToS's integer scores map to a mostly-harmful one (61% harmful). Without the source in training, the risk head reverts toward the prior it learned from ToS;DR, which sits between the two. **The risk head learned the source's scoring convention as much as the clause's severity** — the sharpest single statement Phase 3 supports, and it is a statement about the fused corpus's score-mapping layer (§4.2), not only about the model.

For a consumer risk tool, over-calling harm on unfamiliar register is the less damaging of the two errors, but under-calling it on 100 ToS-style clauses is not. Say so.

---

# Part III — How to read all of this yourself

The five questions to ask of any held-out probe number, in this order.

**1. What is the denominator?** 324 rows and 141 rows. A single clause is 0.3% and 0.7% of those subsets respectively. Every per-topic number with support under 10 — and most of the 100 ToS table is under 10 — is anecdote, not statistic. `limitation_period` scoring a perfect 1.000 on one clause is the clearest example: it is *both* a topic with zero remaining supervision *and* a perfect score, which is arithmetically possible and evidentially worthless.

**2. Are negatives in the evaluation surface?** If not, F1 is a recall proxy and precision is free. Check `fp = 0`; that is the tell. This is the defect in §I.5 and it is a general-purpose lesson: any time you restrict a metric to "the cells the source annotated," check whether the source's annotation format records negatives. Most do not.

**3. Does the delta exclude zero?** Bootstrap the *difference* on paired rows, not the two scores separately. Here every headline delta excludes zero by a wide margin — the topic macro-F1 drop is 0.28 to 0.36 for CLAUDETTE with a CI that never approaches zero. That is what makes it safe to call this a real effect rather than one unlucky retrain.

**4. Did the topic still have supervision?** This is the question that separates "the model cannot generalise" from "the topic no longer existed." Compute, per topic, the training positives that survive the holdout. Anything at or near zero remaining is excluded from the generalisation claim by construction. Skipping this step turns a corpus-coverage problem into a false claim about the model.

**5. What does a trivial predictor score on the same rows?** 0.460 risk accuracy sounds like a collapse from 0.841. It is worse than that: the constant-predictor floor on those specific rows is 0.744, so the held-out model is far *below* trivial. And 0.560 in-distribution on 100 ToS rows is already below its 0.610 floor. **A metric without its floor is uninterpretable, and the floor moves with the subset.** This is the single most common way to over-read a per-subset number.

---

# Part IV — What this protocol does *not* establish

State these before someone else does.

- **One seed.** Both probes are seed 42 only. Phase 2 measured seed noise on topic macro-F1 for this configuration at sd 0.003, and the observed drops are 0.27–0.36, roughly a hundred times that — so the *direction* is not in doubt. The precise retained ratio has one draw of training noise in it and should be quoted to two decimals at most.
- **Precision is unmeasured on the as-published surface.** §I.5. The corrected surface fixes it; the published `phase3_source_holdout.tex` does not. Either regenerate that table or state the restriction in its caption.
- **Supervision withdrawal is confounded with register transfer** for `limitation_of_liability`, `contract_changes`, and the five 100-ToS-exclusive topics. The per-topic partition in §II.4 bounds the confound; it does not remove it. Fully removing it would require a probe that holds out a source's *rows* while preserving its topics' supervision from elsewhere, which is not possible for a single-supplier topic — the corpus has no other supplier.
- **No ToS;DR probe**, so the largest source's contribution to the headline score is unmeasured. Everything Phase 3 says about source dependence is inferred from the 17% of the corpus that is *not* ToS;DR.
- **Test subsets are small.** 324 and 141 rows. Every CI here is wide by construction, and the per-topic tables should be read as pattern, not measurement.
- **The probe cannot separate register from segmentation.** CLAUDETTE annotates single sentences; ToS;DR quotes multi-sentence passages. A held-out CLAUDETTE clause differs from the training distribution in *length* as well as in wording, and this design cannot tell which one the model missed. A length-stratified breakdown would be cheap from the persisted logits and would sharpen the claim.

---

# Part V — What to put in the manuscript

`chapter_4.tex` currently frames the probe as committed work in §4.4.7 (`sec:committedexperiments`), lines 478–480. That framing is now wrong. Phase 3 needs to become a results subsection.

## V.1 Structural change

Move the probe out of "Committed Experiments" and insert a new subsection between **Per-Topic Performance** (`sec:pertopicresults`) and **Model Selection** (`sec:modelselection`) — it must precede model selection, because the probe is evidence about the corpus that bears on what the deployed model can be claimed to do. Suggested label `sec:sourceprobes`.

Keep the motivating paragraph currently at lines 478–480 (it is good — it explains mask shape as a trace, and it explains why ToS;DR is not held out). Change only its tense and append the results. §4.4.7 then retains the readability check and the change-pipeline discussion.

Also update:

| Location | Change |
|---|---|
| Chapter overview (line 16) | "the source-held-out probes and the readability assessment remain committed" → probes are done; only readability remains |
| §4.4.1 (line 378) | Majority-class floor now exists: **risk accuracy 0.469**. One of the two remaining floors is discharged; only the per-topic prevalence predictor is still outstanding |
| §4.4.6 Model Selection (line 473) | Add the generalisation caveat: the deployed checkpoint's score is partly source-conditional, so the user study's documents (TikTok / YouTube / X live ToS) are out-of-distribution relative to all three sources |
| §4.2 Final Corpus (line 233) | Add single-supplier topics as a corpus limitation: five topics are supplied by 100 ToS alone and four more depend on CLAUDETTE for a third or more of their supervision, so removing either source makes them unlearnable |

## V.2 Draft prose

Adapt freely; the numbers are what matter.

> \subsection{Source-Held-Out Probes}
> \label{sec:sourceprobes}
>
> [keep the existing motivating paragraphs from §\ref{sec:committedexperiments} here, in past tense]
>
> Table \ref{tab:source-holdout} reports both probes against an in-distribution reference computed from the Phase 2 Legal-BERT seed-42 run on the identical rows and the identical supervised cells, so the only difference between the two columns is whether the source appeared in training. Neither probe retains its in-distribution performance. Holding out CLAUDETTE reduces topic micro-F1 on CLAUDETTE's test clauses from 0.877 to 0.624, and holding out 100 ToS reduces it from 0.656 to 0.286. Both reductions are far outside their bootstrap confidence intervals (95\% CI on the difference $[-0.301, -0.207]$ and $[-0.444, -0.304]$ over 1{,}000 clause-level resamples). A share of the reported performance is therefore conditional on having seen the annotation project during training, which is a limitation of the fused-corpus design.
>
> The two probes fail differently, and the difference locates the mechanism. Deprived of CLAUDETTE, the model becomes silent rather than wrong: recall falls from 0.806 to 0.464 while precision is unchanged (0.961 to 0.953) and the false-positive count falls from 13 to 9. Deprived of 100 ToS, the model becomes both silent and noisy: recall falls from 0.727 to 0.483 and precision collapses from 0.598 to 0.203, with false positives rising from 100 to 389.
>
> Degradation is not uniform across topics, and the per-topic pattern separates three mechanisms. Formulaic clause families transfer nearly intact with the source entirely absent from training: \textit{account termination} scores F1 0.944, \textit{content removal} 0.870, \textit{mandatory arbitration} 0.865, \textit{choice of law} 0.842, and \textit{class action waiver} 0.833 in the CLAUDETTE probe. These are the same topics Section \ref{sec:pertopicresults} identifies as the model's strongest, and their wording repeats across services, so the concept and not the annotation project is what the model learned for them.
>
> A second group collapses because the holdout removed the topic's supervision rather than its register. \textit{Limitation of liability} draws 89\% of its training positives from CLAUDETTE, and \textit{liability cap}, \textit{severability}, \textit{service changes}, \textit{price changes}, and \textit{limitation period} draw all of theirs from 100 ToS. In the probe condition these topics have little or no supervision left, so their scores measure unlearnability and not source recognition. Excluding the five topics with no remaining supervision raises the 100 ToS probe's topic macro-F1 from 0.520 to 0.596. That these topics exist at all is a corpus limitation: nine of the 44 topics depend on a single source for most or all of their supervised positives.
>
> A third group is the actual evidence of source dependence. \textit{Contract formed through use} retains 2{,}698 training positives from ToS;DR, \textit{privacy incorporation} 1{,}970, \textit{complaint system} 2{,}904, and \textit{content rules} 975 --- between 92\% and 99\% of their supervision --- and every one of them scores F1 0.000 on the withheld source's clauses. For these topics the model learned what the training source's phrasing of the topic looks like rather than what the contractual mechanism is, and it does not recognise the same mechanism in another project's drafting.
>
> The risk head degrades furthest, and it degrades below the trivial floor. Held-out risk accuracy is 0.460 on CLAUDETTE clauses and 0.411 on 100 ToS clauses, against majority-class baselines of 0.744 and 0.610 computed on the identical rows. The two probes err in opposite directions: on CLAUDETTE clauses the held-out model calls 162 of 324 clauses harmful where 66 are, and on 100 ToS clauses it calls 36 of 141 harmful where 86 are. Each source's scoring convention has a different marginal distribution --- CLAUDETTE's fairness levels resolve to 74\% neutral on these rows and 100 ToS's integer scores to 61\% harmful --- and without the source in training the risk head reverts toward the ToS;DR prior that lies between them. The head is therefore fitting the source's scoring convention alongside the clause's severity. The in-distribution references disclose the same limitation more directly: on 100 ToS test rows the Phase 2 model reaches risk accuracy 0.560 against a 0.610 majority-class floor, so the head does not exceed a constant predictor on that source even when trained on it, and the headline risk accuracy of Table \ref{tab:decontaminated-eval} is carried by the ToS;DR rows that make up 83\% of the corpus, where it reaches 0.868 against a 0.435 floor.
>
> Two bounds on these conclusions are stated plainly. Each probe is a single training run at seed 42, so the retained ratios carry one draw of the training noise that Section \ref{sec:encodercomparison} measures at 0.003 for this configuration; the direction of the effect is far outside that noise but the exact ratios are not precise. And the two withheld subsets contain 324 and 141 clauses, so the per-topic figures describe a pattern rather than measuring it.

## V.3 The table

The generated `phase3_source_holdout.tex` reports the positives-only surface with a `Retained ratio` column and no negatives, which the caption does not disclose. Replace it with a table on the corrected surface, or publish both. Suggested single table:

```latex
\begin{table}[t]
\centering
\small
\caption{Source-held-out probes. Each probe retrains Legal-BERT (seed 42, dual-head) with one source removed from train and validation under the otherwise identical protocol, then evaluates only on that source's test clauses. The in-distribution column is the Phase 2 Legal-BERT seed-42 run scored on the identical rows and the identical supervised cells, so the columns differ only in training experience. Topic metrics are computed over each row's own supervision mask, which contains observed negatives; the majority-class risk floor is computed on the same rows. Confidence intervals are percentile bootstrap over 1{,}000 clause-level resamples of the difference. No ToS;DR probe is reported: removing 82.9\% of the training rows would confound source recognition with data starvation.}
\label{tab:source-holdout}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{CLAUDETTE}} & \multicolumn{2}{c}{\textbf{100 ToS}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& \textbf{In-dist.} & \textbf{Held out} & \textbf{In-dist.} & \textbf{Held out} \\
\midrule
Test clauses & \multicolumn{2}{c}{324} & \multicolumn{2}{c}{141} \\
Supervised cells & \multicolumn{2}{c}{842} & \multicolumn{2}{c}{4,286} \\
\midrule
Topic micro-F1  & 0.877 & 0.624 & 0.656 & 0.286 \\
\quad precision & 0.961 & 0.953 & 0.598 & 0.203 \\
\quad recall    & 0.806 & 0.464 & 0.727 & 0.483 \\
$\Delta$ micro-F1, 95\% CI & \multicolumn{2}{c}{$-0.253$ $[-0.301, -0.207]$} & \multicolumn{2}{c}{$-0.370$ $[-0.444, -0.304]$} \\
\midrule
Risk accuracy & 0.772 & 0.460 & 0.560 & 0.411 \\
\quad majority-class floor & \multicolumn{2}{c}{0.744} & \multicolumn{2}{c}{0.610} \\
\bottomrule
\end{tabular}
\end{table}
```

(The risk figures do not depend on which topic surface you choose — every row in both subsets carries risk supervision, so 0.772 and 0.560 are the in-distribution values on the identical 324 and 141 rows either way.)

## V.4 Numbers to transfer, checklist form

- [ ] Probe row counts: CLAUDETTE 324 test clauses / 2,528 train rows removed (11.9%); 100 ToS 141 / 1,175 (5.5%)
- [ ] Topic micro-F1, corrected surface: 0.877 → 0.624 (CLAUDETTE), 0.656 → 0.286 (100 ToS)
- [ ] Δ micro-F1 CIs: [−0.301, −0.207] and [−0.444, −0.304]
- [ ] Precision behaviour: 0.961 → 0.953 (CLAUDETTE, silence) vs 0.598 → 0.203 (100 ToS, noise)
- [ ] Held-out risk accuracy 0.460 / 0.411 against majority floors 0.744 / 0.610 **on the same rows**
- [ ] In-distribution risk accuracy by source: ToS;DR 0.868, CLAUDETTE 0.772, 100 ToS 0.560; floors 0.435 / 0.744 / 0.610
- [ ] **Majority-class risk accuracy on the full test split: 0.469** (discharges half of the trivial-baseline commitment in §4.4.1)
- [ ] Transfers intact: account_termination 0.944, content_removal 0.870, mandatory_arbitration 0.865, choice_of_law 0.842, choice_of_forum 0.837, class_action_waiver 0.833
- [ ] Supervision-withdrawal topics: limitation_of_liability 89% CLAUDETTE-supplied; liability_cap, severability, service_changes, price_changes, limitation_period 100% 100-ToS-supplied
- [ ] Excluding zero-supervision topics: 100 ToS probe macro-F1 0.520 → 0.596
- [ ] Register-failure topics (≥92% supervision retained, F1 0.000): contract_by_use, privacy_incorporation, complaint_system, content_rules, transparency, recommender_transparency
- [ ] Risk prediction distributions: CLAUDETTE 162/135/27 predicted vs 66/241/17 true; 100 ToS 36/83/22 vs 86/47/8
- [ ] Probe cost: 39.3 min + 39.0 min ≈ 1.3 GPU-hours, 14 and 13 epochs, both early-stopped inside the 20-epoch budget
- [ ] Single-seed disclosure and the positives-only-surface disclosure

---

# Part VI — Actions, in order

1. **Decide how to present the two surfaces.** Recommend: report the corrected surface (row's own mask, micro-F1) as the headline and mention the positives-only figures in a footnote as recall-side detail. Both come from persisted logits; no retraining. If you keep the notebook's table, its caption must state that the surface contains no observed negatives.
2. **Fix `source_supervision_mask()` or its use.** The mask is built from `native_annotations`, which stores positives only. Either intersect with each source's *coverage* (derivable from `source_mappings` in `lawgic_topics.json`) or use the row's own `label_mask` for single-source rows, which is 96% of them. Then re-run the notebook's cells 8–10 — no GPU needed.
3. **Add the per-topic supervision-survival column** to the probe output. Without it the probe cannot distinguish unlearnability from source dependence, which is its whole purpose.
4. **Report the majority-class baseline (0.469)** in §4.4.1 now. Free, and the chapter promised it.
5. **Add single-supplier topics to §4.2 as a corpus limitation.** Nine of 44 topics depend on one source for most or all of their positives. This is a finding about the corpus that Phase 3 surfaced and that Chapter 4 does not currently state.
6. **Optional, cheap, high value:** length-stratified breakdown of the CLAUDETTE probe from the persisted logits, to separate register failure from segmentation-length failure.
