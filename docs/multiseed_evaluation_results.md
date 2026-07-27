# Multi-Seed Evaluation Results — Interpretation and Manuscript Guidance

**Date:** 2026-07-27
**Scope:** Phases 1 and 2 of the statistical-rigor upgrade (`CHANGES.md`). Phases 3 and 4 are written but not yet executed.
**Audience:** you, and anyone who asks "how did you conduct your methodology? I want to replicate it."

---

## 0. Status at a glance

| Phase | What it does | Status | Artifacts |
|---|---|---|---|
| 1 — Decontaminated re-scoring | Re-scores the shipped checkpoint on the full test split and on the near-duplicate-free subset, with bootstrap CIs | **Done** | `phase1_decontaminated.{csv,tex}`, `phase1_contamination_flags.csv`, `phase1_test_logits.npz` |
| 2 — Multi-seed / multi-encoder matrix | 18 training runs: 4 encoders × 3 seeds (dual-head) + Legal-BERT × 3 seeds × {topic-only, risk-only} | **Done** | `phase2_runs.csv`, `phase2_aggregate.csv`, `phase2_bootstrap_ci.csv`, `phase2_significance.csv`, `phase2_headline.{csv,tex}`, `phase2_per_topic.{csv,tex}`, `runs/<run_id>/` |
| 3 — Source-held-out probes | Retrain excluding CLAUDETTE / 100 ToS, evaluate on the withheld source only | **Not run** (notebook has zero executed cells) | — |
| 4 — Explanation readability | Flesch scores, generated explanation vs source clause | **Not run** (notebook has zero executed cells) | — |

All Phase 1–2 outputs live in `generated_files/lawgic_taxonomy/evaluation/`.

Total compute already spent on Phase 2: **18.0 GPU-hours** across 18 runs, mean 60.0 min/run (range 19.2–147.9 min), mean 15.3 epochs to early stop.

---

# Part I — How to read any of this (the mental model)

Before the numbers, the four ideas that make them mean something. If you can explain these four, you can defend the whole chapter.

## I.1 Why more than one seed

A neural network's final weights depend on the random seed: the initialisation of the two linear heads, the shuffling of the training batches, and dropout. Train the *same* model twice with different seeds and you get *different* test scores. Nothing about the architecture changed — only the random draw.

That means a single run gives you one sample from a distribution, not a measurement of the model. If Legal-BERT scores 0.771 in one run and BERT scores 0.775 in one run, you cannot say BERT is better. You have compared one coin flip to one coin flip.

The fix is to run each configuration several times with different seeds and report the **mean ± standard deviation**. The standard deviation is your noise floor. **A difference between two configurations only counts as real if it is larger than the noise within each configuration.** That single sentence is the whole logic of Section IV below.

Three seeds (42, 1337, 2024) is the minimum defensible number and the standard in the NLP literature; five is better but the compute cost is linear. You have three.

> Chapter 3 already gives you the citation that makes this argument for you: Melis et al. (2018) re-ran several recurrent architectures under an equal budget and found the older LSTM beat the newer models that had been reported to surpass it. The claimed architectural advantage was an artifact of unequal tuning. Your matrix is the same experiment on your own data — and it produces the same kind of result.

## I.2 Why bootstrap confidence intervals

Seeds tell you about **training noise**. The bootstrap tells you about **test-set noise** — the fact that your 2,648 test clauses are themselves a random sample. If you had drawn a different 2,648 clauses, the score would differ.

Procedure, exactly as implemented in `core.bootstrap_ci`:

1. Draw 2,648 test clauses **with replacement** from your test set. Some clauses appear twice, some not at all. This simulates "a different but equally plausible test set."
2. Recompute the metric on that resample.
3. Repeat 1,000 times.
4. The 2.5th and 97.5th percentiles of those 1,000 values are your **percentile 95% confidence interval**.

Read it as: "if the sampling were repeated, the interval computed this way would contain the true value 95% of the time." In practice: **the width of the interval is your measurement precision.** Your topic macro-F1 interval is roughly ±0.03 wide. Any difference smaller than that is inside the noise of the test set alone.

Note a subtlety visible in `phase2_bootstrap_ci.csv`: for macro-F1 the bootstrap `mean` is systematically *below* the `point` estimate (e.g. 0.7803 point vs 0.7700 mean for `bert-base-uncased__seed2024__dual`). That is expected. Macro-F1 averages over 44 topics, and resampling occasionally drops all positives of a low-support topic, sending that topic's F1 to 0 and dragging the average down. It is a real property of macro-F1 on an imbalanced label set, not a bug — and it is a second, independent argument for the low-support caveat in Section III.4.

## I.3 Why *paired* tests

The strongest thing about your design is that every one of the 18 runs scored **the same 2,648 test clauses in the same order**. That is what the persisted split file (`splits/split_seed42.csv`) buys you.

Because of that, you can compare two models *clause by clause* rather than score-to-score. Pairing removes clause difficulty from the comparison entirely: if a clause is hard, it is hard for both models, and the difficulty cancels. Paired tests are dramatically more powerful than unpaired ones at the same sample size.

Two paired tests are used, and they are different for a reason:

- **McNemar's test — for the risk head.** The risk head makes one 3-way decision per clause, so each clause is either right or wrong for each model. Build the 2×2 table of agreement. The cells that matter are `b` (model A right, model B wrong) and `c` (A wrong, B right). Clauses both models get right, or both get wrong, carry no information about which is better and are *discarded*. The test asks: given `b + c` disagreements, is the split between them further from 50/50 than chance allows? Exact binomial.
- **Paired bootstrap — for the topic head.** Macro-F1 cannot be decomposed into per-item correctness (it is an average of per-topic F1 scores, each computed over the whole set), so McNemar does not apply. Instead: draw one set of clause indices, score **both** models on that same resample, record the difference. Repeat 1,000 times. The reported interval is over the *difference*. **If that interval contains zero, the two models are not distinguishable.**

## I.4 Why decontaminate

ToS clauses are boilerplate. The same arbitration clause appears in fifty companies' terms with three words changed. If a near-copy of a test clause sits in the training set, the model can score it correctly by memorisation, and your test metric is measuring recall of the training set rather than generalisation.

The measurement: for each test clause, compute TF-IDF **character n-gram** cosine similarity against every training clause and keep the maximum. Character n-grams (rather than words) are used because they catch reworded and lightly edited near-duplicates that word-level matching misses.

You then report **both** numbers:
- the full test set, because recurring boilerplate is genuinely what the deployed system will see; and
- the decontaminated subset (similarity < 0.90), because that is the honest estimate of performance on genuinely novel language.

Reporting only the first overstates generalisation. Reporting only the second discards legitimate in-domain repetition. Reporting both, with intervals, is the defensible presentation and is exactly what a reviewer will ask for.

---

# Part II — Phase 1: how much of the score was contamination?

## II.1 The contamination rates were re-measured, and they moved

| Threshold | Manuscript (line 317) | Recomputed against the actual split |
|---|---|---|
| cosine ≥ 0.80 | 15.5% | **15.63%** (414 rows) |
| cosine ≥ 0.90 | 6.8% | **7.85%** (208 rows) |
| cosine ≥ 0.95 | 3.7% | **3.97%** (105 rows) |

**Why they differ, and you must say this in the text.** The original audit script (`scripts/near_duplicate_split_audit.py`) claimed to reproduce the training split but stratified on the **primary topic only**, whereas the fine-tuning notebook stratified on the composite `topic__harm` key with rare-stratum collapsing. So the audit was measuring contamination in a *similar but different* split. Its rates were indicative; the specific clauses it flagged were not the clauses the trained model actually saw.

Phase 1 fixed this by first **persisting the real seed-42 split** to `splits/split_seed42.csv` (verified: 21,183 / 2,648 / 2,648, matching `training_metadata.json` exactly) and recomputing contamination against it, reusing the audit's vectoriser settings verbatim.

This is a small correction but it is the kind of thing that, if a panelist finds it and you have not, costs you credibility. Own it in a footnote.

## II.2 The result

Decontaminated subset: **2,440 of 2,648 clauses retained (92.1%)**.

| Metric | Full test (n = 2,648) | Decontaminated (n = 2,440) | Δ |
|---|---|---|---|
| Topic macro-F1 | 0.754 [0.722, 0.774] | 0.749 [0.712, 0.770] | −0.005 |
| Topic micro-F1 | 0.834 [0.820, 0.848] | 0.831 [0.816, 0.845] | −0.003 |
| Risk accuracy | 0.829 [0.814, 0.843] | 0.829 [0.813, 0.842] | +0.000 |
| Risk macro-F1 | 0.825 [0.810, 0.839] | 0.826 [0.810, 0.840] | +0.001 |

## II.3 What it means, plainly

**The contamination was not doing the work.** Removing every test clause that had a ≥0.90 near-twin in training moved the headline scores by at most half a percentage point, and the risk head did not move at all — it moved *up* by 0.001 on macro-F1, which is noise.

Every delta is far inside its own confidence interval. The full-test and decontaminated intervals overlap almost completely. There is no detectable memorisation effect.

**This is a good result and you should present it as one.** The framing to use: *the near-duplicate audit identified a risk, the risk was quantified, and it was found not to be material.* That is a stronger position than never having looked. It also partly defuses the leave-one-document-out objection (Lippi et al. 2019 / CLAUDETTE protocol) — you can say the clause-level split was retained *and* shown empirically not to inflate the reported metrics at the 0.90 threshold.

Two caveats to state honestly:
1. It does not fully substitute for a document-grouped split, which controls for document-level style and topic co-occurrence, not just textual near-duplication. Phase 1 leaves that as unexecuted code behind `RUN_DOCUMENT_SPLIT = False`. Good news: document-identifier coverage was **verified at 100% of the 26,479 rows across 2,015 distinct groups** (CLAUDETTE `source_id`, 100 ToS `source_id`, ToS;DR `point_id → service_id`), so the experiment is runnable whenever you want it. This is a much better position than the manuscript's current "wherever document identifiers exist" hedge.
2. Risk-head insensitivity to decontamination is partly explained by the risk task being easier and more repetitive (only 3 classes); do not over-claim from it.

## II.4 Sanity check

The recomputed full-test numbers from the checkpoint (0.7540 / 0.8339 / 0.8285 / 0.8251) reproduce the manuscript's reported 0.75 / 0.83 / 0.83 / 0.83. The pipeline reproduces the reported run. Say so in one footnote — it is cheap and it forecloses a whole line of questioning.

---

# Part III — Phase 2: the encoder matrix and the head ablation

## III.1 The headline table

Mean ± sd over seeds 42, 1337, 2024. Identical splits, identical hyperparameters; only the encoder, the seed, and the active heads vary.

| Configuration | Seeds | Topic macro-F1 | Topic micro-F1 | Risk accuracy | Risk macro-F1 |
|---|---|---|---|---|---|
| Legal-BERT (dual) | 3 | 0.771 ± 0.003 | 0.834 ± 0.002 | 0.838 ± 0.002 | 0.833 ± 0.002 |
| BERT (dual) | 3 | 0.775 ± 0.006 | 0.834 ± 0.002 | 0.830 ± 0.006 | 0.824 ± 0.006 |
| XLNet (dual) | 3 | 0.762 ± 0.031 | 0.834 ± 0.007 | 0.836 ± 0.007 | 0.830 ± 0.008 |
| RoBERTa (dual) | 3 | **0.776 ± 0.006** | **0.837 ± 0.003** | **0.842 ± 0.005** | **0.836 ± 0.005** |
| Legal-BERT (topic-only) | 3 | 0.777 ± 0.009 | 0.833 ± 0.004 | — | — |
| Legal-BERT (risk-only) | 3 | — | — | 0.823 ± 0.004 | 0.819 ± 0.004 |

And the per-seed detail, which is where the story actually is:

**Topic macro-F1 by seed (dual-head)**

| Encoder | seed 42 | seed 1337 | seed 2024 | mean |
|---|---|---|---|---|
| Legal-BERT | 0.7741 | 0.7693 | 0.7703 | 0.771 |
| BERT | 0.7757 | 0.7681 | 0.7803 | 0.775 |
| XLNet | 0.7681 | **0.7283** | **0.7892** | 0.762 |
| RoBERTa | 0.7781 | 0.7803 | 0.7682 | 0.776 |

**Risk macro-F1 by seed (dual-head)**

| Encoder | seed 42 | seed 1337 | seed 2024 | mean |
|---|---|---|---|---|
| Legal-BERT | 0.8349 | 0.8329 | 0.8309 | 0.833 |
| BERT | 0.8207 | 0.8196 | 0.8312 | 0.824 |
| XLNet | 0.8274 | 0.8231 | 0.8382 | 0.830 |
| RoBERTa | 0.8323 | 0.8343 | 0.8422 | 0.836 |

## III.2 Finding 1 — the domain-pretraining claim does not survive

**The manuscript currently claims (§4.3, "Base Model") that legal-domain pre-training matters for this task. The matrix does not support that claim for the topic head.**

Read the topic macro-F1 column. The four encoders span 0.762 to 0.776 — a range of 0.014. Legal-BERT is **third of four**. Its seed-to-seed standard deviation is 0.003, and BERT's and RoBERTa's are 0.006, so a 0.004–0.005 gap between them is comfortably inside the noise. The bootstrap intervals on any single run are ±0.03 wide, an order of magnitude larger than the between-encoder gaps.

Seed-paired deltas (Legal-BERT minus the other encoder, computed seed-by-seed so training noise partially cancels):

| Comparison | topic macro-F1 per seed (42 / 1337 / 2024) | mean ± sd |
|---|---|---|
| Legal-BERT − BERT | −0.0016 / +0.0012 / −0.0099 | −0.0034 ± 0.0058 |
| Legal-BERT − XLNet | +0.0060 / +0.0410 / −0.0189 | +0.0094 ± 0.0301 |
| Legal-BERT − RoBERTa | −0.0040 / −0.0110 / +0.0021 | −0.0043 ± 0.0065 |

Every mean is smaller than its own standard deviation. **None of these differences is distinguishable from zero.**

The paired bootstrap agrees. `phase2_significance.csv`, comparing the best-validation seed of each arm:

| Comparison | topic macro-F1 Δ | 95% CI | p |
|---|---|---|---|
| Legal-BERT vs BERT | −0.0099 | [−0.031, +0.010] | 0.329 |
| Legal-BERT vs RoBERTa | +0.0021 | [−0.019, +0.021] | 0.832 |
| Legal-BERT vs XLNet | −0.0189 | [−0.037, −0.0008] | **0.033** |

Two of three intervals straddle zero. The third is discussed next.

**What you should now write instead.** Not "Legal-BERT was chosen because domain pre-training dominates" — the data refutes it. Write something closer to: *legal-domain pre-training conferred no measurable advantage over general-purpose encoders on the topic head under an identical protocol; the four encoders are statistically indistinguishable, and Legal-BERT is retained on the strength of the risk head and of the domain-adequacy argument rather than on a measured topic-classification gain.* This is a **finding**, not a failure, and it is exactly the Melis-style result Chapter 3 already set you up to report. Panelists reward this; a defended null is worth more than an undefended claim.

## III.3 Finding 2 — the XLNet question, answered carefully

The panelist's concern (Adhikari et al. 2022, 2025: XLNet > BERT on privacy-policy sentence classification) needs a precise answer, because the naive readings in either direction are both wrong.

Look at the XLNet row: **0.762 ± 0.031**. That standard deviation is *ten times* Legal-BERT's. XLNet produced both the single best topic run in the entire 18-run matrix (seed 2024, 0.7892) and the single worst (seed 1337, 0.7283). It is not better; it is **unstable**.

That instability is why the significance table is misleading if quoted alone. `compare()` in the notebook selects the **best-validation seed of each arm**. For XLNet that is seed 2024 — its lucky run. So the significant-looking Legal-BERT vs XLNet result (Δ = −0.0189, CI [−0.037, −0.0008], p = 0.033) is a comparison of Legal-BERT's typical run against XLNet's best run. Under seed-averaging the sign flips: Legal-BERT +0.0094 over XLNet on the mean.

**Both facts are true and you must report both.** The honest sentence:

> On a single favourable seed XLNet attains the highest topic macro-F1 in the matrix (0.789) and significantly exceeds Legal-BERT on paired bootstrap (Δ = 0.019, 95% CI [0.001, 0.037], p = 0.033). Averaged over three seeds, however, XLNet is the weakest encoder tested (0.762 ± 0.031) and its seed-to-seed variance is an order of magnitude greater than that of any BERT-family encoder (sd 0.031 vs 0.002–0.006). The advantage reported by Adhikari et al. is therefore not reproduced as a stable effect on this corpus; single-seed comparisons of these encoders are not informative at the magnitude of the differences involved.

That sentence concedes the panelist's point where it holds, refuses it where it does not, and demonstrates that you understand why single-run comparisons fail. It is a much stronger position than either "XLNet won" or "XLNet lost."

One caveat to keep in your pocket: XLNet may simply be more sensitive to the shared hyperparameters, which were fixed at the BERT recipe (lr 3e-5, warmup 0.06). No per-encoder tuning was done. That is a *deliberate* constant-conditions choice (Chapter 3, §3.8) — you cannot attribute a difference to the encoder if you also tuned each one differently — but state it, because it bounds the claim: you tested encoders under one recipe, not encoders at their individual best.

## III.4 Finding 3 — the head ablation is the one that works

This is the result that actually supports a design decision, and it is currently under-sold.

**Risk head, dual vs risk-only, seed-paired:**

| seed | Δ risk macro-F1 (dual − risk-only) |
|---|---|
| 42 | +0.0115 |
| 1337 | +0.0155 |
| 2024 | +0.0156 |

Positive in **all three seeds**, mean +0.014, and the effect is larger than any between-encoder difference in the matrix. Aggregate: 0.833 ± 0.002 (dual) vs 0.819 ± 0.004 (risk-only).

McNemar on item-level risk correctness (`phase2_significance.csv`, last row): b = 157, c = 122, **p = 0.042**. Dual-head is correct on 157 clauses the risk-only model gets wrong, while the reverse holds for only 122. Significant at α = 0.05.

**Topic head, dual vs topic-only, seed-paired:**

| seed | Δ topic macro-F1 (dual − topic-only) |
|---|---|
| 42 | −0.0116 |
| 1337 | −0.0095 |
| 2024 | +0.0024 |

Mean −0.006, sign inconsistent. Paired bootstrap on the best seeds: Δ = +0.0024, CI [−0.018, +0.025], p = 0.80. **No detectable effect.**

**The interpretation, which is clean:** multi-task learning helps the harder-supervised head (risk) at no measurable cost to the topic head. Topic supervision teaches the encoder distinctions the risk head can use — knowing a clause is a *limitation of liability* is informative about whether it is harmful — while the reverse transfer is negligible because the 3-class risk signal is coarse relative to the 44-way topic signal.

This directly validates the multi-task prediction of §3 (`sec:multitasklearning`) and it is now the **best-supported architectural claim in the thesis**. Promote it. The dual-head design is justified empirically; the encoder choice is not, and the honest chapter says so.

⚠️ **Two rows of `phase2_significance.csv` are artifacts — do not quote them.**
- `legal-bert...dual vs legal-bert...topic`, McNemar b = 1371, c = 144, p ≈ 3×10⁻²⁵¹. This compares the dual model's trained risk head against the topic-only model's **untrained, randomly-initialised** risk head. Of course it is significant. It measures nothing.
- `legal-bert...dual vs legal-bert...risk`, topic Δ = +0.679. Same problem, mirrored: the risk-only model has an untrained topic head (its standalone topic macro-F1 is 0.091, essentially the zero-logit floor).

Use only the *matched* half of each row: the **topic** delta for the dual-vs-topic-only comparison, and the **McNemar** for the dual-vs-risk-only comparison. Everything else in those two rows is a comparison against random weights.

Minor caveat for exactness: the risk-only arm's best-validation seed is 1337, while the dual arm's is 2024, so that McNemar is cross-seed. The seed-paired table above (all three seeds positive) is the stronger evidence; lead with it and cite McNemar as corroboration.

## III.5 Finding 4 — macro-F1 is being held down by seven topics, not by the model

From `phase2_per_topic.csv` (best Legal-BERT seed, with seed-mean F1 alongside):

| Topics included | n | macro-F1 |
|---|---|---|
| All 44 | 44 | 0.770 |
| Support ≥ 1 (drops *indemnification*) | 43 | 0.788 |
| **Support ≥ 20** | **36** | **0.828** |
| Support ≥ 50 | 31 | 0.826 |
| Support ≥ 100 | 21 | 0.823 |

The eight topics with fewer than 20 positives in the test split:

| Topic | Test support | F1 |
|---|---|---|
| indemnification | **0** | 0.000 |
| limitation_period | **1** | 1.000 |
| user_participation_in_changes | 4 | 0.167 |
| price_changes | 5 | 0.375 |
| liability_cap | 7 | 0.500 |
| severability | 8 | 0.889 |
| service_changes | 12 | 0.545 |
| logs | 19 | 0.611 |

**Read this carefully, because it is the single most useful number in Phase 2.** Restricted to the 36 topics with at least 20 test positives, macro-F1 is **0.828** — essentially equal to micro-F1 (0.834). The macro/micro gap that the manuscript currently attributes to "uneven supervision" (caption of `tab:baselineresults`) is now *quantified*: it is almost entirely produced by seven or eight topics whose test support is too small for F1 to be a meaningful statistic at all. `limitation_period` scores a perfect 1.000 on **one** test clause; `indemnification` scores 0.000 on **zero**, which is not a model failure but an undefined quantity being silently coerced to zero and then averaged in.

This does exactly what §4.4 already promised ("topics below a declared support threshold are flagged as unreliable and discussed individually instead of being averaged silently into the macro score"). You now have the threshold and the number. Use them.

Two further observations from the per-topic table worth a sentence each:

- **Two pairs of topics are degenerate duplicates — confirmed in the taxonomy, not merely suspected.** `transfer_of_contract` / `business_transfer` (both P 0.9714 / R 0.9714 / F1 0.9714 / support 35) and `transparency` / `recommender_transparency` (both P 0.7717 / R 0.7206 / F1 0.7452 / support 136) have identical metrics to four decimals *including support*. Inspecting `generated_files/lawgic_taxonomy/lawgic_topics.json` confirms why — each pair has **byte-identical `source_mappings`**:

  | Pair | `100_tos` | `tos_dr` | `claudette` |
  |---|---|---|---|
  | `transfer_of_contract`, `business_transfer` | `["transfer"]` | `["Business Transfers"]` | `[]` |
  | `transparency`, `recommender_transparency` | `["recom"]` | `["Transparency"]` | `[]` |

  The two topics in each pair have distinct names, distinct descriptions, and even distinct parent topics — but they draw from exactly the same source annotations, so they receive **exactly the same supervision on every row**. They are duplicate label columns. The model cannot distinguish them because nothing in the training signal distinguishes them, and macro-F1 counts each pair twice.

  A third pair, `contract_by_use` / `governance`, also shares identical `source_mappings` (`claudette: ["use"]`, `tos_dr: ["Governance", "User Choice"]`) but has *different* support (368 vs 337) and different metrics (F1 0.878 vs 0.881), so additional fusion logic separates them somewhere. Worth tracing, but it is not degenerate in the same way.

  **Impact, computed:** collapsing the two true duplicates gives macro-F1 **0.766** over 42 topics versus 0.770 over 44 — a −0.004 correction, and 0.826 over the 34 topics at support ≥ 20 (versus 0.828 over 36). The magnitude is small; the credibility cost of a panelist finding it first is not. **Either fix the mapping tables or state in the taxonomy section that two topic pairs are supervised identically and that the effective label set is 42.** This is a *taxonomy* finding, so it belongs in §4.2, not only in the results discussion.
- **The model is strong where supervision is strong.** Support ≥ 100 topics with F1 ≥ 0.86: `warranty_disclaimer` 0.900, `notice_of_changes` 0.910, `trackers` 0.889, `discretionary_interpretation` 0.882, `governance` 0.881, `contract_by_use` 0.878, `content_removal` 0.871, `contract_changes` 0.871, `account_termination` 0.865, `complaint_system` 0.866. Weakest well-supported topics: `personal_data` 0.669 (support 155), `third_parties` 0.730 (116), `privacy_incorporation` 0.732 (262), `interpretation_clause` 0.727 (133). The weak ones are the semantically broad ones — worth one sentence of qualitative discussion.

## III.6 Finding 5 — two protocol facts you must disclose

**(a) The epoch budget binds for 7 of 18 runs.** `epochs_run` equals the cap of 20 for `bert/2024`, `bert/42`, `legal-bert/2024`, `roberta/{42,1337,2024}`, and `xlnet/2024`, with `legal-bert/42` at 19. Those runs were stopped by the budget, not by early stopping — validation macro-F1 was still improving. They are budget-limited, not converged. This does not invalidate the comparison (the budget is identical across arms, satisfying the constant-conditions requirement) but it does mean the reported numbers are a lower bound and that the encoder ranking could shift with a longer budget. **Disclose it in one sentence.** Notably, all four seed-2024 dual runs hit the cap — which is also why seed 2024 has the highest validation scores and why the significance comparisons all landed on it.

**(b) The pooling representation changed between the shipped checkpoint and the matrix.** The v3 checkpoint fed the heads BERT's `pooler_output`; `scripts/lawgic_train_matrix.py` uses the raw first real token for all encoders (last real token for XLNet, whose summary token is appended, not prepended; selection is by attention mask because XLNet pads left and the BERT-family tokenizers pad right).

The reason is sound: `roberta-base` ships a **randomly initialised** pooler, so keeping `pooler_output` would have handicapped RoBERTa for a reason having nothing to do with the encoder — a constant-conditions violation.

The consequence you have to state: **the Legal-BERT cell of the matrix is not the same model as Table `tab:baselineresults`.** Shipped v3 checkpoint = 0.754 topic macro-F1. Matrix Legal-BERT = 0.771 ± 0.003. The +0.017 is attributable to the pooling change, not to anything else.

**This has a practical implication you should act on:** the matrix's Legal-BERT is *better than the model currently deployed in the application*, and `saved_models/lawgic_classifier_legal-bert_phase2/` (seed 2024, the best-validation run) already exists on disk. Consider swapping the deployed checkpoint, or at minimum decide deliberately not to and say why. Do not let Chapter 4 report 0.771 while the app serves 0.754.

(Note that all four `*_phase2` saved checkpoints are seed 2024. If you ever need a different seed's weights, they were not persisted — only per-run `metrics.json`, `test_logits.npz`, and `per_topic.csv` under `runs/<run_id>/`. The logits are enough to recompute any test metric or re-run any paired test without a GPU, which is the important part.)

---

# Part IV — What this means for the manuscript

## IV.1 Claims that must change

| Location | Current claim | Required change |
|---|---|---|
| §4.3 "Base Model" (line 295) | Legal-BERT's domain adaptation "matters for this task"; it "provides a clause encoder whose representations already separate legal concepts that a general encoder may fail to" | Soften to a *motivation for the choice*, and add a forward reference: the prediction was tested in §4.4 and **not confirmed** for the topic head. Do not delete the rationale — it is why you chose it a priori — but it can no longer be stated as fact. |
| §4.4 (line 385) | "the domain-pretraining claim must beat the strongest general-purpose encoders rather than BERT alone" — framed as a plan | Rewrite as a result. It did not beat them; it did not lose to them either. Report the null. |
| §4.4 (line 366) | "Table 4.x reports the preliminary training run… These figures precede the decontaminated re-scoring, the trivial baselines, the repeated runs, and the ablations committed below" | Phases 1 and 2 are done. Replace the promissory framing with results. Keep the promissory framing only for the trivial baselines (still not computed), the probes (Phase 3), and readability (Phase 4). |
| §4.3 (line 317) | 15.5% / 6.8% / 3.7% | 15.63% / 7.85% / 3.97%, with a footnote explaining that the earlier figures came from a script whose stratification did not match the training notebook's. |
| §4.3 (line 319) / §4.4 (line 392) | Document-grouped re-split "wherever document identifiers exist" | Coverage is **100% of 26,479 rows, 2,015 groups**. Either run it or state precisely why not; the "wherever they exist" hedge is no longer accurate and invites the question. |
| §4.4 (line 362) | "Topics below a declared support threshold are flagged as unreliable" | Declare the threshold: **20 test positives**. Report macro-F1 at 44 topics (0.770) and at the 36 topics above threshold (0.828). |
| §4.3 caption `tab:trainingprotocol` | "Random seed 42" | Add: three seeds (42, 1337, 2024) per configuration for the evaluation matrix; seed 42 for the shipped checkpoint. |
| §4.2 "The Lawgic Taxonomy" / "Final Corpus" | 44 substantive topics, each an independent label | Add the duplicate-supervision disclosure: two topic pairs (`transfer_of_contract`/`business_transfer`, `transparency`/`recommender_transparency`) draw from identical source mappings and are supervised identically, so the effective label set is 42. Report macro-F1 both ways (0.770 / 0.766). |

## IV.2 Claims that get *stronger*

- **The multi-task/dual-head design.** Now empirically supported: +0.014 risk macro-F1 in all three seeds, McNemar p = 0.042, no topic-head cost. Move this from "will be tested by ablation" (line 299) to a stated, defended result. It is your best architectural claim.
- **The near-duplicate audit.** You found a risk, quantified it, and showed it was immaterial (≤0.005 on every metric). Present as diligence rewarded.
- **The evaluation protocol itself.** Persisted split, 18 runs on identical clause ordering, bootstrap CIs, paired tests. Chapter 3 §3.8 already argues that constant conditions are what make an ablation valid; Chapter 4 can now say the conditions were held constant *and show the artifact that enforces it* (`splits/split_seed42.csv`, loaded by every run).

## IV.3 Suggested new tables for Chapter 4

Four tables, in this order:

1. **`tab:decontaminated-eval`** — ready to paste from `phase1_decontaminated.tex`. Replaces or supplements the current preliminary `tab:baselineresults`.
2. **`tab:encoder-matrix`** — ready to paste from `phase2_headline.tex`. The headline result.
3. **`tab:significance`** — not generated; build it by hand from `phase2_significance.csv`, **including only the matched comparisons**. Suggested content:

```latex
\begin{table}[t]
\centering
\small
\caption{Paired significance tests on the held-out test split. All runs scored the identical 2,648 clauses in identical order. Risk-head comparisons use McNemar's exact test on item-level correctness; topic-head comparisons use a paired bootstrap over 1{,}000 clause-level resamples. Arms are compared at their best-validation seed.}
\label{tab:significance}
\begin{tabular}{lrrr}
\toprule
Comparison & $\Delta$ topic macro-F1 & 95\% CI & $p$ \\
\midrule
Legal-BERT vs BERT     & $-0.010$ & $[-0.031, +0.010]$ & 0.329 \\
Legal-BERT vs RoBERTa  & $+0.002$ & $[-0.019, +0.021]$ & 0.832 \\
Legal-BERT vs XLNet    & $-0.019$ & $[-0.037, -0.001]$ & \textbf{0.033} \\
Dual-head vs topic-only & $+0.002$ & $[-0.018, +0.025]$ & 0.800 \\
\midrule
\multicolumn{4}{l}{\textit{Risk head, McNemar exact test}} \\
Dual-head vs risk-only & \multicolumn{2}{r}{$b=157$, $c=122$} & \textbf{0.042} \\
\bottomrule
\end{tabular}
\end{table}
```

4. **`tab:per-topic`** — ready to paste from `phase2_per_topic.tex`. This is a full-page table; consider moving it to Appendix D and keeping only the macro/weighted rows plus the support-threshold summary in the chapter body.

## IV.4 Draft prose you can adapt

For the encoder result:

> The encoder comparison was conducted under the constant-conditions requirement of Section \ref{sec:ablation}. Four encoders were trained under the identical protocol of Table \ref{tab:trainingprotocol}, on the identical persisted clause split, with three random seeds each, varying nothing but the encoder and the seed. Table \ref{tab:encoder-matrix} reports the outcome.
>
> The domain-pretraining prediction of Section \ref{sec:transferlearning} is not confirmed for the topic head. The four encoders span 0.762 to 0.776 topic macro-F1, and Legal-BERT ranks third. Every pairwise difference is smaller than the seed-to-seed standard deviation of the configurations being compared, and the paired-bootstrap intervals against BERT and RoBERTa both contain zero. On the risk head Legal-BERT holds a small consistent advantage over BERT (+0.009 macro-F1, positive in every seed) but is itself exceeded by RoBERTa (+0.003 in RoBERTa's favour). No encoder is reliably best on this corpus under this protocol.
>
> This null result is itself informative and is consistent with the methodological caution of \shortciteA{melis_state_2018}: reported architectural advantages frequently dissolve under equal-budget, multi-seed re-evaluation. Legal-BERT is retained for the deployed system on the strength of its risk-head stability --- it has the lowest seed variance of the four encoders on every metric (sd 0.002--0.003) --- and of the register-adequacy argument of Section \ref{sec:finetuning}, not on a measured topic-classification advantage. The measured advantage does not exist, and the manuscript does not claim it.

For the XLNet question:

> \shortciteA{adhikari_privacy_2022} report XLNet outperforming BERT-family encoders on privacy-policy sentence classification. That result is not reproduced as a stable effect here. XLNet produced both the highest single topic macro-F1 in the matrix (0.789, seed 2024) and the lowest (0.728, seed 1337); its seed-to-seed standard deviation of 0.031 is an order of magnitude larger than that of any BERT-family encoder tested (0.002--0.006). Compared at its best-validation seed, XLNet significantly exceeds Legal-BERT on the topic head ($\Delta = 0.019$, 95\% CI $[0.001, 0.037]$, $p = 0.033$); averaged over three seeds it is the weakest encoder in the matrix. The two statements are not in conflict, and reporting only the first would be a single-seed comparison of exactly the kind this matrix was constructed to avoid. Hyperparameters were held at the BERT recipe for all encoders, as the constant-conditions requirement demands, so this is a comparison of encoders under one protocol rather than of encoders at their individually tuned optima.

For the head ablation:

> The head ablation supports the multi-task prediction of Section \ref{sec:multitasklearning}. Training the risk head jointly with the topic head improves risk macro-F1 by 0.014 on average, and the improvement is positive in all three seeds (+0.012, +0.016, +0.016); McNemar's exact test on item-level risk correctness gives $b = 157$, $c = 122$, $p = 0.042$. The reverse transfer is not detectable: the topic head performs equivalently with and without the risk head ($\Delta = +0.002$, 95\% CI $[-0.018, +0.025]$, $p = 0.80$). Topic supervision therefore teaches the shared encoder distinctions the risk head exploits, at no measurable cost to topic classification. The asymmetry is expected --- a 44-way multi-label signal is far richer than a three-class one --- and the dual-head architecture is justified by it.

For the macro/micro gap:

> Macro-F1 averages each of the 44 topics equally and is therefore dominated by the low-support topics of Section \ref{sec:data}. Restricting the macro average to the 36 topics with at least 20 supervised positives in the test split raises it from 0.770 to 0.828, which is within 0.006 of micro-F1. The gap between macro and micro F1 is thus almost entirely an artifact of eight topics for which F1 is not a meaningful statistic at the available support: \textit{indemnification} has zero test positives, \textit{limitation period} has one, and six more fall below twenty. These topics are reported individually in Appendix \ref{sec:appendixcorpus} rather than treated as evidence about the model.

## IV.5 Numbers to transfer, checklist form

- [ ] Contamination rates: 15.63 / 7.85 / 3.97 % (replaces 15.5 / 6.8 / 3.7), with the stratification-mismatch footnote
- [ ] Decontaminated subset size: 2,440 of 2,648 (92.1% retained)
- [ ] `tab:decontaminated-eval` — four metrics × {full, decontaminated} with CIs and deltas
- [ ] `tab:encoder-matrix` — 6 configurations × 4 metrics, mean ± sd over 3 seeds
- [ ] `tab:significance` — hand-built, matched comparisons only
- [ ] `tab:per-topic` — 44 topics + macro + weighted, P/R/F1/support (Appendix D)
- [ ] Support-threshold macro-F1: 0.770 (all 44) → 0.828 (36 topics at support ≥ 20)
- [ ] Duplicate-topic disclosure (§4.2): two topic pairs share identical `source_mappings`; effective label set is 42, corrected macro-F1 0.766 (0.826 at support ≥ 20)
- [ ] Measured wall time: mean 60.0 min/run, range 19.2–147.9 min, 18.0 GPU-hours total, mean 15.3 epochs to early stop — the methodology section currently has **no** measured timing figure
- [ ] Epoch-cap disclosure: 7 of 18 runs stopped at the 20-epoch budget rather than by early stopping
- [ ] Pooling-deviation footnote: matrix uses first real token (last for XLNet) vs the v3 checkpoint's `pooler_output`; the Legal-BERT matrix cell is not the shipped checkpoint
- [ ] Zero-logit degenerate-model guard: 0.0923 macro-F1, verified identically in all 18 runs (currently the manuscript only asserts "< 0.95"; you now have the actual value)
- [ ] Document-identifier coverage: 100% of 26,479 rows, 2,015 groups
- [ ] Split artifact: `generated_files/lawgic_taxonomy/splits/split_seed42.csv`, verified 21,183 / 2,648 / 2,648

---

# Part V — "How did you conduct your methodology? I want to replicate it."

This is the section to hand someone who asks. It is written as a procedure, in order, with the reasoning for each choice attached — because in an evaluation methodology the *reasons* are the contribution, not the commands.

## Step 0 — Freeze the split before anything else

Everything downstream depends on this. Produce the train/validation/test assignment **once**, write it to disk with a stable row identifier and a hash of the normalised clause text, and have every subsequent experiment *load* that file rather than regenerate it.

```
generated_files/lawgic_taxonomy/splits/split_seed42.csv
  row_id, split, normalized_text_sha
  → 21,183 train / 2,648 validation / 2,648 test
```

**Why this and not "just use the same seed."** Regenerating a split from a seed is only reproducible if every input to the split function is byte-identical — the corpus row order, the pandas version, the stratification key, the rare-stratum collapsing rule. That already failed once in this project: `scripts/near_duplicate_split_audit.py` stratified on the primary topic while the training notebook stratified on the composite `topic__harm` key, so an audit that believed it was reproducing the split was measuring a different one. A persisted file cannot silently drift.

**Verify it.** Row counts must match the training run's own saved metadata (`training_metadata.json`) exactly. If they do not, stop; nothing built on top will be comparable.

## Step 1 — Put the metric code in one place, and pin it with a self-check

All metric definitions, split loading, label-array construction, bootstrap, and paired tests live in one module (`scripts/lawgic_eval_core.py`), imported by every notebook. Four copy-pasted implementations drift; one import cannot.

The bootstrap calls the topic-metric function ~1,000 times per set, so it is written vectorised (numpy TP/FP/FN) rather than looping `sklearn.f1_score` 44 times. **Optimised code is code that can be subtly wrong**, so the module carries a self-check asserting the vectorised masked-F1 equals the sklearn reference:

```
$ python scripts/lawgic_eval_core.py
self-check: vectorised masked F1 matches sklearn reference ✓
```

Run it before trusting anything. This is the cheapest possible insurance against a silent metric bug invalidating 18 GPU-hours.

## Step 2 — Guard against degenerate supervision, every run

The corpus's first fusion attempt was positive-only, and on a positive-only corpus a model that predicts *every topic present for every clause* scores a perfect macro-F1 — because a corpus with no negatives can reward nothing except saying yes.

The guard: before each fit, evaluate a model whose topic logits are identically zero. Zero passes through the sigmoid to exactly 0.5, which meets the 0.5 decision threshold, so it predicts everything present. Assert its topic macro-F1 is below 0.95.

Measured value across all 18 runs: **0.0923**, identical every time. That constancy is itself a check — it confirms every run saw the same labels and the same mask.

Quote the actual number in the manuscript, not just the threshold. "Below 0.95" is a promise; 0.0923 is a measurement.

## Step 3 — Parameterise the training loop, changing exactly one thing at a time

`scripts/lawgic_train_matrix.py` exposes `run_config(RunConfig(encoder_name, seed, heads, holdout_source))`. Everything else — learning rate, batch size, epoch cap, patience, warmup, weight decay, max length, decision threshold, loss definitions, mask handling — is fixed at the original protocol's values and is *not* a parameter. If it is not a parameter, it cannot accidentally vary.

The loss code is copied line-for-line from the original; `heads` only drops one term from the sum. That matters: if you reimplement the loss for the ablation, a difference you attribute to the missing head might be a difference in your reimplementation.

**Cross-architecture handling — the two traps.** The heads sit on a single pooled 768-d vector, and where that vector comes from differs by architecture:

- BERT / Legal-BERT / RoBERTa: the **first** real token.
- XLNet: the **last** real token. XLNet's summary token is appended, not prepended.

Both are selected **by attention mask, not by fixed index**, because XLNet's tokenizer pads on the **left** while the BERT-family tokenizers pad on the **right**. Hard-coding index 0 would silently feed XLNet a padding vector, and you would report a null result for XLNet that was actually a bug in your adapter. This is a real and common failure mode; if you replicate this, test it.

RoBERTa has no `token_type_ids`. Handled by a collator that keeps only `tokenizer.model_input_names` — one line, no `if roberta:` scattered anywhere. XLNet's tokenizer requires `sentencepiece` installed.

**A deviation from the shipped checkpoint, disclosed:** the matrix uses the raw first token for all encoders, while the shipped v3 checkpoint used BERT's `pooler_output`. Reason: `roberta-base` ships a **randomly initialised** pooler, so using `pooler_output` would handicap RoBERTa for a reason unrelated to the encoder — which is precisely the constant-conditions violation the whole design exists to prevent. Consequence: the matrix's Legal-BERT cell (0.771) is not the shipped checkpoint (0.754). State this rather than letting a reader discover the inconsistency.

**Checkpoint selection.** Early stopping on validation topic macro-F1 for dual and topic-only. The risk-only arm *must* select on `risk_macro_f1` instead, because its topic head is never trained — selecting it on topic macro-F1 would pick an arbitrary epoch. Small detail; gets the ablation wrong if missed.

## Step 4 — Run the matrix

4 encoders × 3 seeds (dual) + Legal-BERT × 3 seeds × {topic-only, risk-only} = **18 runs**.

Per run, persist: `metrics.json`, `test_logits.npz`, `per_topic.csv` under `runs/<run_id>/`.

**Persist the raw test logits.** This is the single most valuable operational decision in the whole pipeline. With logits saved for every run on an identical row ordering, every downstream analysis — bootstrap CIs, paired bootstrap, McNemar, per-topic tables, any threshold you later want to tune — is recomputable **on a laptop, in seconds, without a GPU and without retraining**. If a panelist asks for a test you did not run, you can produce it during the defense. If you save only scalar metrics, every new question costs another 18 GPU-hours.

The runner is resumable: a completed `run_id` directory is skipped. 18 hours of GPU time will be interrupted.

**Measured cost:** 18.0 GPU-hours, mean 60.0 min/run, range 19.2 (risk-only, early stop at 6 epochs) to 147.9 min (Legal-BERT seed 2024, full 20 epochs), mean 15.3 epochs.

**Known limitation to report:** 7 of 18 runs hit the 20-epoch cap rather than early-stopping, so those runs are budget-limited rather than converged. The budget is identical across arms, so the comparison remains valid — but the absolute numbers are a lower bound, and a longer budget could reorder the encoders.

## Step 5 — Aggregate: mean ± sd over seeds

For each configuration, mean and sample standard deviation (ddof = 1) of each headline metric across its three seeds.

**How to read the output.** The standard deviation is your noise floor. Compare it to the difference you care about *before* you interpret the difference. In this matrix the between-encoder differences (≤ 0.014) are the same size as or smaller than the within-configuration standard deviations (0.002–0.031). That comparison, alone, settles the encoder question — no significance test required. Do it first; it will save you from over-interpreting a p-value.

Compute seed-**paired** deltas too (encoder A minus encoder B at seed 42, at 1337, at 2024). Sign consistency across seeds is often more convincing than any single test statistic: dual-vs-risk-only is +0.012 / +0.016 / +0.016 — three for three, same direction, which is a clearer story than "p = 0.042."

## Step 6 — Bootstrap confidence intervals

1,000 resamples of the test clause indices with replacement, recomputing each metric per resample, reporting the 2.5th and 97.5th percentiles. Per run, per metric.

This quantifies **test-set** sampling noise, which is separate from and additional to the seed noise of Step 5. Both must be smaller than a difference for that difference to be real.

Watch for the macro-F1 downward bias described in §I.2 — bootstrap mean below point estimate is expected on an imbalanced label set and is not a defect.

## Step 7 — Paired significance tests

Only meaningful because every run scored identical rows in identical order. Assert it in code:

```python
assert np.array_equal(a["row_id"], b["row_id"]), "runs were scored on different rows"
```

- **Risk head → McNemar's exact test.** Per-clause correctness over `harm_mask = 1` rows. Report `b`, `c`, and the exact binomial `p`. Concordant pairs are discarded by construction — that is the point of the test.
- **Topic head → paired bootstrap.** Macro-F1 is not per-item decomposable, so McNemar does not apply. Each resample draws one index set and scores **both** models on it; the interval is over the difference. Interval contains zero ⇒ not distinguishable.

**Two ways to choose which seeds to compare, and the choice must be disclosed.** This implementation compares each arm at its **best-validation seed**. That conditions on validation performance, which is defensible (it is what you would deploy) but it flatters unstable arms — it is exactly why XLNet looks significantly better than Legal-BERT in the significance table while being worse on the three-seed mean. The alternative is fixing the seed across arms. **Report both readings.** If you report only one, report the one that disagrees with your preferred conclusion; the reader will trust the rest of the chapter more for it.

**Never compare against an untrained head.** Two rows of `phase2_significance.csv` do exactly this (dual-head's trained risk head against topic-only's random one, p ≈ 3×10⁻²⁵¹) and are meaningless. When a configuration lacks a head, only the *matched* half of the comparison is interpretable.

## Step 8 — Decontaminated re-scoring

For each test clause, maximum TF-IDF **character n-gram** cosine similarity against all training clauses. Character n-grams rather than word n-grams, because near-duplicates in ToS boilerplate are lightly *edited*, not lightly *paraphrased*, and character n-grams catch edits that word-level matching misses.

Threshold at 0.90 and report metrics on both the full test set and the subset below threshold, each with bootstrap CIs.

**Report both, always.** Full-set-only overstates generalisation on boilerplate; decontaminated-only discards legitimate in-domain repetition that the deployed system will genuinely encounter. Both, with intervals, is the only presentation that survives review.

Reuse the audit's vectoriser settings verbatim, and run it against the *persisted* split — not a regenerated one. See Step 0 for what happens otherwise.

## Step 9 — Per-topic reporting with a declared support threshold

Precision, recall, F1, and **support** for every topic. Never publish macro-F1 on an imbalanced label set without the per-topic table beside it.

Declare a support threshold in advance and report macro-F1 both ways. Here: 20 test positives, giving 0.770 (all 44) versus 0.828 (36 topics). Without the second number, a reader cannot tell whether a 0.77 macro-F1 means "mediocre everywhere" or "good on 36 topics and undefined on 8." Those are completely different systems and they produce the same headline number.

F1 on a topic with one test positive is not a statistic. Report it, flag it, exclude it from the summary claim.

## Step 10 — Things this protocol does *not* establish

Say these out loud before someone else does:

- **No per-encoder hyperparameter tuning.** All encoders ran the BERT recipe (lr 3e-5, warmup 0.06, batch 8). This is a deliberate constant-conditions choice — you cannot attribute a difference to the encoder if you also tuned each differently — but it bounds the claim to "encoders under one protocol," not "encoders at their best." XLNet in particular may be underserved.
- **Clause-level split, not document-level.** The near-duplicate audit bounds textual leakage; it does not control for document-level style or topic co-occurrence. The CLAUDETTE reference protocol (leave-one-document-out) is stricter. Document identifiers now cover 100% of rows, so this is a runnable experiment rather than an acknowledged impossibility.
- **Three seeds.** Enough to expose XLNet's instability; not enough for a tight variance estimate. Five would be better.
- **Test-set reuse.** The test split has now been scored by 18 models plus the shipped checkpoint. No model was *selected* on it — selection was on validation throughout — but the multiplicity is worth one honest sentence.
- **Label noise is unmeasured.** The ToS;DR label-noise audit is still an open TODO in `chapter_4.tex` (line ~229). Until it exists, the F1 ceiling imposed by annotation quality is unknown, and it is plausibly lower than the numbers you are reporting against.

---

# Part VI — What is left to do

## VI.1 Phase 3 — source-held-out probes (notebook written, not executed)

`notebooks/evaluation/03_source_heldout_probes.ipynb`, 13 cells, none executed.

Two probes on Legal-BERT / seed 42 / dual-head, protocol otherwise identical: hold out CLAUDETTE (3,721 long rows, 11.9% of training), hold out 100 ToS (2,048 rows, 5.6%). Each removes the source from train **and** validation and restricts the test set to that source's rows. Scoring respects the supervision masks — only cells the held-out source actually annotated are scored, otherwise the probe measures mask shape rather than comprehension.

No ToS;DR probe: it would remove ~83% of training rows, and a collapse would be confounded with data starvation. That rationale belongs in the manuscript text, not only in the notebook.

**Estimated cost:** two dual-head runs at roughly 60 min each ≈ 2 GPU-hours. Cheap relative to Phase 2 and it addresses the sharpest live objection to the fused corpus — that the model may be recognising datasets rather than reading clauses. The source-classifier probe you already have (`outputs/source_probe_cache/source_probe_results.csv`) shows a fine-tuned-encoder source classifier at 0.872 macro-F1 versus 0.575 for raw Legal-BERT, which establishes that source identity **is** strongly encoded in the fine-tuned representations. That makes Phase 3 more necessary, not less. **Run it.**

## VI.2 Phase 4 — explanation readability (notebook written, not executed)

`notebooks/evaluation/04_explanation_readability.ipynb`, 18 cells, none executed.

Requires manual setup: Ollama endpoint and credentials, `pip install py-readability-metrics`, and confirmation of access to `gemma4:31b-cloud`. Note from `CHANGES.md` §8 that the design was revised — the measured system is `lawgic-tos-changes` and the sample is not the balanced 50/50/50 the original plan called for. Read §8 before running so the manuscript describes what you actually did.

Three framing statements must appear in the eventual text: (1) which app and model, with the run date, since the endpoint is provider-hosted and can change; (2) `harm_label` is LLM-assigned, with Legal-BERT entering the prompt only as context; (3) one fixed user profile was used.

## VI.3 Still uncommitted from the manuscript's own promises

- **Trivial baselines** (§4.4 line 364): majority-class predictor and per-topic prevalence predictor. Only the zero-output floor exists (0.0923). Both remaining baselines are computable from the persisted split and cost **zero GPU time** — pure numpy on the label arrays. Do these; the chapter currently promises three floors and has one.
- **ToS;DR label-noise audit** (TODO at line ~229).
- **Document-grouped re-split** (Phase 1, behind `RUN_DOCUMENT_SPLIT = False`). Coverage is 100%; the only cost is retraining.

## VI.4 Immediate, before anything else

1. **Fix or disclose the two duplicate topic pairs.** Already verified (§III.5): `transfer_of_contract` / `business_transfer` and `transparency` / `recommender_transparency` have byte-identical `source_mappings` in `lawgic_topics.json` and therefore identical supervision on every row. Effective taxonomy is 42 topics, not 44; corrected macro-F1 is 0.766. Decide whether to merge them in the mapping tables (changes the corpus and requires retraining — expensive) or to disclose the redundancy in §4.2 and report both macro-F1 figures (cheap, and sufficient). **Recommend the second.** Also trace why `contract_by_use` / `governance` share mappings yet differ in support.
2. **Decide about the deployed checkpoint.** `saved_models/lawgic_classifier_legal-bert_phase2/` (seed 2024, first-token pooling) outperforms the shipped v3 by +0.017 topic macro-F1. Either swap it in and re-run Phase 1 against it, or keep v3 and explain in the text why the chapter's headline model is not the deployed one. Do not leave this ambiguous.
3. **Run the two trivial baselines.** Free, and the chapter already promised them.
