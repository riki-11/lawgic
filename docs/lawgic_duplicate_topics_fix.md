# Task brief: remove the duplicated topic columns from the Lawgic label set

Copy everything below the line into an agent session opened on the `Coding Projects/Thesis/lawgic` codebase. The thesis manuscript lives in a **separate** repository (`Coding Projects/Thesis/6a26daf36240f4b0d9c1e884`); manuscript edits are listed at the end and are to be applied there, not in `lawgic`.

---

You are working in the `lawgic` codebase, which contains the training and evaluation code for a thesis classifier: a dual-head model over a shared transformer encoder (`nlpaueb/legal-bert-base-uncased`), trained on a fused corpus of 26,479 Terms of Service clauses. The topic head is a 768→44 linear layer, multi-label, sigmoid activation, masked binary cross-entropy, where a source-aware supervision mask marks which topic cells are observed and unknown cells contribute zero loss. The risk head is a 768→3 linear layer, softmax, cross-entropy gated by a risk mask. The corpus was split 80/10/10 (21,183 / 2,648 / 2,648) by stratified sampling on each clause's primary active topic under seed 42, and the resulting clause-to-split assignment is persisted to a file that all runs load.

Your job is to remove two duplicated topic columns from the label set, rebuild the corpus, retrain, and re-report. The change reduces the topic head from 44 outputs to 42.

## The defect

The taxonomy defines 45 topic identifiers, one of which (`unclassified`) is excluded from training and reserved as an application-level fallback, leaving 44 predicted topics. Two pairs among those 44 draw from **identical sets of native source labels**:

| Pair | Members |
|---|---|
| A | `transfer_of_contract` and `business_transfer` |
| B | `transparency` and `recommender_system_transparency` |

Because the mapping tables send the same native labels to both members of a pair, every clause that supervises one member supervises the other with the same value and the same mask bit. As columns of the supervision matrix, the members of each pair are bit-identical. Nothing in the training signal has ever distinguished them.

### Mechanism

The topic head assigns each topic its own row of weights over the shared clause representation. Two rows trained against identical targets under identical masks receive functionally identical gradients, so they converge to the same decision rule and cross the 0.5 inference threshold together on every input. The model does not choose between the members of a pair. It emits both, always.

### Evidence already in hand

Per-topic corpus support (from the corpus profile appendix) is identical within each pair:

```
recommender transparency   1,368 positives   22,019 negatives   ToS;DR 1,341; 100 ToS 27
transparency               1,368 positives   22,019 negatives   ToS;DR 1,341; 100 ToS 27
```

Per-topic test results from the existing checkpoint are identical or near-identical within each pair:

```
                                  P       R      F1     (col4)  support
Transfer of Contract            0.971   0.971   0.971   0.972      35
Business Transfer               0.971   0.971   0.971   0.972      35
Recommender System Transparency 0.772   0.721   0.745   0.745     136
Transparency                    0.772   0.721   0.745   0.743     136
```

The third-decimal difference in pair B is seed noise from independent weight initialization, not a learned distinction.

### Why this matters, in order of importance

1. **Application correctness.** The tool tags every transparency clause with both `transparency` and `recommender_system_transparency`. Most of those 136 clauses have nothing to do with recommender systems, so the second tag is false and is displayed to a user the system is asking to calibrate trust. The risk-communication argument of the thesis depends on the tool being visibly correct. This is the reason to fix it.
2. **Metric integrity.** Topic macro-F1 averages over 44 columns when only 42 are independent. Both duplicates score above the macro, so the reported figure is inflated. The magnitude is small (see the expected numbers below) but the inflation is real and currently only disclosed in prose.

## The decision, already made

**Keep both members of each pair in the taxonomy. Predict only one member of each pair.**

The taxonomy is a design artifact with value independent of what the current corpus can supervise, and the conceptual distinction inside each pair is real even though no present source expresses it. A future source (for example a Digital Services Act-specific corpus for recommender transparency) could supply the distinguishing supervision. The dropped member therefore stays defined in the schema and is marked as not currently learnable, rather than deleted.

**Which member to drop follows a rule the thesis has already committed to** in its label-mapping policy: a native source label licenses only the taxonomy topic it reliably entails. Keep the topic the native labels actually name. Drop the topic whose presence is inferred rather than annotated.

- **Pair B is decided.** Supervision is 1,341 rows from ToS;DR against 27 from 100 ToS, and the ToS;DR label is general transparency. Keep `transparency`. Drop `recommender_system_transparency` from the prediction space.
- **Pair A is NOT decided.** Determine it from the mapping table before acting (see Phase 0). If the driving native label is ToS;DR's business-transfer category, then `business_transfer` is the named topic and `transfer_of_contract` is the inferred one, which inverts the intuitive choice. Report your finding and the resulting decision in the Phase 0 output and proceed on it. Do not guess.

### Explicitly out of scope: indemnification

`indemnification` has zero supervised positives and scores 0.000, and it stays in the prediction space. Removing it would raise topic macro-F1 from roughly 0.770 to roughly 0.788, and dropping a topic on the grounds that it scores badly is indefensible at defense. Keep it, keep reporting the zero. **Do not touch it, and do not let any "unlearnable topic" abstraction you write sweep it up along with the duplicates.** The two cases are different: a duplicate column carries no information that another column does not already carry, while indemnification carries a genuine measurement of failure.

The whole change must move the headline number **down**. If any step of your implementation raises topic macro-F1, you have done something wrong. Stop and report.

## Ground rules

- Follow the existing repo conventions for notebook naming and numbering. Add new notebooks and modules alongside existing ones.
- **Do not run training.** Write and validate the code, dry-run with tiny stubs where cheap, and stop at a clearly marked `MANUAL STEP` cell for every run that requires GPU time. Phases 0 and 1 are cheap and you may execute them.
- **Do not create a new split.** See the split hazard below. Every comparison reuses the persisted seed-42 clause-to-split assignment.
- Do not commit or push. Leave version control to me.
- Do not modify existing notebooks or data artifacts in place. Write new corpus artifacts under new filenames and leave the 44-column corpus on disk for comparison.
- Begin with a read-only exploration pass and write the repo map into your change log before touching anything.

## The split hazard, read this before Phase 2

The split is stratified on each clause's **primary active topic**, defined as the first substantive topic in the clause's stored topic list. Removing two topics from that list can change the primary active topic for any clause whose first listed topic was one of the removed ones. That would change the stratification, which would change the split, which would silently invalidate every comparison against previously reported numbers.

**Requirement: the clause-to-split assignment must not change.** Load the existing persisted split file and apply it by clause identity. After rebuilding the corpus, assert that all 26,479 clauses receive the same split assignment as before, with counts 21,183 / 2,648 / 2,648. If any clause moves, stop and report rather than proceeding.

## Phase 0: diagnosis and confirmation (cheap, execute this)

1. Map the repo: taxonomy schema file, mapping tables, fusion or corpus-builder script, the persisted split artifact, the dual-head module, training script, evaluation script, checkpoints, and where the per-topic results table is generated.
2. Read the mapping table and report, for all four topics in the two pairs, the exact set of native source labels that map to each. Confirm the pairs are identical at the mapping level and state the Pair A decision this implies.
3. Load the built 44-column corpus. Assert programmatically that the label column and the mask column for `transfer_of_contract` are element-wise identical to those for `business_transfer`, and likewise for pair B. Report the result. If they are not identical, the premise of this task is wrong; stop and report.
4. **Run a general duplicate scan, not just on the known pairs.** Compare all 44 label columns pairwise, and all 44 mask columns pairwise, and report every pair that is identical or that differs on fewer than 10 rows. There may be near-duplicates nobody has noticed. Report them; do not act on them without instruction.
5. Write the findings to `docs/duplicate_topic_columns_diagnosis.md`.

## Phase 1: evaluation-time exclusion on the existing checkpoint (cheap, execute this)

This validates the whole analysis before any retraining, and produces a number I can defend immediately.

1. Load the existing trained checkpoint and the persisted test split. Do not retrain.
2. Re-score all reported metrics with the two dropped columns excluded from the metric computation only.
3. **Expected result, use it as a check on your work:** topic macro precision ≈ 0.772, recall ≈ 0.767, F1 ≈ 0.766, against the current 44-column values of 0.777 / 0.771 / 0.770. Macro support falls from 4,930 to 4,759. These are derived from rounded table values, so tolerate ±0.003. A result outside that band means the exclusion is being applied incorrectly.
4. Report micro-F1 and both risk metrics as well, which should be almost unchanged, since micro averaging is dominated by high-support topics and the risk head is untouched.
5. Output a small table comparing 44-column and 42-column metrics side by side, and note explicitly in the markdown that the 42-column figure is the honest one.

## Phase 2: taxonomy and corpus rebuild

1. In the taxonomy schema, add an explicit field to each topic object recording whether it is in the model's prediction space, for example `"predicted": true|false`, with a `"reason"` string for every topic set to false. Set it false for the two dropped members, with the reason recording that its supervision is identical to its pair partner and naming the partner. Leave `unclassified` handled however the codebase already handles it, and document that handling.
2. Derive the model's label ordering from that field rather than hardcoding 42 anywhere. The ordering of the retained 42 must be a deterministic, persisted list, not an incidental dictionary ordering, and it must be saved to disk alongside the corpus so that evaluation and the application read the same order the model was trained on. A silent reordering between training and inference is the most likely way this task produces a subtly wrong model.
3. Rebuild the corpus through the existing fusion pipeline with the reduced label set. Write to new filenames.
4. Assert and report: 26,479 unique clauses unchanged, 50,086 long-format rows unchanged, risk distribution unchanged, per-topic supervised positives and observed negatives unchanged for all 42 retained topics. The only intended difference is the column count.
5. Apply the split hazard check described above. All 26,479 clauses must land in the same split as before.
6. Confirm that the pre-training degenerate-model assertion still exists and still fires against the new corpus shape: a zero-logit model must score topic macro-F1 below the guard threshold.
7. **Add a permanent guard** to the corpus builder: after building, assert that no two label columns are element-wise identical, and fail the build with a message naming the offending pair if any are. This defect must not be able to reappear silently.

## Phase 3: retraining (prepare only, do not run)

1. Change the topic head to 768→42, driven by the persisted label ordering rather than a literal.
2. Keep every other element of the protocol byte-identical: learning rate 3e-5, batch size, epoch cap, early stopping patience 3 on validation topic macro-F1, weight decay 0.01, warmup 0.06, FP16, max length 256, and the same persisted split.
3. Prepare configs for the same seed set used elsewhere in the project (42, 1337, 2024) so results are reported as mean ± standard deviation rather than as single-run point estimates.
4. Stop at a `MANUAL STEP` cell listing the exact runs to execute and their expected wall time, derived from the known duration of prior runs if discoverable.
5. Prepare, but do not run, the aggregation that produces: headline metrics as mean ± sd across seeds, bootstrap 95% confidence intervals over test clauses, and a per-topic table with rows = 42 topics plus macro and weighted averages, columns = precision / recall / F1 / support.

## Phase 4: outputs

1. `docs/duplicate_topic_columns_fix.md`: the mechanism, the decision and its justification, the Pair A determination with the mapping evidence, the Phase 1 numbers, the rebuild assertions and their results, and the retraining plan.
2. A `CHANGES.md` entry at repo root in the existing format: every file added or modified, rationale keyed to the phase numbers, run order, expected runtimes, and a consolidated manual-step checklist.
3. A final section listing every number that must be transferred into the thesis manuscript once runs complete.

## Manuscript edits (separate repository, do not perform, just list them accurately)

Record these in your final output with the corrected values so I can apply them. Locations are given as they stand today.

- `chapter_4.tex`, the redundancy-disclosure paragraph in Section 4.2 (currently around line 176, beginning "The mapping tables carry one redundancy"): rewrite from a disclosure of a live defect into a statement of the resolution. It should state that the duplication was found, that the affected topics remain defined in the taxonomy but are excluded from the prediction space, which member of each pair was dropped and on what rule, and that the correction lowered the reported macro-F1.
- `chapter_4.tex`, Table `tab:corpusprofile`: topic-presence dimensions 44 → 42.
- `chapter_4.tex`, Section 4.3 task formulation (around line 298): the topic vector and mask are in $\{0,1\}^{42}$, and the surrounding prose describing "44 substantive topics" must be updated.
- `chapter_4.tex`, dual-head architecture (around line 315): the topic head is a linear layer from 768 dimensions to 42 outputs.
- `chapter_4.tex`, Section 4.4 evaluation: the existing commitment to report macro-F1 over both 44 and 42 columns is now obsolete, since only 42 exist. Replace it with a single reported macro.
- `appendix_C.tex`: the per-topic support table and the per-topic results table both lose one row per pair, and their macro and weighted average rows are recomputed. Macro support becomes 4,759.
- `chapter_4.tex`, Figure `fig:corpussupport` and its caption, plus the regenerated `figures/corpus_topic_support.png`: one bar per pair is removed.
- Any prose in chapters 1, 3, or 5 stating the topic count. Grep for `44` across all `.tex` files and check each occurrence by hand, since some will be page numbers or unrelated figures.
