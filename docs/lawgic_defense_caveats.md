# Lawgic Defense Caveats

Language: ASD-STE100 Simplified Technical English.
Date: 31 July 2026.
Purpose: everything to remember before the panel. Known defects, open decisions, weak
results, and the exact words to say for each one.

---

## How to use this document

Every item has the same shape:

- **The fact.** What is true, with a measured number.
- **Say this.** The sentence to use if a panelist raises it.
- **Do not say.** The answer that will fail a follow-up.

Volunteer the items in Part 1 and Part 2. Do not wait to be asked. A defect you raise
yourself is evidence of rigour. The same defect found by a panelist is a hole.

---

# Part 1. Lead with these. They are strengths

## 1.1 The corpus rebuilds byte-for-byte

**The fact.** The notebook was executed on 31 July 2026. All seven output files are
identical to a backup taken before the run. A `shasum -a 256 -c` check passed on all six
CSV files.

**Say this.** "The corpus rebuilds byte-for-byte from the committed notebook and the
committed inputs. The pipeline is deterministic."

Few students can say this. Put it on the reproducibility slide.

## 1.2 The degenerate first attempt, and the fix

**The fact.** The first fusion produced 49,154 supervised positives and **0** supervised
negatives. A model that answers yes to every topic agreed with every cell and scored a
validation macro-F1 of **1.00**. Source-aware masking produced **919,461** negatives, and
the positive ratio is 0.0507.

The pipeline now asserts that an all-positive prediction scores below 0.95, and raises
`DEGENERATE` otherwise.

**Say this.** "The perfect score was not a success. It was proof the corpus was built
wrong. The failure is the argument for the current design, and a guard stops it returning."

## 1.3 Contamination does not explain the results

**The fact.** Removing near-duplicate clauses from the test set changes topic macro-F1 by
**-0.008**, from 0.770 to 0.762. Risk accuracy moves **+0.002**. The cleaning removed 208
of 2,648 test rows. 3.66 percent of test rows sit above cosine similarity 0.95 against the
nearest training clause.

**Say this.** "I measured contamination directly. The largest effect is 0.008. It does not
explain the result."

## 1.4 Every exclusion is logged

**The fact.** 1,810 excluded 100 ToS segments with five typed reasons, and 1,464 ToS;DR
rows with no quote text. 60 score conflicts flagged and kept.

**Say this.** "Nothing was dropped silently. Every exclusion carries a typed reason and is
written to a diagnostics table."

---

# Part 2. Known defects. Measured, not fixed

Raise these yourself.

## 2.1 The mask and the labels come from different tables

**The fact.** The topic mask is built from the raw taxonomy file. The ToS;DR labels are
built from an override table, which is narrower. So ToS;DR supervises 42 topics but can
only produce positives on 38.

Four topics receive a supervised negative on every ToS;DR row and can never receive a
ToS;DR positive:

| Topic | Positives | Negatives | Ratio |
| --- | --- | --- | --- |
| `limitation_of_liability` | 1,186 | 23,243 | 1 : 20 |
| `service_changes` | 115 | 23,272 | 1 : 202 |
| `liability_cap` | 57 | 23,330 | 1 : 409 |
| `price_changes` | 50 | 23,337 | 1 : 467 |

The correction removes **87,697** false negative cells, which is 9.5 percent of all
supervised negatives. It loses no positive.

**Say this.** "I found a defect where the mask is wider than the label space. It creates
87,697 false negatives across four topics. I measured it, the fix is written, and it costs
no positive label."

**Do not say.** "It is a small bug." It is 9.5 percent of your negative supervision.

## 2.2 CLAUDETTE clauses carry zero negatives

**The fact.** CLAUDETTE-only clauses are 3,092 of 26,479, which is **11.7 percent** of the
corpus. They carry 3,573 supervised cells, which is **0.37 percent** of all supervision.
Of those cells, **zero** are negative.

**Say this.** "CLAUDETTE clauses are 11.7 percent of my corpus and carry 0.37 percent of
the supervision. Every graded cell on them is a positive. The model pays no penalty for
over-predicting on a CLAUDETTE clause."

## 2.3 The extraction discards 91.5 percent of the CLAUDETTE corpus

**The fact.** The raw corpus holds 37,862 sentences. Only 3,221 carry a tag. The extraction
kept only those. **34,641 untagged sentences were dropped**, and 27,206 remain after the
original authors' 5-word filter.

The CLAUDETTE authors define those sentences as their negative class:

> "the negative class by all the remaining clauses" (Jablonowska et al., 2021, Section 2)

Recovering them would produce about 30,400 CLAUDETTE clauses, a corpus of about 53,700
rows, and about 272,000 real negative cells.

**Say this.** "My extraction used 8.5 percent of the CLAUDETTE corpus. The raw data holds
27,206 usable untagged sentences that the corpus authors themselves define as negatives.
Recovering them roughly doubles the corpus and adds about 272,000 real negative cells. That
is the next iteration, and the raw files are already in my repository."

**Do not say.** "CLAUDETTE only gives positives." The corpus gives negatives. Your
extraction dropped them.

## 2.4 The `a` to class action waiver expansion is not supported

**The fact.** `CLAUDETTE_TOPIC_RULES` maps `a` to both `mandatory_arbitration` and
`class_action_waiver`. The published definition says nothing about class actions:

> "The arbitration clause requires or allows the parties to resolve their disputes through
> an arbitration process, before the case could go to court."

164 of the 559 `class_action_waiver` positives, which is 29 percent, come from this
expansion. It breaks the pipeline's own rule that a label licenses only the topic it
reliably entails.

**Say this.** "I checked all nine mapping rules against the published definitions. One
fails. My arbitration expansion to class action waiver is not supported by the definition,
it affects 29 percent of that topic's positives, and I will remove it."

**Do not say.** "CLAUDETTE's arbitration category jointly covers both." That is in your
current report and the definition does not support it.

## 2.5 Five ToS;DR labels expand to three or four topics

**The fact.** `Governance` makes 4 topics from 2,476 points. `Suspension and Censorship`
makes 4 from 1,149. `Transparency` makes 3 from 1,381. `User Choice` makes 3 from 886.
`Ownership` makes 3 from 411.

ToS;DR has 22,404 points and produced 44,317 long rows. The fan-out factor is **1.98**.
These five labels made 16,231 of the 21,913 extra rows.

**Say this.** "My coarse-to-fine rule is applied to CLAUDETTE but not consistently to
ToS;DR. Five ToS;DR labels still expand to three or four topics, and the overall fan-out
factor is 1.98. I have named the narrowest entailed topic for three of the five. Two need a
legal judgment I have not yet written."

**Do not say.** "The override layer prevents fan-out." It prevents five cases and misses
five more.

## 2.6 There is a near-identical group of three, not only two pairs

**The fact.** Two column pairs are exactly identical:

- `transfer_of_contract` and `business_transfer`
- `recommender_transparency` and `transparency`

A third group is near-identical. `transparency`, `recommender_transparency`, and
`interpretation_clause` agree on 26,452 of 26,479 rows, which is **99.90 percent**. Only 27
rows from 100 ToS separate them.

`contract_by_use` and `governance` are **nested**, not identical. They differ on 366 rows.

**Say this.** "A pairwise scan of all 44 columns found two identical pairs and one
near-identical group of three. My duplicate-fix brief covers only the pairs. After the
planned fix, `interpretation_clause` remains a near-copy of `transparency`."

## 2.7 `indemnification` cannot be learned

**The fact.** It has 0 positives and 23,387 negatives. The cause is precise. ToS;DR never
uses the label `Waivers`, which is its only route. 100 ToS never uses the code `indemn`.

**Say this.** "Indemnification has zero positives because neither source ever emitted the
label. It is unlearnable from this corpus, and I report the zero rather than hide it in an
average."

**Do not say.** "The model performs poorly on indemnification." There is nothing to
perform on.

---

# Part 3. Open decisions. Not yet settled

## 3.1 Should CLAUDETTE get a source-level mask?

**The fact.** CLAUDETTE gets a per-row mask of 1 to 4 topics. ToS;DR and 100 ToS get a
source-level mask of 42 and 30. The stated justification only covers the 34 topics outside
CLAUDETTE's list. It does not explain the other 9 inside its list of 10.

The source papers allow multi-label annotation through nested tags but do **not** state
that annotators checked all nine categories on every sentence. 9.6 percent of tagged
sentences carry two or more categories.

A 10-wide mask would add 27,419 inferred negatives.

**Say this.** "The two masking regimes need a reason, and mine is only half a reason. I
could not confirm from the source papers that CLAUDETTE annotation is exhaustive per
sentence, so I did not infer negatives from tagged clauses. Recovering the untagged
sentences gives ten times more negatives with better evidence, so that is the path I chose."

## 3.2 Should `ter` and `ch` map to two topics each?

**The fact.** The definitions name the second concept directly.

> `ter`: "gives provider the right to **suspend** and/or **terminate** the service"
> `ch`: "could amend and modify the **terms of service and/or the service itself**"

Current rules keep only `account_termination` and `contract_changes`.

**Say this.** "Two of my collapses may be too narrow. The definitions name suspension and
service changes directly. The phrase 'and/or' means an individual clause may be either, so
neither child is reliably entailed alone. I state that reading, and I note the alternative."

**The uncomfortable symmetry to admit first.** You expanded the one label whose definition
does not support it, and collapsed the two whose definitions do. Say that before a panelist
does.

## 3.3 Should the model predict 44 topics or 42?

**The fact.** The taxonomy defines 44. Two pairs are unidentifiable, so 42 columns are
distinct. The model trains all 44. Macro-F1 over 44 is about 0.770. Over 42 it is about
0.766.

**Say this.** "The taxonomy defines 44 because the distinctions are real and a future
source could separate them. The model predicts 44 today. The reported average is corrected
to 42 columns. The reason to move the model to 42 is the user interface, not the score."

**Do not say.** "The 42-head model scores 0.766." It is the 44-head model scored over 42
columns. A 42-head run removes two loss terms and changes every remaining topic by an
unmeasured amount.

## 3.4 Why train 44 when the duplication was already known?

**Say this.** "Each topic head is an independent sigmoid over a shared encoder. Two heads
learning the same function do not leak, do not distort the other 42, and do not inflate any
individual topic's F1. The only thing they corrupt is the macro average, which I corrected
by reporting both counts. In exchange, two heads trained separately reaching the same score
to three decimal places turned the identifiability claim from an assertion into a
measurement. That is the same move as reporting the degenerate first fusion."

Then: "I found this in the current run, and the corrected head count is the next iteration.
What would not be defensible is knowing about the duplication and reporting only 0.770."

## 3.5 Every correction lowered the score

**The fact.** Dropping the duplicate columns moves macro-F1 from 0.770 down to 0.766.
Dropping `indemnification`, which scores zero, would move it **up** to 0.788.
`indemnification` stays.

**Say this.** "Every correction I made to the label set lowered the reported score rather
than raising it. The one change that would raise it is the one I refused to make."

---

# Part 4. Weak results. Frame them correctly

## 4.1 Legal pre-training gave no advantage

**The fact.** Topic macro-F1 over 3 seeds: RoBERTa 0.776, BERT 0.775, Legal-BERT 0.771,
XLNet 0.762. All four are within 0.014. Paired tests against Legal-BERT: BERT p = 0.329,
RoBERTa p = 0.832, XLNet p = 0.033.

XLNet has a standard deviation of 0.031, which is ten times Legal-BERT's 0.003. It produced
both the best and the worst single run.

**Say this.** "Legal-domain pre-training conferred no measurable advantage over
general-purpose encoders on the topic head under an identical protocol. The four encoders
are statistically indistinguishable. Legal-BERT is retained on the strength of the risk
head and the domain-adequacy argument, not on a measured topic gain. This is a finding, not
a failure."

**Do not say.** "Legal-BERT was chosen because domain pre-training dominates." Your data
refutes it.

## 4.2 The dual head helps one task, not both

**The fact.** Risk head, dual minus risk-only, risk macro-F1: Legal-BERT **+0.014**, CI
[0.008, 0.020], all three seeds positive. RoBERTa +0.023, XLNet +0.012, BERT +0.012.

Topic head, dual minus topic-only: BERT +0.028, but Legal-BERT **-0.006**, RoBERTa -0.003,
XLNet -0.013. Three of four lose.

**Say this.** "The second head improves the risk task, with a confidence interval that
excludes zero and agreement across all three seeds. It does not improve the topic task. I
claim that the dual head improves risk classification and costs nothing on topic
classification."

**Do not say.** "The dual head improves both tasks."

## 4.3 The readability result is negative

**The fact.** Across 87 paired changes, Flesch Reading Ease falls from 37.97 to 34.54,
Wilcoxon p = 0.040, rank-biserial -0.25. Only 36.8 percent of explanations read easier.
Grade level rises from 13.69 to 14.26, p = 0.143, not significant.

Two facts must accompany it:

1. The Pravasi and Das instrument returned **zero** scorable pairs.
   `py-readability-metrics` needs 100 words. The median change quotes 35 words and produces
   56.
2. Legal-BERT does not write the text. `gemma4:31b-cloud` does. The classifier supplies
   advisory context only.

**Say this.** "The pooled result does not support the readability claim as stated. I report
it as a negative result. The strict replication was also impossible, because the instrument
requires 100 words and my unit of analysis produces 56."

## 4.4 The source hold-out shows partial transfer only

**The fact.** With CLAUDETTE held out, topic macro-F1 retains 76 percent, micro precision
retains 99 percent, and micro recall falls to 58 percent. With 100 ToS held out, macro-F1
retains 50 percent and micro precision retains 34 percent. Test rows: 324 and 141.

**Say this.** "The probe shows partial transfer. Performance falls but does not collapse,
and on CLAUDETTE the precision holds while the recall drops, so the model becomes cautious
rather than wrong. It does not prove the model ignores source identity."

**Do not say.** "The probe proves the model reads the text, not the source."

## 4.5 Low-support topics cannot carry claims

**The fact.** Test support: `indemnification` 0, `price_changes` 5, `liability_cap` 7,
`severability` 8. `severability` scores F1 0.889 on 8 examples. Corpus positives:
`indemnification` 0, `limitation_period` 14, `user_participation_in_changes` 44.

**Say this.** "Per-topic F1 is reported with support beside it, and low-support topics are
flagged. A topic with eight test examples cannot carry a claim in either direction."

---

# Part 5. Errors in current artifacts. Fix before the defense

| Artifact | Error | Correct value | Status |
| --- | --- | --- | --- |
| `docs/lawgic_dataset_report.md` §9.1 | Coverage 37 and 26 | 42 and 30 | Corrected 31 July |
| `docs/lawgic_dual_head_architecture.md` | Coverage 37 and 26 | 42 and 30 | Corrected 31 July |
| Slide 50 of the PDF deck | 100 ToS has 9 categories | **24** | Open |
| Slide 53 footer | Lists 1-2, 42, 30 as one set | Split into two lines | Open |
| `docs/lawgic_duplicate_topics_fix.md` | Lists two pairs | Add the group of three | Open |
| Any slide saying "contributed" | Wrong verb | "can answer" or "covers" | Open |

---

# Part 6. Likely questions and short answers

**"How did 9 CLAUDETTE categories become 44 topics?"**
The taxonomy is not a union of source labels. It was designed finer than any source, then
the native labels were pointed at it. All three sources overlap heavily, and some single
labels split into five topics. The mapping table is the justification, not arithmetic.

**"What is CLAUDETTE's coverage?"**
The mapping table lists 21. The resolution rule keeps 10. Neither is used as a mask.
CLAUDETTE gets a per-row mask of 1 to 4 topics, keyed to the label on that row.

**"Why do ToS;DR and CLAUDETTE get different masks?"**
ToS;DR has a wide vocabulary, so its silence inside that vocabulary is credible evidence of
absence. I chose not to treat CLAUDETTE's silence as evidence at all. That choice is more
conservative than it needs to be, and I have measured the alternative.

**"Where do the negatives come from?"**
Almost entirely from ToS;DR and 100 ToS. CLAUDETTE contributes none. That is defect 2.2.

**"Did you check for other duplicate columns?"**
Yes. A pairwise scan of all 44 label columns and all 44 mask columns. Two identical pairs,
one near-identical group of three, and one nested pair.

**"Is the duplicate finding from the cosine-similarity audit?"**
No. Different axis. The near-duplicate audit compares clause **text** across the train and
test splits, to detect contamination. The duplicate topics are **columns** of labels, and
they are provable from the mapping table without running any model.

**"Only 109 clauses have more than one source. Why fuse at all?"**
Fusion is not mainly about cross-checking. Only 0.4 percent of clauses overlap. The value
is coverage: no single source spans the label space, and the coverage mask pulls dozens of
real negatives out of every ToS;DR and 100 ToS row.

**"Who built the taxonomy?"**
One researcher. A legal professional will review the full mapping table and a sample of
rows, and the agreement rate will be reported. CLAUDETTE's own inter-annotator agreement is
Cohen's kappa 0.871.

**"Is `blocker` really harmful?"**
It is a severity flag, not a rubric grade. I map it to -1 and tag every row
`blocker_as_bad`, so the policy is auditable and reversible.

**"Does the pessimistic conflict rule make the tool alarmist?"**
It fires on 60 of 50,086 rows, which is 0.12 percent. A change that small cannot move what
the model learns. Whether users over-trust the tool is measured in the user study, not
argued here.

---

# Part 7. Sentences to memorise

1. "The corpus is 26,479 clauses fused from three sources, carrying 44 topic labels and a
   three-class risk label."
2. "The first fusion had 49,154 positives and zero negatives, and a model answering yes to
   everything scored a perfect 1.00. Source-aware masking produced 919,461 negatives."
3. "Legal-BERT reaches topic macro-F1 0.771 plus or minus 0.003, and the four encoders are
   within 0.014, so legal pre-training gave no measurable topic advantage."
4. "Removing near-duplicate clauses from the test set moves macro-F1 by 0.008, so
   contamination does not explain the result."
5. "The corpus rebuilds byte-for-byte from the committed notebook."
6. "I audited my own preparation and found four open defects. I measured every one of them,
   and none of them invalidates what the corpus recorded."

---

# Part 8. The closing position

If a panelist presses hard on the defects, this is the frame:

"What I am presenting is the progression of the work, not a polished output with its
history removed. The mapping table predicted the duplicate columns and the training run
confirmed them. The first fusion failed and the failure is why the design is what it is. I
audited my own mapping rules against the published definitions and found one of my own
rules wrong. I audited my extraction against the raw corpus and found that I had discarded
91.5 percent of it. Each of those is measured, each is written down, and each has a fix
scheduled. What would not be defensible is presenting a single clean number with none of
this attached."
