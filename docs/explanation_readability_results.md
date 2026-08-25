# Explanation Readability Results — Interpretation and Manuscript Guidance

**Date:** 2026-07-27
**Scope:** Phase 4 of the statistical-rigor upgrade. Executed on the MacBook (Apple Silicon, `mps`); no GPU required.
**Notebook:** `notebooks/evaluation/04_explanation_readability.ipynb`
**Companions:** `docs/multiseed_evaluation_results.md` (Phases 1–2), `docs/source_heldout_probes_results.md` (Phase 3).
**Artifacts:** `generated_files/lawgic_taxonomy/evaluation/phase4_{changes.jsonl, readability_pairs.csv, readability_analysis.csv, readability_summary.{csv,tex}, readability.png}`

> ### ⏸ Manuscript status (2026-07-27): results withheld from the proposal defense
>
> **Part VI of this document is on hold. Do not apply it yet.** No number from this run enters the proposal manuscript. §4.4.6 of `chapter_4.tex` instead states the *protocol* in full — unit of analysis, field mapping, instrument and its 100-word floor, document-level fallback, test-selection rule, and three pre-declared stratifications — and invites the panel to critique it before it is committed.
>
> **The reason is not that the result is bad.** It is that the generative layer and its prompt are still under revision, so a readability number reported now would describe a system that will not be the one defended. Reporting the protocol and asking for feedback is the correct move at proposal stage regardless of what this run said — and the panel chair has published on exactly this instrument.
>
> What that means for this document: Parts I–V, VII and VIII stand as the analysis of record and as the specification of the protocol. **Part VI is the reinstatement plan, not a to-do list for now.** The decision, the exact manuscript edits, and a reinstatement checklist are recorded in **Appendix 2** of `misc/das_defense_prep.md` in the manuscript repository. A `TODO (post-defense)` block in `chapter_4.tex` (~line 490) marks where the results come back.
>
> **One consequence worth knowing.** Because the proposal now pre-declares the stratification by source-excerpt readability (Part III), that analysis stops being post hoc. When the results are eventually reported, the banded finding can be presented as confirmatory rather than exploratory, which answers the regression-to-the-mean objection of III.3 far better than any argument in this document does. Writing the protocol out was worth it for that alone.
>
> **Nothing in the code was reverted.** The notebook, the generated changes, the scored pairs, and the figure all remain. Note however that reinstatement should mean a **re-run**, not a paste: if the prompt changes, this run is stale by the same argument that justified withholding it.

---

## 0. The one-paragraph version

The application claims it makes Terms of Service readable. Measured across 87 generated changes from three ToS version pairs, **the pooled result does not support that claim as stated**: mean Flesch Reading Ease falls from 38.0 to 34.5 (Wilcoxon, p = 0.040), grade level rises from 13.7 to 14.3 (p = 0.143, not significant), and only 37% of individual changes score easier than the text they explain. Taken at face value, that is a negative result.

But the pooled number is an average over two populations that behave in opposite directions, and separating them is the actual finding. **On genuinely dense legalese — source Reading Ease below 20, which is where the readability problem lives — the explanations improve readability enormously and almost without exception: +22.4 Reading Ease, −5.8 grade levels, 95% of changes improve, rank-biserial 0.99.** On source text that was already plain (Reading Ease above 60 — largely Google's own plain-English summary page, which the YouTube case study includes verbatim), the explanations score *worse*: −22.5 Reading Ease, 0% improve. The system is not a monotone simplifier. It is a **leveler**: it emits prose at a stable reading level (Reading Ease ≈ 30–43, grade 12–15) almost regardless of what it is given. That is a defensible and more interesting claim than the one the thesis currently makes, and it is what the manuscript should say.

A second finding is methodological and must be reported: **the Pravasi & Das instrument returned zero scorable pairs.** `py-readability-metrics` requires 100 words; the median change quotes 35 words of ToS and produces 56 words of explanation. A strict replication of their protocol is impossible at the unit of analysis this application actually produces.

---

# Part I — What was run, and why in exactly this way

This part is the replication guide. If someone asks "how did you conduct this, I want to replicate it," this is the answer, in order.

## I.1 The claim being tested

Every prior phase measured the **classifier** — can Legal-BERT identify clause topics and risk levels. None of them measured the thing the application is actually sold on: that a user confronted with a Terms of Service update can understand what changed. That claim had been asserted throughout the manuscript and never once measured.

The evaluation approach is borrowed from **Pravasi & Das (2024)**, who assessed ChatGPT's interpretations of privacy policies by scoring them with Flesch readability metrics against the source policies. The logic is a paired comparison: take the legalese, take the machine-generated explanation of that same legalese, score both, and test whether the explanation is easier.

## I.2 Why the unit of analysis is a *change*, not a *clause*

This is the first decision a replicator has to get right, and it is not obvious.

The thesis artifact is **`lawgic-tos-changes`**, not the deprecated `lawgic-web-app`. That distinction changes the design completely:

- `lawgic-web-app` explained **one clause at a time** through the FastAPI `/api/explain_tos_scores` endpoint. A clause in, an explanation out. Pairing is trivial.
- `lawgic-tos-changes` has no per-clause explain flow at all. It diffs **two versions** of a document and emits a list of *changes*, each of which carries its own plain-language fields.

So the paired unit is one **change**:

| Side of the pair | Field(s) | What it is |
|---|---|---|
| Source (legalese) | `new_text`, falling back to `old_text` when the change is a removal | The excerpt the generator quotes verbatim from the real ToS, capped at 400 characters by the prompt schema |
| Explanation (plain language) | `what_changed` + `impact_for_user`, joined | Exactly the prose a user reads on the change card |

`action_needed` is deliberately **excluded** from the explanation side. It is an imperative instruction ("Review your monetization settings"), not an explanation of the clause, and its short command sentences would inflate the readability score without any of that improvement corresponding to comprehension of the legal text. If you include it, you are measuring a different thing and your numbers will be optimistically biased.

The 400-character cap on the source side is imposed by the application's own prompt, not by the evaluation. That cap turns out to matter enormously — see I.6 and II.1.

## I.3 Why drive the live application instead of reimplementing the prompt

The notebook does **not** re-implement `buildChunkPrompt()`. It starts the real Next.js application and drives its own HTTP endpoints:

```
POST /api/analyze        (×2, old and new document)  → Legal-BERT clause predictions
POST /api/diff/prepare                                → aligned section pairs
POST /api/diff/chunk     (× number of pairs)          → the generated changes
```

The reason is fidelity. A reimplementation drifts: you copy the system prompt but miss the `cleanJsonContent()` fence handling, or the retry-once-on-`SyntaxError` behaviour, or the exact `/v1/chat/completions` parameters. Then you have measured a system that does not exist. Driving the app means `lib/ollama-diff.js` executes byte for byte as shipped, and any result you report is a result about the artifact you are defending.

**Replication cost of this choice:** you must have the app, the classifier backend, and Ollama all running simultaneously before the notebook does anything. The preflight cell hard-fails if any of the three is missing.

## I.4 Two orchestration details that will silently corrupt the sample if you skip them

Both are ported from `pages/index.js`, and both are easy to miss:

1. **The Legal-BERT context reaches only the first chunk.** `index.js:146` passes `lawgicClauses` when `batchStart + indexInBatch === 0` and `null` otherwise. If you feed the classifier context to every chunk, you have evaluated a *different system* than the one users run — a better-informed one — and your readability numbers describe software that was never shipped.

2. **The change list is deduplicated client-side** by `mergeAndDedupeChanges()` before the user ever sees it. Sections overlap, so the same change is frequently found twice. Scoring the raw per-chunk output double-counts those, which inflates `n`, breaks the independence assumption of the paired test, and weights duplicated changes more heavily in the mean. The notebook ports that function to Python — same normalisation, same >2-character word filter, same Jaccard threshold of 0.82, same harm-then-type sort — with a self-check asserting that exact duplicates collapse, near-duplicates above threshold collapse, and harmful sorts before neutral before fair.

Raw output across the three services was materially larger than the 87 changes scored. The dedupe is doing real work.

## I.5 Which classifier, and the honest statement of how much it matters

The backend served **`saved_models/lawgic_classifier_legal-bert_phase2`** on `mps` — the best-of-3-seeds dual-head model selected in `02_multiseed_encoder_runs.ipynb` (`nlpaueb/legal-bert-base-uncased`, seed 2024, val metric 0.794). It outperforms the `_v3` checkpoint the API previously hardcoded, on both heads:

| Checkpoint | Topic macro-F1 | Harm macro-F1 |
|---|---|---|
| `_v3` (seed 42) | 0.754 | 0.825 |
| `_phase2` (seed 2024, best of 3) | **0.770** | **0.831** |

The two are architecturally identical — verified: both `model_state_dict.pt` files carry the same 203 keys with identical head shapes, the same 44-topic map, the same three harm classes. It is a checkpoint swap, not a pipeline change. Selection is by `LAWGIC_MODEL_DIR` at server start; the preflight cell reads `GET /api/model-info` and aborts if the backend is serving anything else; every output row records `classifier_model` and `classifier_device`.

**Now the honest part, and it must go in the manuscript.** Legal-BERT does not write the text being scored. Its predictions enter the prompt as advisory context via `buildLawgicContext`, truncated to 2,000 characters per version, and only on the first of 5–11 chunks. The readability measured here is a property of **`gemma4:31b-cloud`'s prose**, not of the classifier.

The justification for using the better checkpoint is therefore *consistency*, not causation: without it, Chapter 4 would report a classifier that Chapter 5 does not use, and the manuscript would be describing two different systems. Do **not** write "the improved classifier produced more readable explanations." There is no A/B, and the causal path is far too thin to support it.

**Hardware note for replicators:** nothing trains here. Legal-BERT runs inference only, and `mps` (or `cpu` via `LAWGIC_DEVICE`) produces the same predictions as the CUDA training machine — device affects speed, not outputs. The generator runs on Ollama's cloud. A MacBook is sufficient, and CUDA buys nothing.

## I.6 The instrument: what Flesch actually measures

Both metrics are computed from exactly two quantities. This is the single most important thing to understand before interpreting any number below.

```
words_per_sentence  (WPS)  =  total words / total sentences
syllables_per_word  (SPW)  =  total syllables / total words

Flesch Reading Ease  =  206.835  −  1.015 × WPS  −  84.6 × SPW      (higher = easier)
Flesch–Kincaid Grade =    0.39  × WPS  +  11.8  × SPW  −  15.59     (lower  = easier)
```

That is the whole instrument. Flesch has **no** notion of vocabulary difficulty, jargon, syntactic nesting, ambiguity, or whether the text is true. It counts sentence length and word length. A text of short sentences made of short words scores as "easy" even if it is nonsense; a text explaining something clearly scores as "hard" if the clear explanation requires long words.

Note the coefficients: Reading Ease weights syllables-per-word 83× more heavily than words-per-sentence. Grade Level weights it 30× more heavily. **Both metrics are dominated by average word length.** Keep that in mind for II.3.

### The 100-word gate

`py-readability-metrics` — the library Pravasi & Das used, and therefore the primary instrument here — raises `ReadabilityException` on any text under 100 words. This is not arbitrary pedantry: on a three-sentence text, one unusually long sentence swings words-per-sentence by 40%, and the resulting score is noise.

The notebook therefore reports **two clearly separated column sets**:

- `*_lib` — the library. The Pravasi & Das replication proper.
- `*_raw` — the same two formulas recomputed inline with no length gate, using a vowel-group syllable heuristic.

These are not interchangeable and the manuscript must never present them as the same measurement. The heuristic syllable counter will disagree with the library's dictionary-backed one on irregular words. The `_raw` columns are a **robustness check**, cited as such.

## I.7 Why a paired test, and why the test was chosen by the data

**Why paired.** Each explanation is generated *from* a specific clause. Clause difficulty varies enormously across the corpus (Reading Ease −10 to +83 in this sample). An unpaired comparison of "all clauses" against "all explanations" would drown the effect in that between-clause variance. Pairing removes it: the test is on the 87 *differences*, each one an explanation compared against the specific text it explains. This is the same pairing logic used for the multi-seed comparisons in Phase 2.

**Why the test is chosen, not fixed.** The notebook runs Shapiro–Wilk on the paired differences first:

- Normality not rejected at α = 0.05 → **paired t-test**, effect reported as **Cohen's d for paired samples**.
- Normality rejected → **Wilcoxon signed-rank**, effect reported as **matched-pairs rank-biserial correlation**.

Choosing the test by an assumption check rather than by preference is what makes the result defensible under questioning. The pooled Reading Ease differences were non-normal, so the headline number is a Wilcoxon.

**How to read rank-biserial.** It runs from −1 to +1 and answers: of all the ranked differences, what is the net balance of positive over negative? +1.00 means every single pair moved in the positive direction. 0 means the positive and negative movements are perfectly balanced by rank. It is the non-parametric cousin of Cohen's d, and unlike d it is not distorted by a few extreme values. Conventional (loose) reading: 0.1 small, 0.3 medium, 0.5 large.

**`proportion_improved`** is reported alongside every test and is the number to quote to a non-statistical audience. It is direction-aware: for Reading Ease, "improved" means the difference is positive; for Grade Level, negative. A p-value tells you the effect is unlikely to be chance; `proportion_improved` tells you how often it actually happens to a user.

## I.8 The user profile is an input variable, not a constant

`buildChunkPrompt` personalises `impact_for_user` to the reader's role and concerns. That means the profile is part of the system's input, and a readability number is conditional on it. One profile was fixed for the entire run and recorded on every row:

```
roleLabel : Content Creator
concerns  : ["I'm monetized on this platform", "I post original music or audio"]
context   : "I run a monetized channel and post original music."
```

`creator` was chosen because it has the most concern options in `lib/constants.js` (five), which exercises the personalisation path rather than leaving it mostly empty. **Readability may differ by role**, and a single-profile design cannot detect that. State it as a limitation; do not imply the result is role-invariant.

---

# Part II — The pooled numbers, and how to read each one

## II.1 The library rows are empty — and that is a finding, not a bug

```
Metric                              n     Test
Flesch Reading Ease (library)       0     insufficient data
Flesch-Kincaid Grade (library)      0     insufficient data
```

**Zero of 87 pairs cleared the 100-word gate.** Not a few — none.

The reason is structural, and it is worth stating precisely because it generalises to anyone trying to replicate Pravasi & Das on a change-level tool:

| | median | 25th–75th | max |
|---|---|---|---|
| Clause (quoted ToS excerpt) | 35 words | 26–43 | 73 |
| Explanation | 56 words | 50–66 | 94 |

The clause side is capped at 400 characters by the application's own prompt schema. The explanation side is two or three sentences by design. Even adding `action_needed` back in — which would be methodologically wrong — lifts only **7 of 87** pairs over 100 words.

**Pravasi & Das scored whole privacy policies against whole ChatGPT interpretations.** Their unit was a document of thousands of words. This application's unit is a change card of fifty. The instrument was built for their unit and does not survive transfer to this one. That is a real, citable methodological result about applying document-level readability instruments to change-level interfaces, and the manuscript should present it as one rather than quietly dropping the library columns.

Everything from here on uses the ungated `_raw` columns, and must be cited as a robustness check computed with the same formulas — never as a Pravasi & Das replication.

## II.2 The pooled result

```
Metric                          n    Clause  Explanation  Diff   Test        p       Effect          Improved
Flesch Reading Ease (all)      87     37.97       34.54  -3.43   Wilcoxon    0.040   -0.25 (rb)      36.8%
Flesch-Kincaid Grade (all)     87     13.69       14.26  +0.57   Wilcoxon    0.143   +0.18 (rb)      41.4%
```

Read literally, left to right:

- **Reading Ease fell by 3.4 points.** The explanations score marginally *harder* than the excerpts they explain. p = 0.040 clears α = 0.05; rank-biserial −0.25 is a small effect.
- **Grade level rose by 0.6.** Not statistically significant (p = 0.143). The two metrics agree in *direction* — both say slightly harder — but only one clears significance.
- **Only 36.8% of individual changes scored easier.** The majority scored harder.
- **Both texts sit at roughly grade 13–14** — first- and second-year university. Neither is remotely accessible to a general audience.

### Three reasons not to over-read this

1. **The effect is tiny in absolute terms.** 3.4 Reading Ease points is well inside the band where both texts carry the same qualitative label ("difficult"). Nobody experiences a 3-point Flesch difference.

2. **p = 0.040 is fragile.** `phase4_readability_analysis.csv` contains 4 metrics × 7 strata of tests, run at an uncorrected α = 0.05. Under any standard multiple-comparison correction, **p = 0.040 does not survive.** Do not present the pooled negative as a robust finding; it is the weakest significant result in the table.

3. **It averages two opposite populations.** This is the real problem, and it is Part III.

## II.3 Where the −3.4 actually comes from (decompose before you interpret)

This is a diagnostic step every replicator should perform, and it takes one line of arithmetic. Recover the two Flesch inputs for each side:

| | Clause | Explanation | Δ |
|---|---|---|---|
| Words per sentence | 23.05 | 23.41 | +0.36 |
| Syllables per word | 1.720 | 1.756 | +0.036 |

Feed those back through the formula:

```
ΔFRE  =  −1.015 × (+0.36)  −  84.6 × (+0.036)
      =  −0.37             −  3.05
      =  −3.42                              ← matches the observed −3.43
```

**89% of the pooled negative result is the syllables-per-word term.** Sentence length is effectively unchanged. The explanations are not more convoluted, do not use longer sentences, and are not structurally denser. They use *marginally longer words* — an extra 0.036 syllables per word, about one extra syllable every 28 words.

Why would a plain-language explanation use longer words than the legalese? Because it has to **name the concept**. The quoted excerpt is often a mid-sentence fragment ("(iv) attempt to circumvent, manipulate, or disable systems and Services…"), while the explanation must supply the noun the user needs: *monetization*, *termination*, *indemnification*, *intellectual property*, *arbitration*. Every one of those is polysyllabic, and every one of them is what makes the explanation useful.

**This is a known and documented failure mode of Flesch as a comprehension proxy.** The formula charges you for naming the thing you are explaining. State this in the manuscript; it is the single most important caveat on the pooled number.

---

# Part III — The finding the pooled table hides

## III.1 Stratifying by how hard the source text was

The pooled test asks "are explanations easier than sources on average." A more useful question is "easier than *what kind of* source." Banding the 87 pairs by the Reading Ease of the clause they explain:

### Flesch Reading Ease (higher = easier)

| Source band | n | Clause | Explanation | Δ | p | rank-biserial | Improved |
|---|---|---|---|---|---|---|---|
| Very dense (<20) | 20 | 8.3 | 30.7 | **+22.4** | <0.0001 | **+0.99** | **95%** |
| Dense (20–40) | 23 | 33.3 | 29.3 | −4.0 | 0.065 | −0.44 | 35% |
| Moderate (40–60) | 33 | 50.0 | 37.7 | −12.3 | <0.0001 | −0.77 | 15% |
| Already plain (>60) | 11 | 65.6 | 43.0 | **−22.5** | 0.0010 | **−1.00** | **0%** |
| *Pooled* | *87* | *38.0* | *34.5* | *−3.4* | *0.040* | *−0.25* | *37%* |

### Flesch–Kincaid Grade (lower = easier)

| Source band | n | Clause | Explanation | Δ | p | rank-biserial | Improved |
|---|---|---|---|---|---|---|---|
| Very dense (<20) | 20 | 20.6 | 14.8 | **−5.8** | <0.0001 | −0.99 | **95%** |
| Dense (20–40) | 23 | 14.8 | 15.5 | +0.7 | 0.247 | +0.28 | 43% |
| Moderate (40–60) | 33 | 10.8 | 13.6 | +2.8 | 0.0001 | +0.73 | 21% |
| Already plain (>60) | 11 | 7.3 | 12.4 | **+5.1** | 0.0010 | +1.00 | **0%** |

The two metrics agree perfectly, band for band, which is a good sign that the pattern is not an artefact of one formula.

## III.2 What this means: the system is a leveler, not a simplifier

Look down the **Explanation** column in either table. As the source ranges over 57 Reading Ease points (8.3 → 65.6), the output moves only 12 points (30.7 → 43.0). Grade level: source spans 20.6 → 7.3 (13 grades), output spans 14.8 → 12.4 (2.4 grades).

The dispersion statistics say the same thing:

```
Clause Reading Ease      sd = 20.2   range −10.2 … 82.7
Explanation Reading Ease sd = 11.8   range  −2.0 … 66.3     variance ratio 2.96
```

The generator **compresses a wide input distribution onto a narrow output distribution centred around grade 13–14.** Carry-through slope from source to explanation is roughly 0.22 — mostly leveling, with slight residual echo of the source.

That has a direct and useful consequence:

- Where the source is **worse** than the target level, the system improves it — dramatically, and near-universally (95% of cases, rank-biserial 0.99, which is about as close to "always" as a rank statistic gets).
- Where the source is **already better** than the target level, the system degrades it — also near-universally (0% improved, rank-biserial −1.00).

**The pooled −3.4 is not a measurement of the system's quality. It is a measurement of what fraction of this particular corpus was already plain.** Change the corpus mix and the pooled number moves anywhere you like, without the system changing at all.

## III.3 The honest caveat: regression to the mean

**A replicator will raise this, and a panelist should.** The bands are defined on the clause score, which is one of the two terms in the difference being tested. That is textbook regression-to-the-mean setup: even with two *random* variables, items selected for an extreme value on the first will, on average, be less extreme on the second, manufacturing exactly this gradient. The correlation between clause score and improvement (r = −0.83) is partly mechanical for the same reason and should **not** be quoted as evidence.

Three arguments that the effect is nonetheless real, none of which depend on the banding:

1. **Clause and explanation are not repeated measures of one quantity.** Regression to the mean is a property of noisy repeated measurement of the *same* underlying value. These are two different texts written by different authors (a lawyer and a language model) about the same subject. There is no measurement-error model under which the noise term alone shrinks a 93-point input range onto a 68-point output range clustered on one value.

2. **The variance reduction is a fact about the marginal distributions, computed without reference to any banding.** sd 20.2 → 11.8, F = 2.96. That number exists whether or not you stratify.

3. **The qualitative reading confirms the mechanism** (III.4). The high-scoring "sources" in the top band are not noisy measurements of legalese; they are genuinely plain English that happens to live inside a ToS file.

**Recommended framing, and this matters for defensibility:** the pooled test is the **pre-specified, confirmatory** analysis and must be reported as the headline, negative and all. The banded analysis is **post hoc and exploratory** and must be labelled as such in the manuscript. Reported that way it is a strong contribution. Reported as if it had been the plan all along, it is p-hacking and a panelist will find it.

## III.4 What the extremes actually look like

Reading the texts is not optional. The numbers are uninterpretable without them.

**System working as intended** — dense legalese, Reading Ease −8.1 → 47.8 (a *negative* Reading Ease means unreadable by the scale's own construction):

> **Clause:** "Represent, imply or otherwise create an impression that your Output is human-generated or otherwise generated without the use of AI, including by removing, obscuring, or altering any watermarks, content-authenticating metadata"
>
> **Explanation:** "It is now explicitly forbidden to hide the fact that content was generated by AI or to remove AI watermarks/metadata. Since you post original music, if you use AI to assist in composition or production via TikTok's tools, you cannot claim it is 100% human-made or strip the AI labels from the file."

The explanation is unambiguously more usable, and the metric agrees by 56 points.

**System scoring "worse"** — source Reading Ease 69.6 → 28.8:

> **Clause:** "Our Terms now include more details about when we might need to terminate our Agreement with bad actors. We provide a greater commitment to give notice when we take such action and what you can do to appeal"
>
> **Explanation:** "YouTube is providing more transparency on why accounts are terminated and is committing to a better notice and appeal process. As a monetized professional, this reduces the risk of an abrupt, unexplained loss of income by providing a clearer path to contest wrongful terminations."

The "clause" here is not legalese at all. It is Google's own marketing summary. The explanation is arguably *more* informative — it names the consequence for this specific user — and Flesch penalises it 41 points for saying "transparency," "terminated," "monetized," and "unexplained." **This single pair is the entire thesis of Part II.3 in miniature.**

---

# Part IV — Corpus contamination: the YouTube case study

The reason the "already plain" band exists at all is a property of the corpus, not of the system, and a replicator must know about it.

`tos_case_studies/youtube/NEW YouTube ToS (As of Dec 10 2019).txt` opens:

> "Our Terms of Service have been updated. **This summary is designed to help you understand some of the key updates we've made to our Terms of Service (Terms).** We hope this serves as a useful guide, but please ensure you read the new Terms in full."

The file interleaves Google's **plain-English change summary** with the terms themselves. When the diff engine quotes from the summary sections, the "legalese" side of the pair is professionally written plain English produced by Google's own communications team. The system is then measured on its ability to simplify text that was already simplified — by a competitor at the same task.

Distribution of source difficulty by service:

| Service | <20 | 20–40 | 40–60 | >60 | n | median clause FRE |
|---|---|---|---|---|---|---|
| TikTok | 9 | 10 | 12 | 5 | 36 | 39.4 |
| X (Twitter) | 7 | 7 | 14 | 3 | 31 | 43.5 |
| YouTube | 4 | 6 | 7 | 3 | 20 | 40.0 |

Contamination is not confined to YouTube — TikTok contributes five already-plain excerpts, mostly boilerplate ("We may make changes to these Terms from time to time…", Reading Ease 82.7). Roughly **13% of the sample (11/87) is text that had no readability problem to solve**, and another 38% sits in the moderate band.

This also explains the per-service results in `phase4_readability_analysis.csv`. TikTok shows the only significant per-service grade-level *increase* (+2.0, p = 0.008), because it carries the most already-plain boilerplate; X (Twitter), whose new ToS is the densest of the three (document-level Reading Ease 29.1), is statistically flat.

**Two defensible responses, and you should state which you chose:**

- **(a) Report as-is and explain.** The system genuinely receives such text in production — this is what users actually paste in. The pooled number is then an honest estimate of average behaviour on realistic input. This is the recommended choice; it requires no re-run.
- **(b) Add a pre-registered filter** excluding source excerpts above some Reading Ease threshold, on the grounds that a simplification tool cannot be evaluated on text needing no simplification. Defensible, but the threshold is arbitrary and it invites the accusation that you filtered until the result improved.

Choose (a), report (b) as a considered alternative, and let the banded analysis carry the nuance.

---

# Part V — The document-level supplement (recommended addition)

Part II.1 showed the Pravasi & Das instrument cannot run at the change level. It *can* run at the level Pravasi & Das actually used — whole document against whole interpretation — and the data for that is already on disk. Concatenating all `what_changed` + `impact_for_user` per service and scoring against the full new ToS, using the real `py-readability-metrics` library (all six texts clear 100 words comfortably):

| Service | Source words | Source FRE | Source FKGL | Explanation words | Expl. FRE | Expl. FKGL | Compression |
|---|---|---|---|---|---|---|---|
| TikTok | 6,682 | 34.6 | 16.2 | 2,086 | 34.1 | **14.1** | 3.2× |
| YouTube | 4,198 | 43.5 | 13.2 | 1,063 | 33.8 | 13.9 | 3.9× |
| X (Twitter) | 9,449 | 29.1 | 17.3 | 1,875 | **37.5** | **13.6** | 5.0× |
| **Mean** | 6,776 | 35.7 | 15.6 | 1,675 | 35.1 | **13.9** | **4.0×** |

Three things to take from this:

1. **Grade level improves by 1.7 grades on average** (15.6 → 13.9), and improves in the two services whose source is genuinely legal text. YouTube is the sole regression, and YouTube is the contaminated case. This is the *same* pattern as the banded change-level analysis, arrived at through a completely different aggregation — which is meaningful convergent evidence.
2. **Reading Ease is flat** (35.7 → 35.1), again because it is dominated by the syllable term.
3. **The explanations are 4× shorter than the source.** Flesch cannot see this at all, and it is plausibly the largest real usability gain the system delivers: a user reads 1,675 words instead of 6,776 and is pointed at what changed. **Volume reduction is a defensible, separately-reportable claim that requires no readability instrument.**

n = 3 means this is **descriptive only** — no statistical test is possible or should be attempted. Report it as a supplementary table with that caveat stated. It is worth including precisely because it is the only analysis in this phase that uses the actual Pravasi & Das instrument.

---

# Part VI — What to write in the manuscript

> ⏸ **On hold — see the status banner at the top.** This part is the reinstatement plan for after the proposal defense, not current instructions. The proposal reports the protocol only. Re-run before applying any of this.

## VI.1 What must be reported

1. **The pooled result, as the primary finding, including its negative direction.** Do not bury it. A thesis that reports a null or negative result on its own headline claim and then explains it is stronger than one that only reports favourable strata.
2. **The library returning n = 0**, with the word-count reason and the point about unit-of-analysis mismatch.
3. **The banded analysis, explicitly labelled post hoc.**
4. **The corpus contamination**, named as a specific property of the YouTube case study.
5. **The `harm_label` caveat** already in the notebook: the per-risk-class breakdown stratifies by the *generator's own* risk judgement, not the classifier's. Never describe it as the classifier's verdict.
6. **Which classifier checkpoint was in the pipeline**, with the explicit statement that the choice is for consistency and does not cause the readability numbers.

## VI.2 Claims you may make

- "On the densest quartile of quoted contract language — where the readability problem is concentrated — the generated explanations reduced reading difficulty by approximately six grade levels, improving 95% of cases."
- "The system produces explanations at a consistent reading level (grade 12–15) largely independent of source difficulty, functioning as a leveler rather than a monotone simplifier."
- "Explanations reduce reading volume approximately fourfold relative to the source document."
- "Both source and explanation remain above a general-audience reading level. The system reduces difficulty; it does not eliminate it."

## VI.3 Claims you may not make

- ❌ "The application makes Terms of Service readable." Unqualified, the pooled data contradicts it.
- ❌ "The improved Legal-BERT checkpoint produced more readable explanations." No A/B, no mechanism.
- ❌ Anything citing the `_raw` columns as a Pravasi & Das replication. Different syllable counter, no length gate.
- ❌ Any causal claim from the banded analysis without the post-hoc label attached.

## VI.4 Draft framing paragraph

> Readability was assessed following Pravasi and Das (2024), pairing each generated change with the Terms of Service excerpt it quotes (n = 87 changes across three version pairs). The instrument they employed, `py-readability-metrics`, requires a 100-word minimum and returned no scorable pairs: the median quoted excerpt is 35 words and the median explanation 56, reflecting a unit-of-analysis mismatch between document-level readability instruments and change-level interfaces. Flesch scores were therefore recomputed without the length gate as a robustness check.
>
> Pooled across all changes, explanations scored marginally *harder* than their sources (Flesch Reading Ease 38.0 → 34.5, Wilcoxon p = 0.040, rank-biserial −0.25; grade level 13.7 → 14.3, p = 0.143). Decomposition attributes 89% of this difference to syllables per word rather than sentence length, consistent with the known limitation that Flesch penalises the explicit naming of legal concepts — *monetization*, *indemnification*, *arbitration* — which is precisely what makes an explanation actionable.
>
> A post hoc stratification by source difficulty shows the pooled figure averages two opposing regimes. On excerpts scoring below 20 Reading Ease (n = 20), explanations improved readability by 22.4 points and 5.8 grade levels, with 95% of cases improving (rank-biserial 0.99). On excerpts already above 60 (n = 11) — largely Google's plain-English change summary, which the YouTube case study reproduces verbatim — explanations scored 22.5 points worse, with no cases improving. The generator emits prose at a stable level (grade 12–15) largely independent of input, functioning as a leveler rather than a monotone simplifier. This is a narrower claim than the application's stated value proposition, but a better-supported one.

---

# Part VII — Limitations to state explicitly

1. **Single generation run.** The LLM is non-deterministic and was sampled once. Run-to-run variance in readability is unmeasured. A second run with a different seed would bound it.
2. **Single user profile.** `impact_for_user` is personalised; all 87 changes used the Content Creator profile. Role-conditional readability is unmeasured.
3. **Three documents.** Service-level results rest on n = 20–36 changes each; the document-level supplement on n = 3. Neither supports generalisation beyond these three platforms.
4. **Flesch is a proxy, not a comprehension measure.** It cannot see jargon, ambiguity, or correctness. The system could produce a fluent, well-scoring explanation that is *wrong*, and nothing in this phase would detect it. **Factual accuracy of the explanations was not evaluated** — this is the largest remaining gap in the artifact's evaluation and should be named as future work.
5. **Uncorrected multiple comparisons.** 28 tests at α = 0.05. The pooled p = 0.040 does not survive correction; the band results at p < 0.001 do.
6. **The `_raw` syllable counter is heuristic** and will differ from a dictionary-based counter on irregular words. Direction and magnitude are trustworthy; the second decimal place is not.
7. **Post hoc stratification**, with the regression-to-the-mean concern of III.3 unresolved by design.

---

# Part VIII — Replication checklist

For a reader who wants to reproduce this end to end:

1. `pip install py-readability-metrics` and `python -m nltk.downloader punkt`.
2. Start the classifier backend on the checkpoint of record:
   `LAWGIC_MODEL_DIR=saved_models/lawgic_classifier_legal-bert_phase2 python3 -m uvicorn api.server:app --port 8000`
   Add `LAWGIC_DEVICE=cpu` if MPS misbehaves. Verify with `curl localhost:8000/api/model-info`.
3. Start the application: `npm run dev` in `lawgic-tos-changes`. Confirm `OLLAMA_MODEL=gemma4:31b-cloud` in `.env.local` and a signed-in Ollama session.
4. Confirm Ollama: `curl -s http://localhost:11434/api/tags`.
5. Remove any stale `phase4_changes.jsonl` from a run against a different checkpoint. The preflight refuses to resume across checkpoints; every row records `classifier_model`.
6. Run the notebook top to bottom. ~20–40 minutes on a MacBook. Cell 8 is resumable per service.
7. **Read `phase4_readability_pairs.csv` by hand** before interpreting anything. Sort by `explanation_flesch_ease_raw − clause_flesch_ease_raw` and read the top and bottom five pairs. The numbers in this document are not interpretable without doing that, and neither are yours.

## Suggested notebook improvement

The current figure (`phase4_readability.png`) plots one violin pair and one slope plot, both pooled. That visualisation now actively hides the finding: the crossing slope lines are the whole story and they are rendered as undifferentiated noise. Recolour the slope plot by source-difficulty band rather than by `harm_label`:

```python
band = pd.cut(plot_data[clause_col], [-100, 20, 40, 60, 200],
              labels=["<20 (very dense)", "20–40", "40–60", ">60 (already plain)"])
colours = {"<20 (very dense)": "#c0392b", "20–40": "#e67e22",
           "40–60": "#7f8c8d", ">60 (already plain)": "#27ae60"}
```

The red lines will fan upward and the green lines downward, and the "leveler" claim becomes visible in one glance instead of requiring a table.

---

## Appendix — How to read a Flesch number

| Reading Ease | Grade | Description | Typical text |
|---|---|---|---|
| 90–100 | 5 | Very easy | Children's books |
| 60–70 | 8–9 | Standard | Reader's Digest, mainstream news |
| 50–60 | 10–12 | Fairly difficult | Broadsheet features |
| 30–50 | 13–16 | Difficult | Academic prose |
| 0–30 | 17+ | Very difficult | Legal and technical documents |
| < 0 | — | Off-scale | Unbroken statutory clauses |

Anchors for this study:

- Quoted ToS excerpts averaged **38.0** — "difficult," university level.
- Explanations averaged **34.5** — also "difficult," also university level.
- The densest quartile of excerpts averaged **8.3** — off the bottom of the practical scale.
- Those same excerpts, once explained, averaged **30.7** — moved from *very difficult* into the *difficult* band.

The last line is the honest one-sentence summary of what the system does: it moves unreadable legal text up into difficult-but-parseable prose. It does not make it easy, and nothing in this evaluation claims it does.
