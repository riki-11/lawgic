# CHANGES — Statistical Rigor Upgrade (Phases 1–4)

The Phase 1–4 evaluation work is **all new files**. No existing notebook, dataset or
checkpoint was modified; the taxonomy, mapping tables, masking logic and loss definitions
are untouched. Nothing has been committed or pushed.

**Exception, made later at the user's explicit direction:** the Ollama model migration and
the `lawgic-web-app` deprecation touched existing files — `api/llm_interpreter.py`, `.env`,
`.env.example`, `README.md`, `lawgic-web-app/{.env,README.md}`. Those are itemised in §6.

> **Read §6 and §8 first if you are picking this up cold.** The thesis artifact is
> **`lawgic-tos-changes`**; `lawgic-web-app` is deprecated, and Phase 4 was redesigned
> accordingly.

---

## 1. Repo map (read-only exploration pass)

### Data artifacts

| Path | Role |
| --- | --- |
| `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv` | **The training corpus.** 26,479 rows, one per unique clause. Columns: `text`, `normalized_text`, `sources` (JSON list), `labels_presence` (45-float), `topic_mask` (45-float, source-aware supervision mask), `scores`, `topic_scores`, `active_topic_ids`, `conflict_topic_ids`, `has_score_conflict`, `harm_score` ∈ {-1,0,1}, `harm_score_class` ∈ {0,1,2}, `harm_mask`, `native_annotations` (per-annotation provenance). |
| `generated_files/lawgic_taxonomy/lawgic_topics.json` | 45-topic taxonomy; `unclassified` is dropped at load, leaving 44. |
| `generated_files/lawgic_taxonomy/lawgic_fusion_summary.json` | Fusion report + canonical `topic_ids`. |
| `generated_files/lawgic_taxonomy/lawgic_combined_long.csv` | Long-format (one row per annotation) precursor to the wide corpus. |
| `generated_files/lawgic_taxonomy/reports/` | `corpus_report.json`, `per_topic_supervision.csv`, `near_duplicate_audit.json`, `near_duplicate_pairs.csv`. |
| `datasets/tos_dr/{points,services,documents,cases,topics}.csv` | Raw ToS;DR export (gitignored). `points.id → service_id` is the join that gives ToS;DR rows a document/service identifier. |
| `saved_models/lawgic_classifier_legal-bert_v3/` | The trained checkpoint (gitignored): `model_state_dict.pt`, `model.safetensors`, `topic_head_weights.pt`, `harm_head_weights.pt`, tokenizer, `lawgic_topics_44.json`, `training_metadata.json`, `test_metrics_topic.json`, `test_metrics_harm.json`, `checkpoints/checkpoint-{37072,45016}/`. |

**Source composition of the wide corpus** (rows can carry more than one source):
ToS;DR 21,949 · CLAUDETTE 3,182 · 100 ToS 1,460.
(The 3,721 / 2,048 figures in the brief are *long-format annotation rows*; the wide corpus
collapses multi-topic annotations of one clause into a single row. Phase 3 operates on wide
rows and reports both numbers.)

### Where each concept lives

| Concept | Location |
| --- | --- |
| Dual-head module definition | `notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb` cell 18 (`LawgicDualHeadModel`) — encoder → `pooler_output` → `Linear(768,44)` + `Linear(768,3)`. Duplicated for serving in `api/server.py:80`. |
| Losses | Same notebook, cell 20 (`DualHeadTrainer.compute_loss`): masked BCE-with-logits / Σmask + masked CE / Σharm_mask, equally weighted. |
| Metrics | Cell 14 (`masked_metric_summary`), cell 22 (`per_topic_report`, `harm_metric_summary`, `compute_metrics`). |
| **Split** | Cell 12 — **computed inline, never persisted.** Composite key `f"{primary_topic}__harm{harm_class}"`, keys with <5 members collapsed to `__rare__`, two-stage `train_test_split(random_state=42)`, then an exact-text leakage assertion. |
| Degenerate-model assertion | Cell 14: zero-logit topic macro-F1 must be < 0.95 (actual: **0.0923**). |
| Training config | Cell 24: lr 3e-5, batch 8 (eval 16), ≤20 epochs, early stop patience 3 on `macro_f1`, wd 0.01, warmup 0.06, fp16 on CUDA, max_length 256, seed 42. |
| Near-duplicate audit | `scripts/near_duplicate_split_audit.py` — TF-IDF `char_wb` 3–5-grams, `min_df=2`, 1-NN cosine vs train. |
| Corpus report | `scripts/corpus_report.py`. |
| Source-classifier probe | `notebooks/model_finetuning/lawgic_classifier_probe.ipynb` (linear probe on frozen encodings). |
| **App generative flow (ACTIVE)** | `lawgic-tos-changes/lib/ollama-diff.js` — `buildChunkPrompt()`, `buildSummarizePrompt()`, `callOllama()` (POST `/v1/chat/completions`, `gemma4:31b-cloud`, `cleanJsonContent()` fence strip, 180 s timeout). Orchestrated by `pages/index.js` via `/api/diff/{prepare,chunk,summarize}`. Per-change chat in `pages/api/chat.js`. |
| App generative flow (legacy) | `api/llm_interpreter.py` — `build_explain_system_prompt()`, `build_explain_prompt()`, `explain_clause()`. Only the deprecated `lawgic-web-app` called it. |
| API server | `api/server.py` — loads the v3 checkpoint, chunks documents, serves `/api/analyze_tos` and `/api/explain_tos_scores`. |

### Two discrepancies found during exploration (both material)

1. **`scripts/near_duplicate_split_audit.py` does not reproduce the training split.** Its
   docstring claims it "replicates the notebook's split exactly", but it stratifies on the
   **primary topic only**, while the fine-tuning notebook stratifies on the composite
   `topic__harm` key and collapses rare strata. The contamination *rates* it reports are
   still indicative, but the specific clauses it flags are not the ones the trained model
   actually saw. Phase 1 therefore re-derives contamination against the **persisted
   seed-42 split**, reusing the audit's vectoriser settings verbatim.
2. **The split was never persisted.** Fixed first (see below) before anything else was
   built, per the ground rules.

---

## 2. Files added

### `scripts/lawgic_eval_core.py`
Shared library for all four notebooks. A faithful lift of the fine-tuning notebook's
corpus loading, 45→44 compaction, split construction and metric definitions, plus the
audit script's TF-IDF machinery. One import instead of four copy-pastes that drift — the
whole premise of the new work is that every comparison sits on identical splits and
identical metric code.

Contents: `load_taxonomy`, `load_corpus`, `build_split_assignment`, `persist_split`,
`load_split`, `split_frames`, `label_arrays`, `topic_metrics`, `harm_metrics`,
`all_metrics`, `per_topic_table`, `bootstrap_ci`, `paired_bootstrap_delta`, `mcnemar`,
`max_train_similarity`, `load_trained_checkpoint`, `predict_logits`,
`document_identifier`, `tosdr_service_map`, `to_booktabs`, `write_outputs`.

`python scripts/lawgic_eval_core.py` runs a self-check (the vectorised masked-F1
implementation is asserted equal to the notebook's sklearn version) and persists the split.

**Already executed once.** Output:

```
self-check: vectorised masked F1 matches sklearn reference ✓
Persisted split -> generated_files/lawgic_taxonomy/splits/split_seed42.csv
Row counts: {'train': 21183, 'test': 2648, 'validation': 2648}
```

which matches `training_metadata.json` exactly. That file
(`generated_files/lawgic_taxonomy/splits/split_seed42.csv`, one row per clause with
`row_id`, `split`, `normalized_text_sha`) is now the single source of truth every new
notebook loads. **It is a new artifact, not a modified one.**

Note on `topic_metrics`: it is vectorised (numpy TP/FP/FN) rather than looping
`sklearn.f1_score` 44 times, because the bootstrap calls it ~1,000 times per set. The
self-check pins it to the original definition.

### `scripts/lawgic_train_matrix.py`
Phase 2/3 support. The original training protocol re-expressed as
`run_config(RunConfig(...))`, with exactly three parameters — `encoder_name`, `seed`,
`heads` — plus a Phase-3-only `holdout_source`. Everything else is fixed at the original
values and read from the persisted split.

Key pieces:
- `pooled_representation()` — the per-architecture adapter. First real token for
  BERT/Legal-BERT/RoBERTa, **last** real token for XLNet (its summary token is appended,
  not prepended). Selection is by attention mask, not fixed index, because XLNet's
  tokenizer pads **left** and the BERT-family tokenizers pad right.
- `LawgicDualHeadCollator` — keeps only `tokenizer.model_input_names`, which is how
  RoBERTa's missing `token_type_ids` is handled without an `if roberta:` anywhere.
- `DualHeadTrainer` — loss code copied line for line; `head_mode` only drops one term
  from the sum.
- `RunConfig.best_metric_key` — dual and topic-only select on topic macro-F1 (the original
  protocol); risk-only must select on `risk_macro_f1`, since its topic head is untrained.
- `source_supervision_mask()` — per-row reconstruction of the topic cells a given source
  actually annotated, from `native_annotations`.
- `assert_not_degenerate()` — the zero-logit macro-F1 < 0.95 check, run before every fit.

`python scripts/lawgic_train_matrix.py` runs a training-free self-check (pooling indices
under both padding sides, 18 unique run ids). **Already executed:** passes.

**Known deviation, must be footnoted in the manuscript:** the v3 checkpoint fed the heads
BERT's `pooler_output`; the matrix uses the raw first token for all encoders instead,
because `roberta-base` ships a *randomly initialised* pooler and keeping `pooler_output`
would have handicapped RoBERTa for reasons unrelated to the encoder. Consequence: the
legal-bert/seed-42 cell of the matrix is **not** expected to reproduce the v3 numbers
exactly. Phase 1 uses the original `pooler_output` architecture
(`core.load_trained_checkpoint`) because it is reading the saved weights.

### `notebooks/evaluation/01_decontaminated_eval.ipynb` — Phase 1
Loads the v3 checkpoint, scores the persisted test split once, then reports metrics on the
full test set and on the subset with no training neighbour at cosine ≥ 0.90, each with
percentile bootstrap 95% CIs over 1,000 clause-level resamples. Includes a sanity check
against the checkpoint's own saved metrics, and the document-grouped re-split as
**unexecuted code** behind `RUN_DOCUMENT_SPLIT = False`.

Rationale: the reported 0.75 / 0.83 / 0.83 include near-duplicate contamination (15.5% of
test clauses at ≥0.80, 6.8% at ≥0.90, 3.7% at ≥0.95). Reporting only the full number
overstates generalisation; reporting only the decontaminated number discards legitimately
recurring boilerplate. Both, with intervals, is the defensible presentation.

**Document-identifier coverage (verified, not estimated):** 100% of the 26,479 rows resolve
to a document/service group — **2,015 distinct groups** — via CLAUDETTE `source_id`
(`service:idx:tag`), 100 ToS `source_id` (`Platform:idx`), and ToS;DR `point_id → service_id`
joined from `datasets/tos_dr/points.csv`. ToS;DR groups at the *service* level because its
points annotate a service rather than one document; that is the coarser, safer unit. If
`datasets/tos_dr/points.csv` is absent (it is gitignored), ToS;DR rows fall back to
unresolved (~83%) and each becomes its own singleton group — a conservative fallback that
never merges unrelated clauses.

Outputs: `phase1_decontaminated.{csv,tex}`, `phase1_contamination_flags.csv`,
`phase1_test_logits.npz`.

### `notebooks/evaluation/02_multiseed_encoder_runs.ipynb` — Phase 2
Config listing → manual-step checks → wall-time estimate → resumable runner → aggregation.
18 runs: 4 encoders × 3 seeds (dual-head) + legal-bert × 3 seeds × {topic-only, risk-only}.
Aggregation reads only persisted run artifacts, so it re-runs without a GPU.

Rationale: single-run point estimates cannot support an encoder claim. The matrix gives
mean ± sd over seeds, bootstrap CIs on the test metrics, and paired significance tests
(McNemar on risk-head item correctness; paired bootstrap for topic macro-F1 deltas) — all
on identical splits, so a difference is attributable to the varied component.

Outputs: `runs/<run_id>/{metrics.json,test_logits.npz,per_topic.csv}`, `phase2_runs.csv`,
`phase2_aggregate.csv`, `phase2_bootstrap_ci.csv`, `phase2_significance.csv`,
`phase2_headline.{csv,tex}`, `phase2_per_topic.{csv,tex}`.

### `notebooks/evaluation/03_source_heldout_probes.ipynb` — Phase 3
Two probes on legal-bert / seed 42 / dual-head: hold out CLAUDETTE, hold out 100 ToS. Each
removes the source from train **and** validation and restricts test to that source's rows.
Scoring is restricted to the topic cells the held-out source itself supervised — otherwise
the probe measures mask shape, not comprehension. The in-distribution baseline is the
Phase 2 legal-bert/seed-42 run re-scored on **the identical rows and identical cell mask**,
so the retained-performance ratio has a matching denominator.

**No ToS;DR probe**, with the rationale written into the notebook: ToS;DR is ~83% of the
corpus (~88% of training rows after the split); removing it leaves ~2,500 training clauses,
so a score collapse would be indistinguishable from data starvation and would answer
neither question.

Outputs: `runs/*__holdout-*/`, `phase3_source_holdout.{csv,tex}`.

### `notebooks/evaluation/04_explanation_readability.ipynb` — Phase 4
**Targets `lawgic-tos-changes`, the active thesis artifact** (see §6 for the app switch and
§8 for why the design changed).

There is no per-clause explain flow in that app — it diffs **two versions** of a document
and emits *changes*. So the paired unit is one **change**, not one clause:

| Side | Field | What it is |
| --- | --- | --- |
| Source (legalese) | `new_text`, or `old_text` for removals | The excerpt quoted verbatim from the real ToS, ≤400 chars |
| Explanation | `what_changed` + `impact_for_user` | The prose body of one change card |

The notebook is a **client of the running app**: it drives `/api/case-studies/<id>`,
`/api/analyze`, `/api/diff/prepare` and `/api/diff/chunk` over HTTP, so `buildChunkPrompt()`
in `lib/ollama-diff.js` executes inside the app itself — prompt, system message,
`gemma4:31b-cloud` call, `cleanJsonContent()` and retry-on-`SyntaxError` all unmodified.

Two orchestration details from `pages/index.js` are replicated exactly because skipping
either would change the sample:
- `lawgicClauses` is passed to the **first chunk only** (`index.js:146`).
- `mergeAndDedupeChanges()` runs client-side before the user sees anything; it is ported to
  Python with a self-check (same 0.82 Jaccard threshold, same harm-then-type sort, same
  renumbering).

Sample: all changes from the three case-study pairs shipped with the app (TikTok, YouTube,
X). N is whatever the documents yield and is reported per service — not a fixed 150.
Readability via `py-readability-metrics` (Pravasi & Das, 2024), paired analysis broken down
by `harm_label` and by service.

**Caveat for the manuscript:** in this app `harm_label` is **assigned by the LLM**, not by
Legal-BERT. The classifier enters the prompt as advisory context only. Do not describe the
strata as the classifier's verdict.

**Minimum-word handling.** `py-readability-metrics` requires ≥100 words and raises
`ReadabilityException` below that. The prompt schema caps quoted excerpts at 400 characters
and explanations run 2–3 sentences, so **most pairs will fall under the gate**; a pair is
excluded if *either* side fails. The notebook reports the exclusion count explicitly.
Because that gate can remove most of the sample, a clearly-labelled **secondary** column set
recomputes the same two Flesch formulas inline (standard definitions, vowel-group syllable
heuristic) with no length gate, as a full-coverage robustness check. Report the library rows
as the Pravasi & Das replication; cite the all-pairs rows as the robustness check, never as
the same measurement.

**Single fixed user profile.** `buildChunkPrompt` personalises `impact_for_user` to the
reader's role, so the profile is an input variable. One profile (`Content Creator`, the role
with the most concern options in `lib/constants.js`) is fixed for the whole run and recorded
on every row. Readability may differ by role — state this as a limitation; a second run
under a different profile is the way to test it.

Outputs: `phase4_changes.jsonl`, `phase4_readability_pairs.csv`,
`phase4_readability_analysis.csv`, `phase4_readability_summary.{csv,tex}`,
`phase4_readability.png`.

### Nomenclature
The repo calls the second head **harm**; the brief calls it **risk**. Internals keep
`harm_*` (matching the corpus columns and checkpoint files); all reported metrics and
table labels use `risk_*`. They are the same head.

---

## 3. Run order

```
0. python scripts/lawgic_eval_core.py          # already done — split is persisted
   python scripts/lawgic_train_matrix.py       # already done — self-check passes

1. notebooks/evaluation/01_decontaminated_eval.ipynb        (needs the v3 checkpoint)
2. notebooks/evaluation/02_multiseed_encoder_runs.ipynb     (needs a GPU)
3. notebooks/evaluation/03_source_heldout_probes.ipynb      (needs 2 to have run)
4. notebooks/evaluation/04_explanation_readability.ipynb    (needs `npm run dev` + Ollama;
                                                             uvicorn optional but wanted)
```

1 and 4 are independent of 2 and 3. 3 hard-depends on 2 (it reads
`runs/legal-bert-base-uncased__seed42__dual/test_logits.npz`). All notebooks are run
top-to-bottom; 2, 3 and 4 skip already-completed work on re-run.

**Notebook 4 no longer touches the v3 checkpoint directly** — it drives `lawgic-tos-changes`
over HTTP, and that app calls the classifier itself through `/api/analyze`. Start the
FastAPI server anyway so `lawgicClauses` is non-null.

### Expected runtime, order of magnitude

| Notebook | Runtime | Dominated by |
| --- | --- | --- |
| 01 | **~10 min** | 2,648-clause forward pass (1–2 min, CPU-viable) + TF-IDF 1-NN over 21k train clauses (~2–5 min) + 2×1,000 bootstrap resamples |
| 02 | **tens of hours** on one GPU | 18 full fine-tunes. **Per-run wall time is not recoverable from the v3 artifacts** — its `trainer_state.json` records 17 epochs and eval throughput but no `train_runtime`, because the original notebook never logged the training summary. The notebook prints a derived lower bound (~30–60 min/run, i.e. ~10–20 h total) and then stores the real `wall_seconds` for every run. **Measure on run 1 and re-plan.** |
| 03 | **~2 runs' worth** of 02, i.e. a few hours | 2 fine-tunes on slightly smaller training sets |
| 04 | **~10–20 min** (estimated) | 3 documents × 5–11 diff-chunk calls each against `gemma4:31b-cloud` (~1–3 s per call, measured), plus 2 classifier passes per document if uvicorn is running. Scoring and analysis are negligible. Not yet measured end to end — the notebook prints real elapsed time. |

---

## 4. Manual steps — consolidated checklist

- [ ] **(01) Checkpoint present.** `saved_models/lawgic_classifier_legal-bert_v3/` with
      `model_state_dict.pt`, `config.json`, `model.safetensors`, tokenizer files. Gitignored,
      so absent on a fresh clone.
- [ ] **(02) GPU.** Run notebook 02 on the CUDA machine that produced v3. On CPU this is
      days. FP16 auto-enables on CUDA, matching the original protocol.
- [ ] **(02) Model downloads.** Network access for `bert-base-uncased`, `xlnet-base-cased`,
      `roberta-base` (~440 MB each). Pre-fetch command is in the notebook's manual-step cell.
- [ ] **(02) `sentencepiece`** importable for the XLNet tokenizer. Already in
      `notebooks/requirements.txt` (`0.2.1`); the notebook verifies rather than installs.
- [ ] **(02) Disk.** ~10 GB under `generated_files/lawgic_taxonomy/runs/`.
- [ ] **(03) Run 02 first**, at least the legal-bert/seed-42/dual config.
- [ ] **(04) `pip install py-readability-metrics`** — the only new dependency any phase adds.
      Then `python -m nltk.downloader punkt` (the library needs NLTK's sentence tokenizer).
- [x] **(04) Ollama model — nothing to do.** `lawgic-tos-changes` already runs
      `gemma4:31b-cloud` and always has. See §6.
- [ ] **(04) Start the app:** `cd lawgic-tos-changes && npm run dev` (port 3000). The
      notebook is only a client — it fails immediately at the preflight cell without it.
- [ ] **(04) Start the classifier** (`uvicorn api.server:app --port 8000`) so `lawgicClauses`
      is non-null. Optional — the app degrades to `null` — but **record which mode you ran**,
      since it changes what the prompt contained.
- [ ] **(04) Confirm Ollama is up and signed in:**
      `curl -s http://localhost:11434/api/tags >/dev/null && echo ok`. The cloud model needs
      internet plus a signed-in Ollama account on the daemon host.
- [ ] **(04) Browser sanity check of `lawgic-tos-changes`** before trusting a batch run.
      Steps in §7.
- [ ] **(optional, 01) Document-grouped re-split.** Requires `datasets/tos_dr/points.csv`
      for full coverage. Left at `RUN_DOCUMENT_SPLIT = False` on purpose — producing a
      second split forks the corpus from the trained checkpoint and means nothing without a
      full retrain.

---

## 5. Numbers to transfer into the manuscript

### Chapter 4 baseline table ← Phase 1
Source: `generated_files/lawgic_taxonomy/evaluation/phase1_decontaminated.tex`
(`\label{tab:decontaminated-eval}`).
- Topic macro-F1, topic micro-F1, risk accuracy, risk macro-F1 — **full test** and
  **decontaminated**, each as `point [CI-low, CI-high]`, plus the delta column.
- The decontaminated subset size and the retained fraction (printed by the contamination
  cell), and the ≥0.80 / ≥0.90 / ≥0.95 contamination rates recomputed against the *actual*
  training split (these will differ slightly from
  `reports/near_duplicate_audit.json`, which used a different stratification — say so).
- The sanity-check table's max absolute drift from the saved metrics, as a one-line
  footnote that the recomputation reproduces the reported run.

### Ablation / encoder table ← Phase 2
Source: `phase2_headline.tex` (`\label{tab:encoder-matrix}`) and
`phase2_per_topic.tex` (`\label{tab:per-topic}`).
- Headline: 6 rows (4 encoders dual-head + legal-bert topic-only + risk-only) × 4 metrics,
  mean ± sd over 3 seeds.
- Per-topic: 44 topics + macro avg + weighted avg, precision / recall / F1 / support, for
  the best legal-bert seed, with seed-mean F1 alongside.
- From `phase2_significance.csv`: for legal-bert vs each other encoder and vs each head
  ablation — McNemar `b`/`c`/`p` for the risk head, and topic macro-F1 delta with its
  paired-bootstrap 95% CI and p. **This is what settles the XLNet-vs-Legal-BERT question**;
  state the direction and whether the interval excludes zero.
- From `phase2_runs.csv`: measured mean wall time per run and epochs to early stop (the
  methodology section currently has no measured figure at all).
- The pooling deviation footnote (§2, `lawgic_train_matrix.py`).

### Probe table ← Phase 3
Source: `phase3_source_holdout.tex` (`\label{tab:source-holdout}`).
- Per source (CLAUDETTE, 100 ToS) × 4 metrics: in-distribution, held-out, retained ratio,
  test rows, observed supervised cells.
- The written justification for excluding a ToS;DR probe — the ~88%-of-training-rows figure
  belongs in the text, not only in the notebook.

### Readability table ← Phase 4
Source: `phase4_readability_summary.tex` (`\label{tab:readability}`) and
`phase4_readability.png`.
- Per metric (Flesch Reading Ease, Flesch-Kincaid Grade; library and all-pairs rows):
  n, excerpt mean, explanation mean, difference, test used, p, effect size, % improved.
- **N and its composition**: total changes, the per-service breakdown (TikTok / YouTube / X)
  and the `harm_label` distribution. The design is not a balanced 50/50/50 — say so.
- **The exclusion count**: how many pairs `py-readability-metrics` could score and how many
  were dropped for the 100-word minimum. A limitation the manuscript must state, not omit.
- The per-`harm_label` and per-service breakdowns from `phase4_readability_analysis.csv`.
- **Three framing statements that must appear in the text:**
  1. The measured system is `lawgic-tos-changes`, model `gemma4:31b-cloud` — cite the run
     date, since the endpoint is provider-hosted and can change.
  2. `harm_label` is LLM-assigned; Legal-BERT enters the prompt as context only.
  3. One fixed user profile (`Content Creator`) was used; readability may vary by role.
- Whether the classifier backend was running (`classifier_context` column in the pairs CSV).

---

## 6. Which app is the thesis artifact, and which Ollama model it uses

**`lawgic-tos-changes` is the active artifact. `lawgic-web-app` is deprecated.**
Deprecation is now recorded in `lawgic-web-app/README.md` (banner) and `lawgic/README.md`
(new *Frontends* table + a note on the legacy endpoint).

### The model question, settled

**`lawgic-tos-changes` has always run `gemma4:31b-cloud`. It never used `gemma4:e4b`.**
Verified exhaustively — the entire app contains exactly **one** model reference:

```
lib/ollama-diff.js:131   const ollamaModel = process.env.OLLAMA_MODEL || "gemma4:31b-cloud";
```

Both sides agree: the env value (`.env.local` → `OLLAMA_MODEL=gemma4:31b-cloud`) and the
hardcoded fallback. `.env.local` is the only env file present, and Next.js gives it highest
precedence in dev. **No change was needed in that repo, and none was made.**

It also already handles fenced output — `cleanJsonContent()` at `lib/ollama-diff.js:120`
strips ```` ```json ```` before `JSON.parse`. The app was built for this model from day one.

### The three apps/endpoints, disambiguated

| Component | Model | Status |
| --- | --- | --- |
| `lawgic-tos-changes` → `lib/ollama-diff.js` | `gemma4:31b-cloud` | **Active.** Diff/summarise/chat prompts. Calls `POST /api/analyze_tos` for classification only. |
| `lawgic/api/llm_interpreter.py` → `/api/explain_tos_scores` | now `gemma4:31b-cloud` | **Legacy.** Only `lawgic-web-app` ever called it. |
| `lawgic-web-app` → `src/services/llmService.js` | now `gemma4:31b-cloud` | **Deprecated.** Not measured. |

### Changes made to the legacy path (still correct, no longer on the critical path)

Before the deprecation was known, `/api/explain_tos_scores` was migrated to
`gemma4:31b-cloud`. Those changes stand — they are correct and harmless — but they now
affect an endpoint the thesis does not measure. Recorded here for completeness:

| File | Change |
| --- | --- |
| `api/llm_interpreter.py` | Added `_strip_code_fence()`; `_parse_explain_response()` calls it before `json.loads()`. `DEFAULT_OLLAMA_MODEL` → `gemma4:31b-cloud`, new `FALLBACK_OLLAMA_MODEL`, updated docstring, `__main__` self-check. |
| `.env`, `.env.example` | `VITE_OLLAMA_MODEL=gemma4:31b-cloud`, `e4b` fallback commented inline. |
| `lawgic-web-app/.env` | Same switch. Moot now that the app is deprecated. |

**Why the parser fix was needed at all** (worth keeping in the methodology notes, because it
is a property of Ollama, not of this code): the `format` schema constraint is **not enforced
for cloud-routed models** — the request is proxied upstream and the constraint dropped.
Measured: 5/5 clauses returned fenced; three request-level escapes (`format=<schema>`,
`format="json"`, `think=False`) all still fenced; stripping the fence parsed every time.
`lawgic-tos-changes` already worked around this; the FastAPI endpoint did not.

Verification:
```
python -m api.llm_interpreter          → self-check ✓
explain_clause() on gemma4:31b-cloud   → 1.7s / 1.4s / 0.8s, all OK
explain_clause() on gemma4:e4b         → 42.3s, OK  (fallback unbroken)
```

### Trade-off accepted

The cloud model needs internet and a signed-in Ollama session on the daemon host; there is
no API key in `lawgic/.env`. The `e4b` fallback is one commented line away and stays fully
local. Note also that a cloud endpoint can be updated or retired by the provider — cite the
model **and the run date** in the manuscript, and keep `phase4_changes.jsonl` as evidence.

---

## 7. How to verify `lawgic-tos-changes` yourself

### Step 1 — services up (1 minute)

```bash
curl -s http://localhost:11434/api/tags >/dev/null && echo "ollama up"

cd "/Users/riki/Coding Projects/Thesis/lawgic"
/Users/riki/anaconda3/envs/thesis-env/bin/python3 -m uvicorn api.server:app --port 8000
# second terminal:
curl -s http://localhost:8000/api/test-analyze | head -c 200

cd "/Users/riki/Coding Projects/Thesis/lawgic-tos-changes"
npm run dev            # http://localhost:3000
```

The classifier backend is **optional** — `pages/api/analyze.js` catches a refused connection
and returns `skipped: true`, and the diff still runs with `lawgicClauses: null`. Run it
anyway, so the Legal-BERT context actually reaches the prompt.

### Step 2 — confirm the model at the wire

```bash
curl -s http://localhost:3000/api/case-studies/tiktok | head -c 200
```
Then in the browser at `localhost:3000`, DevTools Network tab open:

1. Pick a case study (TikTok / YouTube / X), choose a role, run the analysis.
2. Watch for `POST /api/analyze` ×2 → then `/api/diff/prepare` → then repeated
   `/api/diff/chunk` → finally `/api/diff/summarize`.
3. **Pass:** change cards render with topic, harm badge, `what_changed`, `impact_for_user`.
4. Open a card and use the chat ("ask about this change") → `POST /api/chat` streams a
   2–3 paragraph answer.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Chunk analysis failed — could not parse Ollama response as JSON` | Model returned something `cleanJsonContent` could not rescue | Retry; if persistent, capture the raw response |
| Every section returns 0 changes | `isMateriallyDifferent` filtered them, or the wrong pair loaded | Check the two files under `tos_case_studies/<service>/` really differ |
| `Ollama error: 404` | Cloud session expired | `ollama run gemma4:31b-cloud "ok"` to re-auth |
| Analysis runs but no Legal-BERT context | FastAPI not running — expected, not a bug | Start uvicorn on port 8000 and re-run |
| Very slow / timeouts | `callOllama` allows 180 s, `/api/diff/chunk` 300 s | Check network; the cloud model is normally 1–3 s |

### Step 3 — offline fallback drill (do this before the defense)

`lawgic-tos-changes` has **no local fallback configured**. To make one, set
`OLLAMA_MODEL=gemma4:e4b` in `lawgic-tos-changes/.env.local` and restart `npm run dev`.
Expect it to be far slower and lower quality — but confirm it *runs*, so a dead network on
defense day is an inconvenience rather than a failure. Then set it back.

For the legacy FastAPI endpoint, the same drill is: comment `31b-cloud` and uncomment
`gemma4:e4b` in `lawgic/.env`, restart uvicorn, confirm `/api/explain_tos_scores` still
answers.

---

## 8. Phase 4 redesign — what changed and why

Phase 4 was originally built against `lawgic-web-app`: 150 test-split clauses, 50 per
predicted risk class, each explained by `/api/explain_tos_scores`. When the artifact moved
to `lawgic-tos-changes`, that design measured a component the thesis no longer ships.

**Why it could not simply be repointed.** `lawgic-tos-changes` has no single-clause explain
flow. Its generative unit is a *diff section* producing change objects. There is no prompt
in that app that takes one clause and returns one explanation — inventing one would have
broken the "reuse the app's prompt verbatim" requirement, which is the whole basis for
claiming the measurement describes the deployed system.

**Decisions taken (user-confirmed):**

| Decision | Choice | Consequence |
| --- | --- | --- |
| Paired unit | ToS excerpt (`new_text`/`old_text`) vs `what_changed` + `impact_for_user` | Measures exactly the prose users read on a change card |
| Prompt | `buildChunkPrompt` via live HTTP to the running app | Verbatim by construction; no port of the prompt |
| Sample | The three case-study pairs shipped with the app | N is whatever the documents yield; reported per service |
| Stratification | `harm_label` | **LLM-assigned, not Legal-BERT** — must be stated |

**What this costs.** The sample is no longer a controlled 50/50/50 across risk classes, and
it is no longer drawn from the held-out test split — it comes from three real ToS version
pairs. That is a fair trade: the previous design had clean stratification but measured the
wrong system. Report N per service and the harm-label distribution rather than implying a
balanced design.

**Rough size estimate.** 25–58 KB per document at `DIFF_CHUNK_MAX_CHARS = 5500` gives
roughly 5–11 chunks each, capped at `DIFF_MAX_CHANGES_PER_CHUNK = 5` changes per chunk, over
3 documents — order of 50–150 raw changes before dedupe. Not verified; the notebook prints
the real counts.

**Still open:** if N or the library-scorable subset is too small for a defensible paired
test, the fix is more ToS version pairs dropped into `tos_case_studies/`. The notebook globs
`CASE_STUDY_SERVICES`, so adding pairs needs a list entry and a re-run, no code change.

---

## 9. Manuscript corrections applied to `chapter_4.tex`

Four numeric/conceptual errors were found while checking Chapter 4 against the corpus and
the Phase 1 output, and corrected in
`/Users/riki/Coding Projects/Thesis/6a26daf36240f4b0d9c1e884/chapter_4.tex`. That file is
git-tracked, so all four are revertible.

### 9.1 Supervision-mask semantics (the load-bearing one)

**A source's coverage is the set of topics its annotation scheme can express, fixed by the
mapping table — it is NOT narrowed to the topics that happen to receive a positive in this
corpus snapshot.** Author-confirmed. The implementation was correct; the prose was not.

The two quantities had been conflated because they are close together:

| Source | Mask coverage (what the code does) | Topics ever positive |
| --- | --- | --- |
| ToS;DR | **42** of 44, constant per row | 37 |
| 100 ToS | **30** of 44, constant per row | 26 |
| CLAUDETTE | 1–4, per row, driven by the native label | 10 (union) |

So 5 ToS;DR topics and 4 100-ToS topics are inside coverage but never receive a positive.
Under the confirmed reading these still contribute legitimate **observed negatives**: the
questionnaire asked about the topic, no annotator flagged it, that is real negative evidence.

CLAUDETTE is the exception by design — its mask is per-row rather than coverage-wide,
because a missing CLAUDETTE annotation genuinely says nothing about topics that source never
evaluates. Widening it would manufacture false negatives.

**Line 219 corrected:** "all 37 topics … 36 observed negatives … the 26 within its coverage
… the 35 topics that source never evaluates" → **42 / 41 / 30 / 34**, with a new leading
sentence stating the coverage definition explicitly so the two quantities cannot be confused
again. (The 35 → 34 change also moves the sentence into the 44-topic training space, which
is what the rest of the paragraph uses: "one supervised cell and 43 unknown ones".)

**Reproduce:**
```python
import sys; sys.path.insert(0, "scripts")
import numpy as np, lawgic_eval_core as core, lawgic_train_matrix as tm
df = core.load_corpus()
mask = np.vstack(df["label_mask"].to_numpy()); lab = np.vstack(df["labels"].to_numpy())
single = df["sources"].map(len).to_numpy() == 1
for s in ("tos_dr", "100_tos", "claudette"):
    m = tm.source_row_mask(df, s) & single
    print(s, "coverage:", int((mask[m] == 1).any(axis=0).sum()),
          "| ever positive:", int((lab[m] == 1).any(axis=0).sum()))
```

### 9.2 Source-trace paragraph (line 387)

Same conflation, downstream. "a ToS;DR row is marked as observed for 37 topics while a
CLAUDETTE row is marked for only one or two" → now gives all three signatures (42 / 30 /
1–4) and states that they do not overlap.

This **strengthens** the motivation for the Phase 3 probe: the mask is not merely suggestive
of provenance, it is a perfect identifier — three disjoint constants. A model can read source
identity off the mask alone, with no reference to the clause text.

### 9.3 Held-out probe percentages (line 390)

Stated as fractions of the **training split** (21,183 rows), which is what the sentence
claims, not of the full corpus:

| Claim in text | Was | Now |
| --- | --- | --- |
| CLAUDETTE holdout | "under 8\%" | **11.9\%** (2,528 rows) |
| 100 ToS holdout | "under 8\%" | **5.6\%** (1,175 rows) |
| ToS;DR holdout | "88\%" | **82.9\%** (17,559 rows) |

The single "under 8\%" could not cover both small sources, so they are now stated
separately. The old 88\% figure was the corpus-wide **annotation** share (44,317 / 50,086 =
88.5\%), not the training-row share — an easy slip, since both numbers are real and describe
ToS;DR dominance. Typo `exluded` → `excluded` fixed in the same sentence.

The argument survives: 11.9\% is still small enough that a collapse on held-out CLAUDETTE
indicates trace matching rather than data starvation, which is the point the sentence makes.

### 9.4 Readability paragraph (line 396)

Rewritten for the `lawgic-tos-changes` design (§8). Three substantive changes:

1. **Unit.** "the proportion of clauses whose readability improves" → *changes*. The app has
   no per-clause explain flow; the pair is a quoted ToS excerpt against
   `what_changed` + `impact_for_user`.
2. **"by risk class" removed.** Chapter 4 uses "risk class" for the 3-class classifier head
   everywhere else, so that phrasing told readers the breakdown came from the classifier. It
   does not — `harm_label` is assigned by the explanation model, with Legal-BERT entering the
   prompt as context only. The new text says so explicitly rather than leaving it inferable.
3. **Instrument limits stated up front.** Both Flesch Reading Ease and Flesch-Kincaid Grade
   Level are named, and the 100-word minimum of `py-readability-metrics` is pre-registered
   along with a commitment to report the excluded count — rather than discovering the
   exclusion in the results and explaining it after the fact.

### Verified as still correct

- Headline table (lines 376–380): 0.75 / 0.83 / 0.83 / 0.83. Phase 1 recomputed
  0.754 / 0.834 / 0.829 / 0.825 from the checkpoint.
- Line 385's ablation design matches the Phase 2 matrix (dual vs single-head; Legal-BERT vs
  BERT, XLNet, RoBERTa).
- Line 400's "three document pairs" matches the case studies shipped with the app.

## 10. Taxonomy/corpus fixes — 2026-08-24

Full write-up: **[`docs/lawgic_taxonomy_revisions.md`](docs/lawgic_taxonomy_revisions.md)**.

Four defects from the panel's Recommendation 3 definition audit were fixed, producing a new
`_v2`-suffixed corpus generation. **No retraining.** The v1 artifacts and the v3 checkpoint's eval
harness are untouched.

| File | Change |
| --- | --- |
| `scripts/corpus_report.py` | `positive_sources` now credits a source per topic only from `native_annotations`, gated on the mask — not from row-level `sources` presence. Accepts an optional `v2` CLI arg to report on the 42-topic corpus. |
| `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` (cell 4) | Added `"ind": "indemn"` to `HUNDRED_TOS_CODE_ALIASES` (`sugg`/`inter` deliberately excluded, noted in `MAPPING_NOTES`). Topic-count guard now checks for `unclassified` presence instead of a literal `45`. |
| `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` (cell 2) | Added `ARTIFACT_VERSION = "_v2"`; every output path takes the suffix. |
| `notebooks/lawgic_taxonomy/lawgic_taxonomy.ipynb` (cell 22) | Added a permanent duplicate-supervision-column guard, and persists `lawgic_topic_order_v2.json` (the 42-topic prediction order). |
| `generated_files/lawgic_taxonomy/lawgic_topics_v2.json` (new) | `business_transfer` and `recommender_transparency` deleted (45→43 topics). `price_chg` removed from `payments`'s `source_mappings`. |
| `scripts/lawgic_eval_core.py` | `TOTAL_TAXONOMY_TOPICS`/`NUM_LAWGIC_TOPICS` now derived from `load_taxonomy()` instead of hardcoded. **Still points at the v1 corpus/checkpoint on purpose** — see the module's own NOTE comment. `active_topic_ids_44` renamed `active_topic_ids_predicted`. |
| `api/server.py` | Removed `NUM_TOPICS = 44`; topic count now read from the serving checkpoint's own label map before model construction. v3 checkpoint behavior unchanged. |
| `scripts/build_split_v2.py` (new) | Rebuilds the seed-42 stratified split for the v2 corpus, reusing `build_split_assignment()` from `lawgic_eval_core.py`. |
| `scripts/near_duplicate_split_audit_v2.py` (new) | Same, for the contamination audit, reusing functions from `near_duplicate_split_audit.py`. |

Headline deltas: 45→43 taxonomy topics, 44→42 predicted, 26,479→26,554 clauses,
21,183/2,648/2,648→21,243/2,655/2,656 split. Full before/after tables and manuscript-transfer list
in the linked doc.

## 11. v4 checkpoint trained — 2026-08-25

The v2 corpus (§10) was used to retrain Legal-BERT dual-head, producing the **v4 checkpoint**
(`saved_models/lawgic_classifier_legal-bert_v4`). Training protocol identical to v3 — same
hyperparameters, same seed, same architecture — only the corpus changed (42-topic taxonomy
instead of 44).

### v3 → v4 test-set comparison

Both models early-stopped at epoch 17 (patience 3 on `macro_f1`).

| Metric | v3 (44 topics) | v4 (42 topics) | Δ |
| --- | --- | --- | --- |
| **Topic macro F1** | 0.7542 | **0.7772** | **+0.0230** |
| Topic micro F1 | 0.8340 | 0.8360 | +0.0020 |
| Topic weighted F1 | 0.8327 | 0.8370 | +0.0043 |
| Predicted positive rate | 0.0482 | 0.0509 | +0.0027 |
| **Harm accuracy** | 0.8285 | **0.8535** | **+0.0250** |
| Harm macro F1 | 0.8251 | 0.8476 | +0.0225 |
| Harm weighted F1 | 0.8292 | 0.8534 | +0.0242 |
| Test loss | 1.3320 | 1.1633 | −0.1687 |
| Masked positions (test) | 96,838 | 91,863 | −4,975 |

### What drove the topic macro-F1 gain (+2.3 pp)

Three mechanisms, all traceable to §10 fixes:

1. **`indemnification`**: 0.00 → 0.56 F1 (+0.56). The `ind` alias (§10.2) gave it 76
   supervised positives. Still low-support, but no longer unlearnable. This alone lifts
   macro F1 by ~0.56/42 ≈ +1.3 pp.
2. **Duplicate deletion** removed `business_transfer` (0.97 F1) and
   `recommender_transparency` (0.73 F1) from the average. Their surviving partners
   (`transfer_of_contract` 0.97, `transparency` 0.77) improved slightly, freed from
   sharing identical supervision columns.
3. **`price_chg` decontamination** (§10.3) cleaned the `payments` column, which held
   at 0.92 F1 while `price_changes` rose from 0.53 to 0.63.

### Per-topic F1 movers (|Δ| ≥ 0.05, both present in v3 and v4)

| Topic | v3 F1 | v4 F1 | Δ | Note |
| --- | --- | --- | --- | --- |
| `indemnification` | 0.000 | 0.556 | **+0.556** | alias fix |
| `user_participation_in_changes` | 0.000 | 0.200 | +0.200 | still very low support (4) |
| `service_changes` | 0.300 | 0.385 | +0.085 | low support (12) |
| `feedback_reuse` | 0.833 | 0.902 | +0.069 | |
| `logs` | 0.485 | 0.550 | +0.065 | low support (19) |
| `right_to_leave` | 0.699 | 0.762 | +0.063 | |
| `anonymity` | 0.618 | 0.677 | +0.059 | |
| `interpretation_clause` | 0.729 | 0.784 | +0.055 | |
| `severability` | 0.933 | 0.857 | −0.076 | support 8→7, noise |

### Harm head per-class detail

| Class | v3 F1 | v4 F1 | Δ |
| --- | --- | --- | --- |
| Harmful (−1) | 0.79 | 0.82 | +0.03 |
| Neutral (0) | 0.86 | 0.88 | +0.02 |
| Fair (+1) | 0.83 | 0.84 | +0.01 |

All three classes improved. Neutral gained the most rows (+75) from the corpus rebuild;
the improvement is proportional.

### Parameters

v3: 109,518,383 trainable. v4: 109,516,845 trainable. Difference: −1,538 (exactly
`768 × 2 + 2`, the two deleted topic-head outputs).

### What this means for downstream

- **Phase 1–3 evaluation notebooks** still target the v3 checkpoint and v1 corpus by
  default (`LAWGIC_CORPUS_VERSION` unset). Those results remain valid as the "v1 baseline".
  Re-running them with `LAWGIC_CORPUS_VERSION=v2` against v4 is a separate task.
- **`api/server.py`** already defaults to v4 (§10).
- **`lawgic-tos-changes`** calls `/api/analyze_tos`, so it now uses v4 classification
  context in the diff prompt. No code change needed.
