# Fine-Tuning Legal-BERT for Lawgic: A Dual-Head Approach

**Model artifact:** `saved_models/lawgic_classifier_legal-bert_v3/`
**Training notebook:** `notebooks/model_finetuning/legal_bert_finetuning_dual_head.ipynb`
**Inference pipeline:** `notebooks/lawgic_pipeline/document_inference_pipeline.ipynb`
**Training data:** `generated_files/lawgic_taxonomy/lawgic_multihead_wide.csv`

This report continues directly from the dataset report. That report ended with a specific claim: the multi-head design was forced by the data, because a single-head fusion cannot represent an observed absence and therefore contains no negatives to learn from. This report takes the corrected multi-head corpus as given and documents the model built on top of it — what base model was extended, how it was extended, how it was trained, what it achieves, and how it is served to the Lawgic web application. The account is deliberately matter-of-fact: the object is to state precisely how this model differs from the single-head ToS classifiers in the literature and what that difference produces, not to argue that the approach is optimal.

---

## 1. From a Fused Corpus to a Model

The dataset report produced `lawgic_multihead_wide.csv`: one row per unique normalised clause, carrying two independent supervision signals. The first is a 44-dimensional topic-presence vector `labels_presence` paired with a source-aware `topic_mask` that marks which topic cells are supervised (positive or negative) versus unknown. The second is a single row-level `harm_score_class ∈ {0, 1, 2}` (`Harmful`, `Neutral`, `Fair`) with a `harm_mask` gating rows whose harm score is resolvable.

Two supervision signals of different type — a masked multi-label target and a single-label multi-class target — cannot be consumed by a conventional single-head classifier. The corpus therefore dictates the model shape: one encoder, two heads, two losses. Section 3 makes that architecture concrete; first, the encoder it extends.

---

## 2. Legal-BERT: The Base Model

The encoder is **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`), released by the AUEB NLP group (Chalkidis et al., 2020). It is a BERT-base model — 12 transformer layers, 768 hidden dimensions, ~110M parameters — that is not the general-purpose BERT but a **domain-adapted** variant: it was pre-trained from scratch on a large English legal corpus (EU and UK legislation, European Court of Justice and ECtHR case law, US court cases, and US contracts) rather than on general web and encyclopaedia text.

The domain adaptation matters for the task at hand. General BERT's masked-language-model pre-training under-represents the register of contractual English — the vocabulary of indemnification, severability, arbitration, and warranty disclaimers, and the long nominal clauses in which they appear. Legal-BERT's tokenizer and learned representations are fitted to exactly this register. It provides, out of the box, a clause encoder whose `[CLS]` pooler output already separates legal concepts that a general encoder would blur. Lawgic uses it purely as a **feature extractor over clause text**; the base model itself is not modified beyond fine-tuning its weights jointly with the two new heads.

What Legal-BERT does **not** provide is any notion of Lawgic's taxonomy or of consumer harm. It is a language model, not a classifier of ToS fairness. The contribution of this work is the structure placed on top of it and the supervision used to train that structure.

---

## 3. The Dual-Head Expansion

Lawgic replaces BERT's usual single classification head with **two independent linear heads on the shared `[CLS]` embedding**:

```
        clause text
            │
            ▼
   ┌──────────────────────┐
   │  Legal-BERT encoder  │   (frozen architecture, fine-tuned weights)
   └──────────┬───────────┘
              │ [CLS] pooler output  (batch, 768)
        ┌─────┴─────┐
        ▼           ▼
  ┌──────────┐ ┌──────────┐
  │Topic head│ │Harm head │
  │Lin(768,44)│ │Lin(768,3)│
  └────┬─────┘ └────┬─────┘
       ▼            ▼
  topic_logits  harm_logits
  (masked BCE)  (cross-entropy)
```

- **Topic head** — `nn.Linear(768, 44)`, multi-label. Sigmoid at inference, decision threshold 0.5. Trained with **masked** binary cross-entropy so that unknown topic cells (`mask = 0`) contribute zero loss. This is the mechanism that lets a corpus fused from sources with disjoint coverage train without manufacturing false negatives.
- **Harm head** — `nn.Linear(768, 3)`, multi-class over `{Harmful, Neutral, Fair}`. Softmax at inference, standard cross-entropy at training, gated by `harm_mask`.

The two heads are wired into an equally weighted joint objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{topic}} + \mathcal{L}_{\text{harm}}$$

with no task-weighting hyperparameter (λ = 1.0 each). Both losses backpropagate into the shared encoder, so a single set of clause representations is pressured to serve both objectives at once.

### Why a custom module was necessary

HuggingFace's `AutoModelForSequenceClassification` supports exactly one classification head and one loss function. It cannot express two heads of different type (multi-label vs multi-class) with two different losses. The implementation therefore wraps the raw `AutoModel` encoder in a custom `LawgicDualHeadModel` (`nn.Module`) exposing both linear heads, and subclasses the `Trainer` (`DualHeadTrainer`) to compute the joint loss and to carry the topic and harm masks through evaluation. This is an engineering consequence of the architecture, not an incidental choice: the standard wrapper is structurally incapable of the two-objective setup the corpus requires.

The full architectural specification — collator, mask-aware metric plumbing, and the `prediction_step` override required so the evaluation loop receives labels — is documented in `docs/lawgic_dual_head_architecture.md` and not repeated here.

---

## 4. Training Protocol

| Parameter | Value |
| --- | --- |
| Base model | `nlpaueb/legal-bert-base-uncased` |
| Heads | Topic `Linear(768,44)` · Harm `Linear(768,3)` |
| Max sequence length | 256 tokens |
| Learning rate | 3 × 10⁻⁵ |
| Batch size | 8 (train), 16 (eval) |
| Max epochs | 20, early stopping patience 3 on `eval_macro_f1` |
| Weight decay | 0.01 · Warmup ratio 0.06 |
| Loss weighting | Equal (1.0 + 1.0) |
| Precision | FP16 on CUDA |
| Seed | 42 |

**Split.** The corpus (26,479 rows) is divided 80/10/10 into **21,183 train / 2,648 validation / 2,648 test** rows by *multi-objective stratified sampling*: the stratification key concatenates each clause's primary active topic and its harm class (`f"{primary_topic}__harm{harm_class}"`), so both the topic distribution and the three-class harm distribution are preserved across splits. An **exact-text leakage check** aborts training if any clause string appears in more than one split — necessary because the corpus was fused from overlapping sources and identical clauses recur across services.

Training ran on CUDA, converging under early stopping at **epoch 17** with a best validation masked macro-F1 of **0.761**.

---

## 5. Verifying the Corpus Is Learnable: The Zero-Logit Check

Before training, a mandatory diagnostic tests whether the dataset fix from the dataset report actually took effect. The reasoning is exact: a model emitting all-zero logits predicts every topic present (`sigmoid(0) = 0.5 ≥ 0.5`). On the *old* degenerate corpus, which contained only positives, that all-positive prediction would score a **perfect macro-F1 of 1.0**. On the corrected corpus, which contains supervised negatives, the same all-positive prediction must score **well below 1.0**, because it is now wrong on every negative cell. The notebook asserts `macro_f1 < 0.95` and raises `DEGENERATE` otherwise.

The corrected corpus passes: the zero-logit topic macro-F1 falls far below the 0.95 tripwire, and the zero-logit harm head collapses to the majority class at ≈31% accuracy (the `Harmful` base rate). This is the empirical vindication of the dataset report's argument — the negatives are real, the task is non-trivial, and any performance the trained model shows is earned rather than an artifact of a degenerate objective.

---

## 6. Results

All test metrics are computed mask-aware: topic metrics count only supervised cells (`mask = 1`), harm metrics only rows with a valid harm label.

### Topic head (test, 44 topics)

| Metric | Value |
| --- | --- |
| Masked macro-F1 | 0.754 |
| Masked micro-F1 | 0.834 |
| Masked weighted-F1 | 0.833 |
| Supervised positions evaluated | 96,838 |

### Harm head (test, 2,648 rows)

| Metric | Value |
| --- | --- |
| Accuracy | 0.829 |
| Macro-F1 | 0.825 |
| Weighted-F1 | 0.829 |

The gap between micro-F1 (0.834) and macro-F1 (0.754) on the topic head is entirely a **support** effect, and the per-topic breakdown makes the pattern unambiguous. Well-supported topics are learned strongly: `business_transfer` and `transfer_of_contract` reach 0.97, `notice_of_changes` 0.93, `trackers` 0.90, `class_action_waiver` 0.90, `warranty_disclaimer` 0.90, with `contract_by_use`, `governance`, `complaint_system`, and the dispute-resolution topics all in the high 0.80s. The macro average is dragged down by a small set of rare topics:

| Topic | Test support | F1 |
| --- | --- | --- |
| `indemnification` | 0 | 0.00 |
| `user_participation_in_changes` | 4 | 0.00 |
| `service_changes` | 12 | 0.30 |
| `liability_cap` | 7 | 0.43 |
| `logs` | 19 | 0.49 |
| `price_changes` | 5 | 0.53 |

These are precisely the topics the dataset report flagged as thinly annotated — the fine subtypes that only one source (100 ToS) supplies, and that even the pessimistic fusion could not populate densely. `indemnification` has **zero** positive test instances, so its F1 is undefined-as-zero rather than a genuine model failure. The result is honest: the model is competent on the topics for which the fused corpus carries real signal, and unreliable on the long tail where it does not. This is a data-coverage ceiling, not an architectural one, and it localises exactly where future annotation effort should go.

---

## 7. What the Architecture Is, and What It Adds

Stated plainly, the difference from prior ToS classifiers is structural:

1. **Two objectives from one encoder.** Existing work trains a single head over a single dataset's native taxonomy — CLAUDETTE's unfairness categories, or ToS;DR's topics — and predicts one thing. The Lawgic model predicts, from a single forward pass, both *which* of 44 contractual mechanisms a clause invokes (multi-label) and *how harmful* its treatment of them is (a three-level ordinal-style harm class). The two are learned jointly on shared representations rather than by two separate models.

2. **Supervision over a taxonomy broader than any single source.** Because the topic head is trained with masked BCE over the fused corpus, it classifies against the 44-topic Lawgic taxonomy — finer than CLAUDETTE's, broader than any one source on the privacy/governance axis — while never being penalised on topics a contributing source did not evaluate.

3. **A graded, clause-level harm signal.** The harm head outputs a per-clause severity (`Harmful` / `Neutral` / `Fair`) with a confidence, rather than a single document-level fairness verdict.

What this brings to the task is a clause-level output that is simultaneously categorical and severity-graded, produced by one model over a unified taxonomy. That combination is what the downstream application consumes; it is not available from a single-head classifier trained on any one of the source datasets alone.

---

## 8. From Model to Application: The Inference Pipeline

The trained model is served to the web application through `document_inference_pipeline.ipynb`, which turns a raw ToS document into a structured, clause-level report. The pipeline reloads `LawgicDualHeadModel` from the saved directory and runs five stages:

- **A — Read and normalise.** The document is read and NFKC-normalised, mirroring the exact normalisation used to build the training corpus, so inference-time text matches training-time text.
- **B — Paragraph segmentation.** Text is split on double-newline boundaries into clause-level paragraphs. Fragments shorter than 15 characters (section headers, date stamps) are retained with `skipped=True` for structural traceability but excluded from inference.
- **C — Defensive token budgeting.** Paragraphs whose estimated token count exceeds 200 (the 256-token training limit minus special tokens and a WordPiece-inflation buffer) are sub-chunked, so no clause is silently truncated below what the encoder saw in training.
- **D — Batched inference.** Chunks are tokenised and run through the dual head in batches of 16. The topic head's sigmoid outputs are thresholded at 0.5; the harm head's softmax yields a harm class and a confidence.
- **E — Structured report assembly.** Results are assembled into a per-clause table — predicted topics, harm class, harm confidence — with skipped structural rows preserved in document order, plus summary statistics (harm distribution and most-frequent topics across the document).

The pipeline enforces train/inference parity at every stage that could break it: identical normalisation, the same 256-token budget, the same 44-topic map, and the same decision threshold. Its output is the interface between the model and the application.

---

## 9. Toward the Lawgic Web Application

The consumer of this pipeline is the Lawgic web application (`~/Coding Projects/Thesis/lawgic-tos-changes`), a Next.js project that takes a user-supplied ToS document, runs it through the dual-head inference pipeline, and presents the per-clause topic and harm assessment in a readable interface. The mechanics of that application — its ingestion path, API surface, and presentation of the clause-level report — are the subject of the next section and are not elaborated here. The relevant point for the modelling methodology is that the application requires exactly what the dual-head model produces: a per-clause verdict that names the contractual mechanisms present and grades their consumer harm, over a single unified taxonomy.
</content>
