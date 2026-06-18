# Document-to-Clause Processing Pipeline: Text Ingestion Methodology

This document provides a comprehensive architectural trace of the Lawgic document-level inference pipeline, which extends the dual-head Legal-BERT classifier from isolated clause analysis to full Terms of Service (ToS) document processing. It is written for direct inclusion in the thesis methodology chapter.

## 1. Introduction & Motivation

The Lawgic dual-head model — a fine-tuned `nlpaueb/legal-bert-base-uncased` encoder with simultaneous topic presence (44-class multi-label) and consumer harm (3-class multi-class) classification heads — was validated on isolated, pre-segmented clauses from the training corpus. However, the practical utility of the classifier depends on its ability to process *entire* ToS documents as they appear in the wild: unsegmented, arbitrarily long, and structurally heterogeneous.

The document inference pipeline bridges this gap by implementing a complete ingestion-to-report pathway that:

1. Accepts raw `.txt` ToS files as input
2. Segments them into clause-level units suitable for the model's context window
3. Runs batched inference with hardware acceleration
4. Produces structured, per-clause risk and topic reports

This pipeline serves as the proof-of-concept for end-to-end local inference, demonstrating that the fine-tuned model can be applied to brand-new, unseen legal documents outside the training distribution.

## 2. Document Ingestion Strategy

### 2.1 File Format Scope

The current iteration targets plain text (`.txt`) files exclusively. This constraint was chosen for several reasons:

- **Eliminates parsing complexity**: `.txt` files contain no markup, embedded objects, or formatting directives that would require a secondary parser (as HTML or PDF would)
- **Deterministic encoding**: Text files with explicit UTF-8 encoding provide unambiguous character-to-byte mappings
- **Sufficient for thesis demonstration**: The goal is to validate the inference pipeline's viability, not to build a production document ingestion system

### 2.2 Encoding & Normalization

The pipeline applies two normalization passes to the raw file content:

**UTF-8 Encoding with Graceful Fallback.** Files are read with `encoding="utf-8"` and `errors="replace"`. The `replace` error handler substitutes any malformed byte sequences with the Unicode replacement character (U+FFFD) rather than raising an exception. This ensures the pipeline never crashes on encoding issues while making any data corruption visible in the output.

**Line-Ending Normalization.** Windows-originated files use `\r\n` (carriage return + line feed) as line terminators, while Unix/macOS systems use `\n`. Since the paragraph segmentation stage (§3) relies on `\n\n` as a delimiter, inconsistent line endings would cause missed paragraph boundaries. The pipeline normalizes all `\r\n` sequences to `\n` before any further processing.

## 3. Segmentation Architecture

### 3.1 Primary Segmentation: Double-Newline Paragraph Splitting

Legal documents follow well-established formatting conventions where double-newline boundaries (`\n\n`) separate distinct provisions, clauses, or itemized rights. This convention holds across ToS documents from diverse sources (technology companies, financial services, social media platforms) and was consistently observed in the training data sources (ToS;DR, 100 ToS, CLAUDETTE).

The pipeline splits the normalized document text on `\n\n` boundaries, producing a list of paragraph-level fragments. Each fragment is stripped of leading and trailing whitespace.

### 3.2 Skip Detection: Structural vs. Substantive Paragraphs

Not all double-newline-separated fragments contain substantive legal clauses. A ToS document typically includes:

- **Section headers** (e.g., `"DEFINITIONS."`, `"ARBITRATION AGREEMENT, CLASS ACTION WAIVER AND APPLICABLE LAW"`)
- **Date stamps** (e.g., `"Last Updated: February 5, 2026"`)
- **Instructional text** (e.g., `"PLEASE READ THIS SECTION CAREFULLY."`)

These fragments are too short or structurally distinct to yield meaningful classifier predictions. The pipeline applies a minimum character length threshold (`MIN_CLAUSE_LENGTH = 15`) to identify non-substantive fragments.

**Critically, these fragments are not discarded.** They are retained in the output with a `skipped = True` flag, preserving the full document structure for downstream auditing and traceability. This design ensures that a human reviewer can see the complete document alongside the model's predictions, with clear indication of which segments were analyzed and which were bypassed.

### 3.3 Defensive Token Budgeting

#### The Context Window Constraint

Legal-BERT has a maximum context window of 512 positional embeddings. However, the Lawgic dual-head model was fine-tuned with `max_length = 256` tokens (as recorded in `training_metadata.json`). Using a longer sequence at inference time would expose the model to positional embeddings it was never trained on, potentially degrading prediction quality.

#### Training Data Token Distribution

Analysis of the training corpus (26,479 annotated clauses) reveals the following token length profile:

| Percentile | Token Count |
|---|---|
| 50th (median) | 38 |
| 95th | 137 |
| 99th | 268 |
| Maximum | 3,579 |

This distribution shows that 99% of training clauses naturally fit within a 268-token window. However, the 3,579-token maximum confirms that wall-of-text paragraphs *do* exist in real-world ToS documents and must be handled defensively.

#### Threshold Derivation

The defensive chunking threshold is set at **200 tokens**, derived as follows:

```
Training max_length:                     256 tokens
- [CLS] token:                            -1 token
- [SEP] token:                            -1 token
- WordPiece inflation buffer (~15-25%):  ~-54 tokens
= Safe content token budget:             ≈200 tokens
```

The WordPiece inflation buffer accounts for the fact that the BERT tokenizer's subword segmentation can split a single English word into multiple tokens (e.g., "indemnification" → "in", "##de", "##mn", "##ification"), making the token count consistently higher than the raw word count. A 15–25% inflation rate is typical for legal text, which contains specialized vocabulary with high subword segmentation rates.

### 3.4 Sub-Chunking Fallback: Sentence-Level Splitting

Paragraphs exceeding the 200-token threshold undergo a secondary segmentation pass:

1. **Sentence tokenization**: The paragraph is split into individual sentences using `nltk.sent_tokenize()`, which applies the Punkt sentence boundary detection algorithm. This NLP-driven approach handles legal text's complex punctuation patterns (e.g., abbreviations like "e.g.", "i.e.", section references like "Section 7(A)(v)") more reliably than simple regex-based splitting.

2. **Greedy recombination**: Sentences are sequentially accumulated into sub-chunks. When adding the next sentence would cause the sub-chunk to exceed the 200-token threshold, the current sub-chunk is finalized and a new one begins. This greedy strategy maximizes the context available to the model in each sub-chunk.

3. **Single-sentence overflow**: If an individual sentence itself exceeds the 200-token threshold (rare but possible in legal text with deeply nested subordinate clauses), it is placed into its own sub-chunk. The tokenizer's `truncation=True` setting at the inference stage will truncate it to `max_length=256`, ensuring no memory overflow. This is the safest fallback: the model receives as much of the sentence as it can handle, and truncation matches the training-time behavior.

**Key invariant: no sentence is ever split mid-phrase.** The sub-chunking algorithm operates at the sentence boundary level, preserving semantic coherence within each chunk.

### 3.5 Output of the Chunking Stage

Each inference-ready chunk carries metadata for downstream traceability:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `int` | Global sequential index across all chunks |
| `paragraph_id` | `int` | Source paragraph's index in the document |
| `text` | `str` | The chunk text content |
| `estimated_tokens` | `int` | Exact token count (content tokens only, no specials) |
| `is_subchunk` | `bool` | `True` if the source paragraph was split into multiple chunks |

## 4. Inference Architecture

### 4.1 Hardware Acceleration Cascade

The pipeline dynamically detects the best available compute device:

1. **CUDA** (NVIDIA GPU): Preferred for maximum throughput via parallel tensor operations
2. **MPS** (Apple Metal Performance Shaders): Available on macOS with Apple Silicon (M-series) chips, providing GPU acceleration without CUDA
3. **CPU**: Fallback when no GPU is available

All tensors (model weights, tokenized inputs) are moved to the detected device before inference. The model's `state_dict` is loaded with `map_location=device` to avoid unnecessary CPU-to-GPU transfers.

### 4.2 Batch Collation Strategy

Chunks are processed in batches of 16 (configurable via `BATCH_SIZE`) to exploit parallel tensor computation. For each batch:

1. **Tokenization**: The HuggingFace tokenizer converts text to input tensors with `padding=True` (pad to the longest sequence in the batch), `truncation=True` (hard cap at `max_length=256`), and `return_tensors="pt"` (PyTorch format)
2. **Device transfer**: `input_ids`, `attention_mask`, and optionally `token_type_ids` are moved to the active device
3. **Forward pass**: Executed within `torch.no_grad()` to disable gradient computation, reducing memory usage and improving throughput
4. **Output extraction**: The model returns `(topic_logits, harm_logits)` — raw, unactivated scores

### 4.3 Activation Routing

The two classification heads require different activation functions:

**Topic Head → Sigmoid.** Each of the 44 topic dimensions is treated as an independent binary classification. The sigmoid function maps each logit to a `[0, 1]` probability, where values above the configurable `TOPIC_THRESHOLD` (default 0.5) indicate predicted presence. Multiple topics can be simultaneously active for a single clause (multi-label).

**Harm Head → Softmax.** The three harm classes (Harmful, Neutral, Fair) are mutually exclusive. The softmax function normalizes the three logits into a probability distribution that sums to 1. The predicted class is the argmax of this distribution.

### 4.4 Configurable Topic Threshold

The topic decision threshold is exposed as a pipeline-level parameter (`TOPIC_THRESHOLD`, default 0.5), matching the training-time `decision_threshold` recorded in `training_metadata.json`. Users can adjust this value to control the sensitivity of topic detection:

- **Lower threshold** (e.g., 0.3): More permissive — detects more topics but increases false positive risk
- **Higher threshold** (e.g., 0.7): More conservative — higher precision but may miss marginal topics

## 5. Output Schema

### 5.1 Per-Chunk Result Dictionary

Each processed chunk produces a structured result dictionary:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `int` | Global chunk index |
| `paragraph_id` | `int` | Source paragraph index |
| `text` | `str` | Full chunk text |
| `estimated_tokens` | `int` | Content token count |
| `is_subchunk` | `bool` | Whether the source paragraph was split |
| `topic_probabilities` | `dict[str, float]` | All 44 topic names → sigmoid probabilities |
| `predicted_topics` | `list[str]` | Topics exceeding the threshold |
| `harm_probabilities` | `dict[str, float]` | Harm class names → softmax probabilities |
| `predicted_harm_class` | `str` | Dominant harm class ("Harmful" / "Neutral" / "Fair") |
| `harm_confidence` | `float` | Softmax probability of the predicted class |

### 5.2 DataFrame Schema

The final report DataFrame merges inference results with skipped paragraphs:

| Column | Inference Rows | Skipped Rows |
|---|---|---|
| `chunk_id` | Sequential integer | `NaN` |
| `paragraph_id` | Source paragraph index | Source paragraph index |
| `text_preview` | First 100 characters | First 100 characters |
| `predicted_topics` | Comma-separated topic names | `NaN` |
| `harm_class` | "Harmful" / "Neutral" / "Fair" | `NaN` |
| `harm_confidence` | `[0, 1]` float | `NaN` |
| `is_subchunk` | `True` / `False` | `False` |
| `skipped` | `False` | `True` |
| `estimated_tokens` | Integer token count | `NaN` |

### 5.3 Summary Statistics

The pipeline produces aggregate statistics including:
- Total paragraph count (including skipped)
- Number of inference chunks produced
- Number of paragraphs that required sub-chunking
- Harm class distribution across inference chunks
- Top 10 most frequently predicted topics

## 6. Real-World Validation: Apollo.io Terms of Service

The pipeline was validated against Apollo.io's Terms of Service (`data/new_tos/apollo_io.txt`), a 188-line, ~56KB production document from Zenleads Inc. d/b/a Apollo.io. This document was selected because it exhibits the structural challenges typical of enterprise SaaS ToS agreements:

- **Long definition blocks**: The "DEFINITIONS" section (lines 19–35) contains single paragraphs spanning 3–4 sentences of dense legal terminology
- **Wall-of-text warranty disclaimer**: Line 119 contains a single paragraph of approximately 200+ words in ALL CAPS, which will exceed the 200-token threshold and trigger the sub-chunking fallback
- **Multi-paragraph arbitration section**: Lines 122–153 contain the arbitration agreement, class action waiver, and bellwether arbitration provisions — a mixture of short procedural headers and extremely long substantive paragraphs
- **Indemnification clause**: Line 155 contains the standard indemnification provision, which the model previously analyzed in isolation via `test_legal_bert.ipynb`

The Apollo.io document serves as a critical end-to-end test because several of its paragraphs were independently tested as isolated clauses in the model validation notebook, enabling direct comparison between isolated-clause and pipeline-derived predictions.

## 7. Limitations & Future Work

### 7.1 Current Limitations

**Text-only scope.** The pipeline processes `.txt` files exclusively. Real-world ToS documents frequently appear as HTML pages (with embedded navigation, footers, and cookie banners), PDFs (with headers, page numbers, and multi-column layouts), or DOCX files. Each format requires a dedicated parser.

**Paragraph-level segmentation heuristic.** The `\n\n` splitting heuristic assumes consistent double-newline boundaries between clauses. Some documents may use single newlines, numbered lists, or other formatting conventions that would require more sophisticated segmentation logic.

**No sub-chunk aggregation.** When a paragraph is split into multiple sub-chunks, each chunk receives independent predictions. The pipeline does not currently aggregate these predictions back to the paragraph level. For thesis reporting purposes, the per-chunk granularity is sufficient; however, a production system may need aggregation strategies (e.g., union of predicted topics, maximum harm score across sub-chunks).

**Un-aggregated output.** The pipeline produces per-clause predictions without document-level aggregation (e.g., overall document risk score, topic coverage summary). This is intentional for the current thesis scope — demonstrating per-clause prediction viability — but would be a natural next step for a user-facing application.

### 7.2 Future Extensions

- **PDF and HTML ingestion**: Integrate document parsing libraries (e.g., `pdfplumber`, `beautifulsoup4`) with format-specific normalization
- **Sub-chunk prediction aggregation**: Implement paragraph-level rollup strategies for topic union and harm score resolution
- **Threshold sensitivity analysis**: Systematically evaluate prediction quality across different `TOPIC_THRESHOLD` values using the annotated training corpus as ground truth
- **Document-level risk scoring**: Aggregate per-clause predictions into an overall document risk profile
- **Batch document processing**: Scale the pipeline to process multiple ToS files in sequence with consolidated reporting
