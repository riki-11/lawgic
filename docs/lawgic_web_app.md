# The Lawgic Web Application and Inference API

**Status: work in progress.** This report is a lighter companion to the dataset and fine-tuning reports. It documents the current state of the deployment layer — the Lawgic inference API and the `lawgic-tos-changes` web application — and the problem that application is built to solve. Implementation details below reflect the current prototype and will change.

**Inference API:** `api/server.py`, `api/llm_interpreter.py` (this repository), documented in `docs/lawgic_api.md`.
**Web application:** `~/Coding Projects/Thesis/lawgic-tos-changes` (Next.js).

---

## 1. The Point of the Application

The fine-tuning report ended by noting that the web application requires exactly what the dual-head model produces: a per-clause verdict naming the contractual mechanisms present and grading their consumer harm. This report states what the application does with that capability.

Existing ToS-assessment tools evaluate a document **in isolation**: given one Terms of Service, they flag which clauses are unfair. That framing serves a *new* user deciding whether to sign up. It does not serve the far more common situation of an **existing** user of a service whose Terms were already accepted and are subsequently changed. Such a user has no practical way to know what changed, whether the change affects their rights, or whether it matters given how they actually use the service.

The Lawgic web application targets that gap. Its purpose is not "is this ToS fair?" but **"what changed in this ToS since the version you agreed to, and does the change matter to you?"** It compares two versions of a document, isolates the substantive differences, grades each difference by consumer harm, and explains it in plain language personalised to the individual user. This shift — from static document assessment to **change assessment for returning users** — is the application's contribution, and it is what the dual-head classifier's topic-plus-harm output was designed to feed.

---

## 2. Two Cooperating Systems

The deployment is split across two processes with distinct responsibilities:

| System | Stack | Responsibility |
| --- | --- | --- |
| **Lawgic inference API** | Python / FastAPI (default `localhost:8000`) | Serves the fine-tuned Legal-BERT dual-head model. Classifies clauses; optionally explains harmful clauses. |
| **Web application** | Next.js (`lawgic-tos-changes`) | User-facing flow: version discovery, diffing, personalised presentation. Orchestrates calls to the inference API. |

The separation is deliberate. The classifier is Python-bound (PyTorch, HuggingFace, the saved `lawgic_classifier_legal-bert_v3` weights) and is wrapped once as an HTTP service so the JavaScript frontend consumes real model predictions rather than re-implementing inference. The FastAPI process loads the model **once** at startup into a module-level global; requests pay only inference time.

---

## 3. The Lawgic Inference API

The API exposes the notebook inference pipeline (`document_inference_pipeline.ipynb`) as HTTP. Its core is `LawgicDualHeadModel` and the same five-stage document pipeline described in the fine-tuning report — read/normalise, paragraph-segment, token-budget/sub-chunk, batched inference, structured assembly — inlined into `api/server.py`.

Two endpoints:

1. **`POST /api/analyze_tos`** — accepts an uploaded `.txt` ToS, runs Legal-BERT, returns per-clause `clauses[]` with predicted topics and harm classes.
2. **`POST /api/explain_tos_scores`** — accepts classified clauses plus a `harm_filter`, and enriches only the matching clauses (default: **Harmful** only) with plain-language titles and descriptions via a local Ollama model, returning `points[]` for UI cards. Explanation is isolated in `api/llm_interpreter.py` so the LLM integration stays separate from the classifier.

CORS is whitelisted to the frontend origin rather than wildcarded. A legacy `GET /api/test-analyze` runs a fixture file for integration testing.

---

## 4. The Web Application Flow

The Next.js app replaces an earlier hardcoded single-service prototype with a generic, user-supplied flow. The user pastes any ToS or Privacy Policy URL, selects **how they use the service** (a role), and optionally adds free-text context. The app then:

1. **Discovers two versions.** `pages/api/fetch-tos.js` fetches the live current page server-side, then finds a prior version — either a native "previous version" link on the page (`lib/tos-version-discovery.js` recognises date-stamped legal URLs and archive links) or the most recent Wayback Machine snapshot. All third-party HTTP happens server-side; the browser never contacts external URLs directly.
2. **Classifies both versions.** Old and new documents are each sent through `pages/api/analyze.js`, which proxies to the FastAPI `/api/analyze_tos` endpoint. This yields Legal-BERT clause-level topic and harm classifications for each version.
3. **Pairs and diffs sections.** `lib/tos-chunking.js` aligns the two documents into comparable chunk pairs. Pairs that are not materially different (formatting, renumbering, whitespace) are discarded by a similarity check before any LLM call. Remaining pairs are diffed by a local Ollama model (`lib/ollama-diff.js`), prompted to report only substantive changes to rights, obligations, permissions, or restrictions — personalised to the user's role and context.
4. **Grounds the diff in the classifier.** The Legal-BERT clause classifications from step 2 are injected into the diff prompt as *additional context*, so the language model's change detection is anchored to the model's topic/harm assessment rather than operating on raw text alone.
5. **Merges, grades, and summarises.** `lib/merge-diff-results.js` deduplicates near-identical changes (Jaccard similarity) and sorts them by harm (`harmful → neutral → fair`) then change type (`modified → added → removed`). A final Ollama pass produces an overall summary. The result is rendered as ranked change cards in a three-step UI (profile → progress → results).

The net output is a prioritised, plain-language list of what changed between the two versions, hardest-hitting changes first, framed for the specific user.

---

## 5. How the Model and the Application Meet

The dual-head model contributes to the application in two concrete ways, both flowing from its topic-plus-harm output:

- **Grounding.** Clause-level topic and harm classifications are passed as structured context to the LLM performing the version diff, tying free-form change detection to the taxonomy-based assessment the thesis validated.
- **Prioritisation.** The harm classes (`Harmful` / `Neutral` / `Fair`) provide the ordering key that surfaces the most consequential changes first, rather than presenting a flat, unranked diff.

Without the classifier, the application would be a generic text-diff-plus-LLM tool. The classifier is what makes the diff *risk-aware*: it lets the application say not only that a clause changed, but that the change touches, for example, `limitation_of_liability` and moved in a harmful direction.

---

## 6. Current Limitations

As a work in progress, the deployment carries open items:

- **Two LLM paths coexist.** The FastAPI `explain_tos_scores` route and the Next.js Ollama diff both call local LLMs for plain-language output. In the current change-analysis flow the frontend drives its own diff-and-explain path and uses the FastAPI service primarily for classification; consolidating these is outstanding.
- **Version discovery is best-effort.** Reliance on native "previous version" links and the Wayback Machine means some services yield no usable prior version, or a poorly aligned one.
- **Local model dependency.** Both Ollama and the FastAPI classifier run locally; the system is not yet packaged for hosted deployment.
- **Personalisation is prompt-level only.** Role and context influence the LLM prompt but are not yet used to filter or weight changes structurally.

These are expected at the prototype stage. The full treatment of the web application — its interface, ingestion path, and evaluation — is the subject of the next section.
</content>
