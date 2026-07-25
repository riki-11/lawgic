# Lawgic

Lawgic is a thesis research project for analyzing Terms of Service (ToS) documents. It fuses annotations from multiple legal corpora into a unified 44-topic taxonomy, fine-tunes Legal-BERT for clause-level topic detection and consumer harm scoring, and exposes local inference to a web frontend via a FastAPI server.

## Frontends

| App | Status | Role |
|---|---|---|
| [`lawgic-tos-changes`](../lawgic-tos-changes) (Next.js) | **Active — the thesis artifact** | ToS version-diff flow. Calls `POST /api/analyze_tos` for classification and runs its own Ollama prompts (`lib/ollama-diff.js`, `gemma4:31b-cloud`) for plain-language change explanations. All experiments and the defense demo use this app. |
| [`lawgic-web-app`](../lawgic-web-app) (Vite/React) | **⚠️ Deprecated** | Original clause-card UI. The only consumer of `POST /api/explain_tos_scores`. Kept for provenance; do not measure or cite it. |

The core classifier is a **dual-head model**: one head predicts which Lawgic topics appear in a clause (multi-label), the other predicts harm level — Harmful, Neutral, or Fair (multi-class).

## Repository layout

| Folder | Contents |
|---|---|
| [`data/`](data/) | Committed reference data: sample/new ToS `.txt` files for testing (`sample_tos/`, `new_tos/`) and JSON schema definitions (`schema/`). |
| [`datasets/`](datasets/) | Raw downloaded datasets (gitignored). Place external corpora here — e.g. 100 ToS, CUAD, ToS;DR exports. Not committed to the repo. |
| [`docs/`](docs/) | Methodology writeups, architecture notes, and operational guides. See [`docs/lawgic_api.md`](docs/lawgic_api.md) for the full API reference. |
| [`generated_files/`](generated_files/) | Pipeline outputs: fused taxonomy tables, parsed annotations, evaluation CSVs, and per-service JSON extracts (`100_tos/`, `lawgic_taxonomy/`, `tos_dr/`, etc.). |
| [`notebooks/`](notebooks/) | Jupyter workflows — data exploration, taxonomy fusion, Legal-BERT finetuning, and the document inference pipeline. Dependencies in [`notebooks/requirements.txt`](notebooks/requirements.txt). |
| [`api/`](api/) | FastAPI server (`server.py`) that serves model predictions to the frontend. |
| `saved_models/` | Fine-tuned model weights (gitignored). Produced by the dual-head finetuning notebook. |

## Running the Lawgic API

The API pre-loads the dual-head model and exposes:
- `GET /api/test-analyze` — integration test against hardcoded Apollo.io ToS
- `POST /api/analyze_tos` — real `.txt` upload with dynamic `service_name` (BERT-only)
- `POST /api/explain_tos_scores` — plain-language titles/descriptions for harm-filtered clauses (Ollama). **Legacy:** only the deprecated `lawgic-web-app` called this. `lawgic-tos-changes` generates its explanations in-app and uses this server for classification only.

**Prerequisites:** `thesis-env` conda environment and local weights at `saved_models/lawgic_classifier_legal-bert_v3/`.

From the repo root:

```bash
/Users/riki/anaconda3/envs/thesis-env/bin/python3 -m uvicorn api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Test it:

```bash
curl http://localhost:8000/api/test-analyze
```

The server listens on **port 8000** (avoids conflicts with Vite on `5173` and Ollama). CORS is configured for `http://localhost:5173`.

For endpoint schema, troubleshooting, and architecture details, see [`docs/lawgic_api.md`](docs/lawgic_api.md).

## Dataset versions

- **v1** — [Annotated Terms of Service of 100 Online Platforms](https://data.mendeley.com/datasets/dtbj87j937/3)
- **v2** — [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad)
