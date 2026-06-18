# Lawgic

Lawgic is a thesis research project for analyzing Terms of Service (ToS) documents. It fuses annotations from multiple legal corpora into a unified 44-topic taxonomy, fine-tunes Legal-BERT for clause-level topic detection and consumer harm scoring, and exposes local inference to a React frontend via a FastAPI server.

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

The API pre-loads the dual-head model and exposes a test endpoint that analyzes a hardcoded Apollo.io ToS file.

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
