"""
Lawgic Inference API — lightweight FastAPI server for local Legal-BERT dual-head inference.

Pre-loads the model and document processor at import time, then exposes:
  - GET  /api/test-analyze       — hardcoded apollo_io.txt fixture (dev/integration test)
  - POST /api/analyze_tos        — real multipart .txt upload with dynamic service_name (BERT-only)
  - POST /api/explain_tos_scores — plain-language titles/descriptions for classified clauses

Run from repo root (use the thesis-env conda env that powers the notebooks):
    /Users/riki/anaconda3/envs/thesis-env/bin/python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import nltk
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

from api.llm_interpreter import (
    ClauseExplainError,
    OllamaExplainError,
    explain_clauses,
)

# ── Logging (replaces notebook print statements) ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Repo-root paths (resolved relative to this file, not cwd) ────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

# Load .env from repo root (Ollama URL/model for /api/explain_tos_scores)
load_dotenv(REPO_ROOT / ".env")
# Override with LAWGIC_MODEL_DIR to serve a different checkpoint, e.g.
# LAWGIC_MODEL_DIR=saved_models/lawgic_classifier_legal-bert_v3 for the old 44-topic model.
MODEL_DIR = Path(
    os.getenv("LAWGIC_MODEL_DIR")
    or REPO_ROOT / "saved_models" / "lawgic_classifier_legal-bert_v4"
)
if not MODEL_DIR.is_absolute():
    MODEL_DIR = REPO_ROOT / MODEL_DIR
TEST_TOS_PATH = REPO_ROOT / "data" / "new_tos" / "apollo_io.txt"

# Participant interaction telemetry (user-study exploratory measure). Local
# JSONL only, never transmitted elsewhere — see chapter_4.tex Ethical Considerations.
INTERACTION_LOG_PATH = REPO_ROOT / "generated_files" / "interaction_logs" / "events.jsonl"
_interaction_log_lock = threading.Lock()

# ── Inference parameters (must match training conditions) ────────────────────
MAX_LENGTH = 256
TOKEN_THRESHOLD = 200
TOPIC_THRESHOLD = 0.5
BATCH_SIZE = 16
NUM_HARM_CLASSES = 3
HARM_CLASS_NAMES = {0: "Harmful", 1: "Neutral", 2: "Fair"}
MIN_CLAUSE_LENGTH = 15


# =============================================================================
# Hardware detection — CUDA > MPS > CPU
# =============================================================================

def detect_device() -> torch.device:
    """Pick best available accelerator for inference. LAWGIC_DEVICE forces a choice."""
    forced = os.getenv("LAWGIC_DEVICE")
    if forced:
        logger.info("Using device: %s (forced via LAWGIC_DEVICE)", forced)
        return torch.device(forced)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: cuda (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: mps (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")
    return device


# =============================================================================
# Model architecture — dual-head Legal-BERT (copied from inference notebook)
# =============================================================================

class LawgicDualHeadModel(nn.Module):
    """Dual-head Legal-BERT model for simultaneous topic and harm classification."""

    def __init__(
        self,
        model_name: str,
        num_topics: int,
        num_harm_classes: int = NUM_HARM_CLASSES,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.topic_head = nn.Linear(hidden_size, num_topics)
        self.harm_head = nn.Linear(hidden_size, num_harm_classes)

        self.num_topics = num_topics
        self.num_harm_classes = num_harm_classes

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_embedding = outputs.pooler_output

        topic_logits = self.topic_head(cls_embedding)
        harm_logits = self.harm_head(cls_embedding)

        return topic_logits, harm_logits


# =============================================================================
# Model + tokenizer loading
# =============================================================================

def load_model_and_tokenizer(device: torch.device) -> tuple[LawgicDualHeadModel, object, dict[int, str]]:
    """
    Load tokenizer, dual-head model weights, and 44-topic label map.

    Raises FileNotFoundError if MODEL_DIR is missing (common on fresh clones
    because saved_models/ is gitignored).
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Run the finetuning notebook and ensure weights are saved locally."
        )

    logger.info("Loading tokenizer from %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Topic count comes from the checkpoint's own label map, not a hardcoded
    # constant -- accept either filename so a future non-44-topic checkpoint
    # (e.g. trained on the v2 42-topic corpus) drops in with no code change.
    topics_json_path = MODEL_DIR / "lawgic_topics_44.json"
    if not topics_json_path.exists():
        topics_json_path = MODEL_DIR / "lawgic_topic_order.json"
    with topics_json_path.open("r", encoding="utf-8") as f:
        topics_data = json.load(f)
    id2name = (
        {t["classifier_id"]: t["name"] for t in topics_data}
        if topics_data and isinstance(topics_data[0], dict)
        else dict(enumerate(topics_data))  # lawgic_topic_order.json: a flat id list
    )
    num_topics = len(id2name)

    logger.info("Loading dual-head model...")
    model = LawgicDualHeadModel(str(MODEL_DIR), num_topics=num_topics)

    state_dict_path = MODEL_DIR / "model_state_dict.pt"
    if state_dict_path.exists():
        model.load_state_dict(
            torch.load(state_dict_path, map_location=device, weights_only=True)
        )
    else:
        topic_weights = torch.load(
            MODEL_DIR / "topic_head_weights.pt", map_location=device, weights_only=True
        )
        harm_weights = torch.load(
            MODEL_DIR / "harm_head_weights.pt", map_location=device, weights_only=True
        )
        model.topic_head.load_state_dict(topic_weights)
        model.harm_head.load_state_dict(harm_weights)

    model = model.to(device)
    model.eval()

    logger.info(
        "Model loaded on %s | %d topics | %d harm classes",
        device,
        len(id2name),
        NUM_HARM_CLASSES,
    )

    return model, tokenizer, id2name


# =============================================================================
# Document pipeline — Steps A through D (copied from inference notebook)
# =============================================================================

def normalize_document_text(raw_text: str) -> str:
    """Normalize line endings and strip outer whitespace."""
    return raw_text.replace("\r\n", "\n").strip()


def read_document(file_path: Path) -> str:
    """Read a .txt file, normalize line endings, return stripped UTF-8 text."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    if file_path.suffix.lower() != ".txt":
        raise ValueError(f"Expected .txt file, got: {file_path.suffix}")

    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    normalized = normalize_document_text(raw_text)

    logger.info(
        "Document read: %s | %d characters",
        file_path.name,
        len(normalized),
    )
    return normalized


def segment_paragraphs(text: str, min_length: int = MIN_CLAUSE_LENGTH) -> list[dict]:
    """Split on double-newline boundaries; flag short fragments as skipped."""
    raw_fragments = text.split("\n\n")
    paragraphs = []
    paragraph_id = 0

    for fragment in raw_fragments:
        stripped = fragment.strip()
        if not stripped:
            continue

        paragraphs.append({
            "paragraph_id": paragraph_id,
            "text": stripped,
            "skipped": len(stripped) < min_length,
        })
        paragraph_id += 1

    skipped_count = sum(1 for p in paragraphs if p["skipped"])
    logger.info(
        "Segmented: %d paragraphs | %d substantive | %d skipped",
        len(paragraphs),
        len(paragraphs) - skipped_count,
        skipped_count,
    )
    return paragraphs


def estimate_tokens(text: str, tokenizer) -> int:
    """Exact token count without special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_paragraph(paragraph: str, tokenizer, threshold: int = TOKEN_THRESHOLD) -> list[str]:
    """Pass-through or sentence-level sub-chunking for oversized paragraphs."""
    token_count = estimate_tokens(paragraph, tokenizer)

    if token_count <= threshold:
        return [paragraph]

    sentences = nltk.sent_tokenize(paragraph)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence, tokenizer)

        if current_tokens + sentence_tokens > threshold and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_chunks(
    paragraphs: list[dict],
    tokenizer,
    threshold: int = TOKEN_THRESHOLD,
) -> list[dict]:
    """Convert substantive paragraphs into inference-ready chunk dicts."""
    all_chunks: list[dict] = []
    chunk_id = 0
    subchunked_paragraphs = 0

    for para in paragraphs:
        if para["skipped"]:
            continue

        sub_chunks = chunk_paragraph(para["text"], tokenizer, threshold)
        is_subchunked = len(sub_chunks) > 1

        if is_subchunked:
            subchunked_paragraphs += 1

        for chunk_text in sub_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "paragraph_id": para["paragraph_id"],
                "text": chunk_text,
                "estimated_tokens": estimate_tokens(chunk_text, tokenizer),
                "is_subchunk": is_subchunked,
            })
            chunk_id += 1

    logger.info(
        "Prepared: %d inference chunks | %d sub-chunked paragraphs",
        len(all_chunks),
        subchunked_paragraphs,
    )
    return all_chunks


def run_batch_inference(
    chunks: list[dict],
    model: nn.Module,
    tokenizer,
    device: torch.device,
    id2name: dict[int, str],
    batch_size: int = BATCH_SIZE,
    topic_threshold: float = TOPIC_THRESHOLD,
) -> list[dict]:
    """Run dual-head inference on all chunks in batches."""
    results: list[dict] = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_idx : batch_idx + batch_size]
        batch_texts = [c["text"] for c in batch_chunks]
        current_batch = batch_idx // batch_size + 1

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            topic_logits, harm_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        topic_probs = torch.sigmoid(topic_logits).cpu()
        harm_probs = torch.softmax(harm_logits, dim=-1).cpu()
        harm_preds = torch.argmax(harm_logits, dim=-1).cpu()

        for i, chunk in enumerate(batch_chunks):
            sample_topic_probs = topic_probs[i].tolist()
            topic_prob_dict: dict[str, float] = {}
            predicted_topics: list[str] = []

            for topic_idx, prob in enumerate(sample_topic_probs):
                topic_name = id2name.get(topic_idx, f"Topic_{topic_idx}")
                topic_prob_dict[topic_name] = round(prob, 4)
                if prob >= topic_threshold:
                    predicted_topics.append(topic_name)

            sample_harm_probs = harm_probs[i].tolist()
            harm_prob_dict = {
                HARM_CLASS_NAMES[idx]: round(p, 4)
                for idx, p in enumerate(sample_harm_probs)
            }

            harm_class_idx = harm_preds[i].item()
            predicted_harm = HARM_CLASS_NAMES.get(harm_class_idx, "Unknown")
            harm_confidence = sample_harm_probs[harm_class_idx]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "paragraph_id": chunk["paragraph_id"],
                "text": chunk["text"],
                "estimated_tokens": chunk["estimated_tokens"],
                "is_subchunk": chunk["is_subchunk"],
                "topic_probabilities": topic_prob_dict,
                "predicted_topics": predicted_topics,
                "harm_probabilities": harm_prob_dict,
                "predicted_harm_class": predicted_harm,
                "harm_confidence": round(harm_confidence, 4),
            })

        logger.info(
            "Batch %d/%d: processed %d chunks",
            current_batch,
            total_batches,
            len(batch_chunks),
        )

    logger.info("Inference complete: %d chunks processed", len(results))
    return results


# =============================================================================
# Document processor — thin wrapper over the pipeline
# =============================================================================

class LawgicDocumentProcessor:
    """End-to-end ToS document analyzer using the dual-head Legal-BERT model."""

    def __init__(
        self,
        model: LawgicDualHeadModel,
        tokenizer,
        device: torch.device,
        id2name: dict[int, str],
        token_threshold: int = TOKEN_THRESHOLD,
        batch_size: int = BATCH_SIZE,
        topic_threshold: float = TOPIC_THRESHOLD,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.id2name = id2name
        self.token_threshold = token_threshold
        self.batch_size = batch_size
        self.topic_threshold = topic_threshold

    def analyze_text(self, text: str, source_label: str = "upload") -> list[dict]:
        """
        Run the full ingestion-to-inference pipeline on raw ToS text.

        Returns raw per-chunk inference dicts from run_batch_inference.
        Skipped paragraphs (headers, short fragments) are excluded.
        """
        normalized = normalize_document_text(text)
        if not normalized:
            raise ValueError("Document is empty")

        logger.info("Analyzing text: %s | %d characters", source_label, len(normalized))

        paragraphs = segment_paragraphs(normalized)
        chunks = prepare_chunks(paragraphs, self.tokenizer, self.token_threshold)
        return run_batch_inference(
            chunks,
            self.model,
            self.tokenizer,
            self.device,
            id2name=self.id2name,
            batch_size=self.batch_size,
            topic_threshold=self.topic_threshold,
        )

    def analyze_tos(self, file_path: Path) -> list[dict]:
        """Run the full pipeline on a .txt ToS file on disk."""
        text = read_document(file_path)
        return self.analyze_text(text, source_label=file_path.name)


# =============================================================================
# Global pre-load — model loads once at import, before any routes are defined
# =============================================================================

# NLTK sentence tokenizer data required for sub-chunking fallback
nltk.download("punkt_tab", quiet=True)

device = detect_device()

try:
    model, tokenizer, id2name = load_model_and_tokenizer(device)
except FileNotFoundError as exc:
    logger.error("%s", exc)
    raise

processor = LawgicDocumentProcessor(model, tokenizer, device, id2name)


# =============================================================================
# FastAPI app — CORS middleware MUST be added before route declarations
# =============================================================================

app = FastAPI(
    title="Lawgic Inference API",
    description="Local Legal-BERT dual-head inference for ToS document analysis",
)

# Whitelist the Vite React dev server origin explicitly (no wildcard *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_api_response(
    results: list[dict],
    *,
    service_name: str,
    is_dynamic_upload: bool,
) -> dict:
    """Map raw inference results to the flat JSON schema expected by the frontend."""
    return {
        "service_name": service_name,
        "is_dynamic_upload": is_dynamic_upload,
        "clauses": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "predicted_topics": r["predicted_topics"],
                "harm_class": r["predicted_harm_class"],
                "harm_confidence": r["harm_confidence"],
            }
            for r in results
        ],
    }


@app.get("/api/model-info")
def model_info() -> dict:
    """Which checkpoint and device are actually serving. Evaluation notebooks record this."""
    metadata_path = MODEL_DIR / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "model_dir": MODEL_DIR.name,
        "model_path": str(MODEL_DIR),
        "device": str(device),
        "num_topics": len(id2name),
        "num_harm_classes": NUM_HARM_CLASSES,
        "training_metadata": metadata,
    }


@app.get("/api/test-analyze")
def test_analyze() -> dict:
    """
    Test endpoint: read hardcoded apollo_io.txt, run inference, return flat JSON.

    No file upload — uses data/new_tos/apollo_io.txt from the repo.
    """
    if not TEST_TOS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Test ToS file not found: {TEST_TOS_PATH}",
        )

    results = processor.analyze_tos(TEST_TOS_PATH)
    return format_api_response(
        results,
        service_name="Apollo.io (Test Upload)",
        is_dynamic_upload=True,
    )


@app.post("/api/analyze_tos")
async def analyze_tos_upload(
    file: UploadFile = File(...),
    service_name: str = Form(...),
) -> dict:
    """
    Analyze an uploaded .txt ToS file with Legal-BERT dual-head inference.

    Accepts multipart form data with a plain-text file and a display name for the service.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported",
        )

    service_name = service_name.strip()
    if not service_name:
        raise HTTPException(
            status_code=400,
            detail="service_name is required",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    text = raw_bytes.decode("utf-8", errors="replace")

    try:
        results = processor.analyze_text(text, source_label=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return format_api_response(
        results,
        service_name=service_name,
        is_dynamic_upload=True,
    )


# =============================================================================
# Score explanation route — plain-language point titles via Ollama
# Route name intentionally omits "llm" so the public API hides implementation.
# =============================================================================


class AnalyzeClauseInput(BaseModel):
    """One BERT-classified clause from POST /api/analyze_tos."""

    chunk_id: int = Field(description="Sequential chunk index from analyze_tos")
    text: str = Field(description="Verbatim clause text passed to Legal-BERT")
    predicted_topics: list[str] = Field(
        default_factory=list,
        description="Topic names with sigmoid probability >= 0.5",
    )
    harm_class: Literal["Harmful", "Neutral", "Fair"] = Field(
        description="Predicted consumer harm class from the harm head",
    )
    harm_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted harm class",
    )


class ExplainTosScoresRequest(BaseModel):
    """Request body for POST /api/explain_tos_scores."""

    service_name: str = Field(
        min_length=1,
        description="Display name of the analyzed service (echoed in response)",
    )
    clauses: list[AnalyzeClauseInput] = Field(
        min_length=1,
        description="clauses[] from a prior POST /api/analyze_tos response",
    )
    harm_filter: list[str] = Field(
        default_factory=lambda: ["harmful"],
        description=(
            "Which harm classes to explain and return. "
            "Aliases: bad/harmful/-1, neutral/0, good/fair/+1. Default: harmful only."
        ),
    )


class ExplainedPoint(BaseModel):
    """One harm-filtered clause enriched with a plain-language title and description."""

    chunk_id: int
    primary_topic: str = Field(description="First predicted topic, or unclassified")
    predicted_topics: list[str]
    harm_class: Literal["Harmful", "Neutral", "Fair"]
    harm_label: Literal["bad", "neutral", "good"] = Field(
        description="ToS;DR-style UI label derived from harm_class",
    )
    harm_confidence: float
    point_title: str = Field(description="Short plain-language takeaway for the UI card title")
    description: str = Field(description="1–3 sentence elaboration for the UI description block")
    quoted_text: str = Field(description="Verbatim clause text for the quoted-from-ToS block")


class ExplainTosScoresResponse(BaseModel):
    """Response from POST /api/explain_tos_scores."""

    service_name: str
    harm_filter: list[str] = Field(
        description="Canonical harm classes applied (Harmful, Neutral, Fair)",
    )
    llm_model: str = Field(description="Ollama model used for explanations")
    point_count: int
    points: list[ExplainedPoint]


@app.post("/api/explain_tos_scores", response_model=ExplainTosScoresResponse)
def explain_tos_scores(request: ExplainTosScoresRequest) -> ExplainTosScoresResponse:
    """
    Generate plain-language point titles and descriptions for classified clauses.

    Call **after** POST /api/analyze_tos. Only clauses matching harm_filter are
    enriched and returned (default: Harmful / bad only).

    Requires Ollama running locally with the model from VITE_OLLAMA_MODEL (.env).

    Errors:
        400 — invalid harm_filter token
        502 — Ollama returned unparseable output for a clause
        503 — Ollama unreachable
    """
    service_name = request.service_name.strip()
    if not service_name:
        raise HTTPException(status_code=400, detail="service_name is required")

    clause_dicts = [c.model_dump() for c in request.clauses]

    try:
        points, model_name, normalized_filter = explain_clauses(
            clause_dicts,
            request.harm_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaExplainError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc}. Ensure Ollama is running and the model is pulled.",
        ) from exc
    except ClauseExplainError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return ExplainTosScoresResponse(
        service_name=service_name,
        harm_filter=normalized_filter,
        llm_model=model_name,
        point_count=len(points),
        points=points,
    )


# =============================================================================
# Participant interaction telemetry — user-study exploratory measure
# =============================================================================
# The researcher starts/ends a session from their own laptop; the server stamps
# participant/session onto every client event. The browser has no session UI.

class InteractionEvent(BaseModel):
    """One client-side interaction event. Participant/session ids are stamped server-side."""

    event_type: Literal[
        "client_hello", "click", "screen_view", "heartbeat", "change_card_expand",
        "change_card_collapse", "original_clause_open", "category_toggle",
        "preview_toggle", "ask_ai_open", "ask_ai_message", "analysis_result",
        "profile_selection",
    ]
    screen: str | None = Field(default=None, description="Named screen/step the event occurred on")
    client_id: str = Field(description="Pseudonymous per-browser id, generated client-side")
    client_seq: int = Field(description="Client-side event counter")
    client_ts: str = Field(description="Client-side ISO timestamp")
    app_commit: str | None = Field(default=None, description="Web app build commit")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific fields, e.g. target descriptor or duration_ms")


class SessionStartBody(BaseModel):
    participant_id: str = Field(pattern=r"^[0-9]{1,6}$")


class SessionEndBody(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


MAX_EVENT_DATA_BYTES = 64 * 1024
ACTIVE_SESSION_PATH = INTERACTION_LOG_PATH.parent / "active_session.json"
_CLIENT_FIELDS = ("client_id", "client_seq", "client_ts", "app_commit")
# Active session (or None), guarded by _interaction_log_lock together with the log file.
_session: dict | None = None


def _require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    # Read at request time; fail closed when unset. Needed because the ngrok
    # tunnel makes public requests arrive from localhost too.
    expected = os.getenv("LAWGIC_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Session routes disabled: LAWGIC_ADMIN_TOKEN is not set")
    # Compare as bytes: str compare_digest raises TypeError on non-ASCII input.
    if not x_admin_token or not secrets.compare_digest(x_admin_token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


def _save_session() -> None:
    # Only what a restart needs. next_seq is rewritten on every event so a
    # resumed session continues its seq numbering instead of restarting at 0.
    keys = ("participant_id", "session_id", "started_at", "next_seq")
    tmp = ACTIVE_SESSION_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: _session[k] for k in keys}), encoding="utf-8")
    os.replace(tmp, ACTIVE_SESSION_PATH)


def _load_session() -> None:
    global _session
    try:
        saved = json.loads(ACTIVE_SESSION_PATH.read_text(encoding="utf-8"))
        _session = {
            **saved,
            "started_dt": datetime.fromisoformat(saved["started_at"]),
            "last_event": None, "last_heartbeat": None, "current_screen": None,
        }
        logger.info("Resumed telemetry session %s (participant %s)", saved["session_id"], saved["participant_id"])
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Ignoring unreadable %s: %s", ACTIVE_SESSION_PATH, exc)


def _append_line(event_type: str, screen: str | None, data: dict, client: dict | None = None) -> dict:
    """Stamp and write one JSONL line for the active session. Caller holds the lock; raises OSError."""
    now = datetime.now(timezone.utc)
    record = {
        "server_ts": now.isoformat(),
        "participant_id": _session["participant_id"],
        "session_id": _session["session_id"],
        "seq": _session["next_seq"],
        "t_ms": int((now - _session["started_dt"]).total_seconds() * 1000),
        "event_type": event_type,
        "screen": screen,
        **(client or dict.fromkeys(_CLIENT_FIELDS)),
        "data": data,
    }
    INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INTERACTION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _session["next_seq"] += 1
    return record


_load_session()


@app.post("/api/session/start", dependencies=[Depends(_require_admin)])
def session_start(body: SessionStartBody) -> dict:
    """Researcher starts a session for one participant. One active session at a time."""
    global _session
    with _interaction_log_lock:
        if _session:
            raise HTTPException(
                status_code=409,
                detail=f"Session already active for participant {_session['participant_id']}",
            )
        now = datetime.now(timezone.utc)
        _session = {
            "participant_id": body.participant_id,
            "session_id": str(uuid.uuid4()),
            "started_at": now.isoformat(),
            "started_dt": now,
            "next_seq": 0,
            "last_event": None, "last_heartbeat": None, "current_screen": None,
        }
        # Encoder checkpoint only. The generative model and RISK_SCORER live in the
        # web app's process (OLLAMA_MODEL, RISK_SCORER), so its client_hello reports those.
        meta = {"model_dir": MODEL_DIR.name}
        try:
            _save_session()
            _append_line("session_start", None, meta)
            _save_session()
        except OSError as exc:
            logger.error("Failed to start telemetry session: %s", exc)
            _session = None
            ACTIVE_SESSION_PATH.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Could not write session log") from exc
        return {
            "participant_id": _session["participant_id"],
            "session_id": _session["session_id"],
            "started_at": _session["started_at"],
            **meta,
        }


@app.post("/api/session/end", dependencies=[Depends(_require_admin)])
def session_end(body: SessionEndBody) -> dict:
    global _session
    with _interaction_log_lock:
        if not _session:
            raise HTTPException(status_code=404, detail="No active session")
        duration_ms = int((datetime.now(timezone.utc) - _session["started_dt"]).total_seconds() * 1000)
        try:
            _append_line(
                "session_end", None,
                {"end_reason": "researcher_ended", "duration_ms": duration_ms, "note": body.note},
            )
        except OSError as exc:
            # Keep the session active so the researcher can retry.
            logger.error("Failed to write session_end: %s", exc)
            raise HTTPException(status_code=500, detail="Could not write session log") from exc
        summary = {
            "participant_id": _session["participant_id"],
            "session_id": _session["session_id"],
            "duration_ms": duration_ms,
            "event_count": _session["next_seq"] - 2,  # excludes session_start and session_end
        }
        _session = None
        ACTIVE_SESSION_PATH.unlink(missing_ok=True)
        return summary


@app.get("/api/session/status", dependencies=[Depends(_require_admin)])
def session_status() -> dict:
    with _interaction_log_lock:
        if not _session:
            return {"active": False}
        now = datetime.now(timezone.utc)

        def age(ts: datetime | None) -> float | None:
            return None if ts is None else round((now - ts).total_seconds(), 1)

        return {
            "active": True,
            "participant_id": _session["participant_id"],
            "session_id": _session["session_id"],
            "elapsed_ms": int((now - _session["started_dt"]).total_seconds() * 1000),
            "event_count": _session["next_seq"] - 1,  # client events; seq 0 is session_start
            "last_event_age_s": age(_session["last_event"]),
            "last_heartbeat_age_s": age(_session["last_heartbeat"]),
            "current_screen": _session["current_screen"],
        }


@app.post("/api/track_event")
def track_event(event: InteractionEvent) -> dict:
    """
    Append one participant interaction event to the local telemetry log,
    stamped with the active session. Dropped when no session is active.

    Best-effort only: never raises on write failure, since a logging hiccup
    must not interrupt a live study session. Never proxied anywhere else.
    """
    if len(json.dumps(event.data, ensure_ascii=False).encode("utf-8")) > MAX_EVENT_DATA_BYTES:
        raise HTTPException(status_code=413, detail="Event data exceeds 64 KB")
    with _interaction_log_lock:
        if not _session:
            return {"logged": False, "reason": "no_active_session"}
        try:
            _append_line(
                event.event_type, event.screen, event.data,
                event.model_dump(include=set(_CLIENT_FIELDS)),
            )
            _save_session()
        except OSError as exc:
            logger.warning("Failed to write interaction event: %s", exc)
            return {"logged": False, "reason": "write_failed"}
        now = datetime.now(timezone.utc)
        _session["last_event"] = now
        if event.event_type == "heartbeat":
            _session["last_heartbeat"] = now
        if event.screen is not None:
            _session["current_screen"] = event.screen
    return {"logged": True}
