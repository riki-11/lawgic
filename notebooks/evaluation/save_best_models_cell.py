# Save the best (highest validation metric) dual-head model per encoder to
# saved_models/, in the same file layout as lawgic_classifier_legal-bert_v3
# (model_state_dict.pt + encoder/tokenizer + head weights + taxonomy + metadata).
# Reads metrics.json directly from disk, so it works even for encoders whose
# runs finished before the rest of the matrix does. Writes to a NEW directory
# per encoder (suffix "_phase2") — never touches lawgic_classifier_legal-bert_v3.
# Skips an encoder whose target directory already exists (idempotent / resumable),
# and skips an encoder with no completed dual-head run yet.

import shutil
from datetime import datetime, timezone

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

SAVE_TARGETS = {
    "nlpaueb/legal-bert-base-uncased": "legal-bert",
    "bert-base-uncased": "bert",
    "xlnet-base-cased": "xlnet",
    "roberta-base": "roberta",
}
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"


def completed_dual_runs(encoder_name: str) -> list[dict]:
    records = []
    for metrics_path in sorted(tm.RUNS_DIR.glob("*/metrics.json")):
        record = json.loads(metrics_path.read_text())
        if record["encoder_name"] == encoder_name and record["heads"] == "dual":
            records.append(record)
    return records


def best_checkpoint_dir(run_id: str) -> Path:
    checkpoints = sorted(
        (tm.RUNS_DIR / run_id / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint saved for {run_id}")
    # save_total_limit=1 + load_best_model_at_end=True: the one surviving
    # checkpoint is the best validation checkpoint, not just the last epoch.
    return checkpoints[-1]


def save_best_model(encoder_name: str, short_name: str) -> None:
    candidates = completed_dual_runs(encoder_name)
    if not candidates:
        print(f"skip {short_name}: no completed dual-head runs yet")
        return

    best = max(candidates, key=lambda r: r["best_val_metric"])
    run_id = best["run_id"]

    output_dir = SAVED_MODELS_DIR / f"lawgic_classifier_{short_name}_phase2"
    if output_dir.exists():
        print(f"skip {short_name}: {output_dir} already exists, not overwriting")
        return

    checkpoint_dir = best_checkpoint_dir(run_id)

    model = tm.LawgicDualHeadModel(encoder_name)
    weights_file = checkpoint_dir / "model.safetensors"
    state_dict = (
        load_safetensors(str(weights_file))
        if weights_file.exists()
        else torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    )
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    output_dir.mkdir(parents=True)

    # Full state dict + encoder/tokenizer + heads separately, mirroring v3's layout.
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.topic_head.state_dict(), output_dir / "topic_head_weights.pt")
    torch.save(model.harm_head.state_dict(), output_dir / "harm_head_weights.pt")

    topic_ids, name_by_topic, _ = core.load_taxonomy()
    compact_taxonomy = [
        {"classifier_id": i, "topic_id": tid, "name": name_by_topic[tid]}
        for i, tid in enumerate(topic_ids)
    ]
    (output_dir / "lawgic_topics_44.json").write_text(json.dumps(compact_taxonomy, indent=2))
    shutil.copy2(core.TAXONOMY_PATH, output_dir / "lawgic_topics_original_45.json")

    (output_dir / "test_metrics.json").write_text(json.dumps(best, indent=2, default=str))

    metadata = {
        "model_name": encoder_name,
        "architecture": "dual_head",
        "num_topics": core.NUM_LAWGIC_TOPICS,
        "num_harm_classes": core.NUM_HARM_CLASSES,
        "max_length": core.MAX_LENGTH,
        "decision_threshold": core.DECISION_THRESHOLD,
        "seed": best["seed"],
        "source_run_id": run_id,
        "best_val_metric": best["best_val_metric"],
        "seeds_considered": sorted(r["seed"] for r in candidates),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Best-of-3-seeds model from the Phase 2 multi-encoder matrix "
            "(notebooks/evaluation/02_multiseed_encoder_runs.ipynb); does not "
            "replace lawgic_classifier_legal-bert_v3."
        ),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[{short_name}] saved best seed {best['seed']} (run {run_id}) -> {output_dir}")


for encoder_name, short_name in SAVE_TARGETS.items():
    save_best_model(encoder_name, short_name)
