#!/usr/bin/env python3
"""Calibrate Hey Embry from held-out WAVs using OpenWakeWord native streaming scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import wave
from typing import Any

import numpy as np
from openwakeword.model import Model

from _common import atomic_write_json, read_json, read_jsonl, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        compression = handle.getcomptype()
        frame_count = handle.getnframes()
        payload = handle.readframes(frame_count)
    if channels != 1:
        raise RuntimeError(f"calibration_wav_not_mono:{path}:{channels}")
    if sample_rate != 16000:
        raise RuntimeError(f"calibration_wav_rate_invalid:{path}:{sample_rate}")
    if sample_width != 2 or compression != "NONE":
        raise RuntimeError(
            f"calibration_wav_not_pcm16:{path}:{sample_width}:{compression}"
        )
    samples = np.frombuffer(payload, dtype="<i2").copy()
    if samples.size != frame_count or samples.size == 0:
        raise RuntimeError(f"calibration_wav_empty_or_truncated:{path}")
    return samples, sample_rate


def predict_chunks(
    model: Model,
    model_name: str,
    samples: np.ndarray,
    chunk_samples: int,
) -> list[float]:
    scores: list[float] = []
    for offset in range(0, samples.size, chunk_samples):
        chunk = samples[offset:offset + chunk_samples]
        prediction = model.predict(chunk)
        score = float(prediction.get(model_name, 0.0))
        if not np.isfinite(score):
            raise RuntimeError("nonfinite_streaming_score")
        scores.append(score)
    return scores


def score_wav(
    model: Model,
    model_name: str,
    path: Path,
    chunk_samples: int,
    warmup_seconds: float,
    trailing_seconds: float,
) -> dict[str, Any]:
    samples, sample_rate = read_pcm16_mono(path)
    model.reset()

    warmup = np.zeros(
        int(round(sample_rate * warmup_seconds)),
        dtype=np.int16,
    )
    predict_chunks(model, model_name, warmup, chunk_samples)

    trailing = np.zeros(
        int(round(sample_rate * trailing_seconds)),
        dtype=np.int16,
    )
    scored_samples = np.concatenate((samples, trailing))
    scores = predict_chunks(model, model_name, scored_samples, chunk_samples)
    if not scores or not np.isfinite(scores).all():
        raise RuntimeError(f"invalid_streaming_scores:{path}")
    peak_index = int(np.argmax(scores))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "maximum_score": scores[peak_index],
        "peak_frame_index": peak_index,
        "peak_elapsed_ms_from_source_start": round(
            (peak_index + 1) * chunk_samples * 1000.0 / sample_rate,
            3,
        ),
        "source_duration_ms": round(
            samples.size * 1000.0 / sample_rate,
            3,
        ),
        "warmup_seconds": warmup_seconds,
        "trailing_seconds": trailing_seconds,
        "frame_scores": scores,
    }


def threshold_metrics(positive: list[float], negative: list[float], threshold: float) -> dict[str, Any]:
    tp = sum(score >= threshold for score in positive)
    fp = sum(score >= threshold for score in negative)
    return {
        "threshold": round(threshold, 6), "true_positive": tp, "false_negative": len(positive) - tp,
        "false_positive": fp, "true_negative": len(negative) - fp,
        "recall": tp / len(positive), "false_accept_rate": fp / len(negative),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--positive-split", default="positive_calibration")
    parser.add_argument("--negative-split", default="negative_calibration")
    parser.add_argument("--chunk-samples", type=int, default=512)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--trailing-seconds", type=float, default=1.0)
    parser.add_argument("--min-recall", type=float, required=True)
    parser.add_argument("--max-false-accept-rate", type=float, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--wake-word-buffer-duration", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.warmup_seconds <= 0 or args.trailing_seconds < 0:
        raise RuntimeError("calibration_padding_invalid")
    if args.chunk_samples <= 0:
        raise RuntimeError("calibration_chunk_samples_invalid")

    training = read_json(args.training_manifest)
    if training.get("status") != "PASS" or sha256_file(args.model) != (training.get("output") or {}).get("model_sha256"):
        raise RuntimeError("training_model_hash_mismatch")
    dataset = read_json(args.dataset_manifest)
    if dataset.get("status") != "PASS" or (dataset.get("semantic_qc") or {}).get("status") != "PASS":
        raise RuntimeError("calibration_dataset_semantic_qc_not_passed")
    records = read_jsonl(Path(dataset["records_path"]))
    positive_records = [
        item for item in records
        if item.get("status") == "accepted"
        and item.get("split") == args.positive_split
    ]
    negative_records = [
        item for item in records
        if item.get("status") == "accepted"
        and item.get("split") == args.negative_split
    ]
    if any(
        item.get("synthetic") is True
        and (item.get("semantic_qc") or {}).get("status") != "PASS"
        for item in positive_records
    ):
        raise RuntimeError("calibration_positive_semantic_qc_missing")
    positives = [Path(item["normalized_wav_path"]) for item in positive_records]
    negatives = [Path(item["normalized_wav_path"]) for item in negative_records]
    if not positives or not negatives:
        raise RuntimeError("calibration_split_empty")

    quarantine_path = Path(str(dataset.get("quarantine_records_path") or ""))
    if quarantine_path.is_file():
        exposed_hashes = {
            str(
                (
                    (item.get("original_record") or {}).get("normalized_wav")
                    or {}
                ).get("sha256")
                or ""
            )
            for item in read_jsonl(quarantine_path)
            if "calibration_audio_exposed_by_prior_model_selection"
            in (item.get("reason_codes") or [])
        }
        current_hashes = {sha256_file(path) for path in positives + negatives}
        overlap = sorted((exposed_hashes - {""}) & current_hashes)
        if overlap:
            raise RuntimeError(f"calibration_reuses_exposed_audio:{overlap[:5]}")
    training_hashes = {str(item.get("sha256")) for item in training.get("used_records") or []}
    calibration_hashes = [sha256_file(path) for path in positives + negatives]
    if len(calibration_hashes) != len(set(calibration_hashes)):
        raise RuntimeError("calibration_duplicate_audio_hash")
    overlap = sorted(set(calibration_hashes) & training_hashes)
    if overlap:
        raise RuntimeError(f"calibration_training_leakage:{overlap[:5]}")

    model_name = args.model.stem
    runtime = Model(wakeword_models=[str(args.model)], inference_framework="onnx")
    model_input_frames = int(runtime.model_inputs[model_name])
    minimum_warmup_seconds = model_input_frames * 0.08
    if args.warmup_seconds < minimum_warmup_seconds:
        raise RuntimeError(
            "calibration_warmup_shorter_than_model_context:"
            f"{args.warmup_seconds}:{minimum_warmup_seconds}"
        )
    positive_records = [
        score_wav(
            runtime, model_name, path, args.chunk_samples,
            args.warmup_seconds, args.trailing_seconds,
        )
        for path in positives
    ]
    negative_records = [
        score_wav(
            runtime, model_name, path, args.chunk_samples,
            args.warmup_seconds, args.trailing_seconds,
        )
        for path in negatives
    ]
    positive_scores = [item["maximum_score"] for item in positive_records]
    negative_scores = [item["maximum_score"] for item in negative_records]
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step / 2.0, args.threshold_step)
    table = [threshold_metrics(positive_scores, negative_scores, float(value)) for value in thresholds]
    eligible = [item for item in table if item["recall"] >= args.min_recall and item["false_accept_rate"] <= args.max_false_accept_rate]
    selected = max(eligible, key=lambda item: item["threshold"]) if eligible else None
    receipt = {
        "schema": "embry.openwakeword_calibration_receipt.v1", "status": "PASS" if selected else "FAIL",
        "live": True, "mocked": False, "model_id": "hey_embry_v1", "phrase": "Hey Embry",
        "model_path": str(args.model.resolve()), "model_sha256": sha256_file(args.model),
        "training_manifest_path": str(args.training_manifest.resolve()), "training_manifest_sha256": sha256_file(args.training_manifest),
        "dataset_manifest_path": str(args.dataset_manifest.resolve()), "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "calibration_source": "held_out_synthetic_horus", "physical_human_wake_proven": False,
        "positive_split": args.positive_split, "negative_split": args.negative_split,
        "positive_count": len(positives), "negative_count": len(negatives),
        "chunk_samples": args.chunk_samples,
        "model_input_frames": model_input_frames,
        "minimum_model_context_seconds": minimum_warmup_seconds,
        "warmup_seconds": args.warmup_seconds,
        "trailing_seconds": args.trailing_seconds,
        "score_statistic": "maximum_native_streaming_frame_score_after_discarded_zero_warmup",
        "requirements": {"minimum_recall": args.min_recall, "maximum_false_accept_rate": args.max_false_accept_rate},
        "selected_threshold": selected["threshold"] if selected else None,
        "selected_sensitivity": selected["threshold"] if selected else None,
        "wake_word_buffer_duration": args.wake_word_buffer_duration,
        "selected_metrics": selected, "threshold_table": table,
        "positive_scores": positive_records, "negative_scores": negative_records,
        "generated_at": utc_now(), "failed_gates": [] if selected else ["no_measured_operating_point"],
    }
    atomic_write_json(args.output, receipt)
    print(args.output)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
