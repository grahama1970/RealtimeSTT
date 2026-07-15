#!/usr/bin/env python3
"""Package exactly the runtime model and four required provenance files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import onnxruntime as ort
import scipy

from _common import atomic_write_json, read_json, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/wake"))
    parser.add_argument("--receipt", type=Path, help="Optional package receipt outside models/wake")
    args = parser.parse_args()

    training = read_json(args.training_manifest)
    calibration = read_json(args.calibration_receipt)
    model_sha = sha256_file(args.model)
    if training.get("status") != "PASS" or (training.get("output") or {}).get("model_sha256") != model_sha:
        raise RuntimeError("training_manifest_model_mismatch")
    if calibration.get("status") != "PASS" or calibration.get("model_sha256") != model_sha or calibration.get("selected_threshold") is None:
        raise RuntimeError("calibration_not_passed_for_model")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_names = {
        "hey_embry_v1.onnx", "hey_embry_v1.onnx.sha256", "hey_embry_v1.model.json",
        "hey_embry_v1.training-manifest.json", "hey_embry_v1.calibration-receipt.json",
    }
    unexpected = {item.name for item in output.iterdir() if item.is_file()} - expected_names
    if unexpected:
        raise RuntimeError(f"package_directory_contains_unexpected_files:{sorted(unexpected)}")

    final_model = output / "hey_embry_v1.onnx"
    atomic_copy(args.model, final_model)
    if sha256_file(final_model) != model_sha:
        raise RuntimeError("packaged_model_hash_mismatch")
    session = ort.InferenceSession(str(final_model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = [1 if not isinstance(value, int) or value <= 0 else value for value in input_meta.shape]
    values = session.run(None, {input_meta.name: np.zeros(shape, dtype=np.float32)})
    if not values or not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("packaged_model_onnxruntime_failed")

    sha_path = output / "hey_embry_v1.onnx.sha256"
    sha_path.write_text(f"{model_sha}  hey_embry_v1.onnx\n", encoding="utf-8")
    training_copy = output / "hey_embry_v1.training-manifest.json"
    calibration_copy = output / "hey_embry_v1.calibration-receipt.json"
    atomic_copy(args.training_manifest, training_copy)
    atomic_copy(args.calibration_receipt, calibration_copy)
    metadata = {
        "schema": "embry.openwakeword_model.v1", "model_id": "hey_embry_v1", "phrase": "Hey Embry",
        "framework": "openwakeword_onnx", "model_path": "models/wake/hey_embry_v1.onnx",
        "model_sha256": model_sha, "model_bytes": final_model.stat().st_size,
        "python_version": (training.get("environment") or {}).get("python") or platform.python_version(),
        "scipy_version": (training.get("environment") or {}).get("scipy") or scipy.__version__,
        "openwakeword_commit": training.get("openwakeword_commit"),
        "piper_sample_generator_commit": training.get("piper_sample_generator_commit"),
        "positive_train_count": (training.get("dataset_counts") or {}).get("positive_train"),
        "positive_validation_count": (training.get("dataset_counts") or {}).get("positive_validation"),
        "negative_train_count": (training.get("dataset_counts") or {}).get("negative_train"),
        "negative_validation_count": (training.get("dataset_counts") or {}).get("negative_validation"),
        "hard_negative_count": ((training.get("dataset_counts") or {}).get("negative_train") or 0) + ((training.get("dataset_counts") or {}).get("negative_validation") or 0),
        "dataset_manifest_sha256": training.get("dataset_manifest_sha256"),
        "training_command": training.get("training_command"),
        "calibration": {"positive_count": calibration.get("positive_count"), "negative_count": calibration.get("negative_count"),
                        "selected_threshold": calibration.get("selected_threshold"), "selected_metrics": calibration.get("selected_metrics")},
        "selected_sensitivity": calibration.get("selected_sensitivity"),
        "wake_word_buffer_duration": calibration.get("wake_word_buffer_duration"),
        "qualification_state": "SYNTHETIC_CALIBRATED_PHYSICAL_CANARY_PENDING",
        "synthetic_horus_build_proves": ["real_horus_wav_generation", "official_openwakeword_training", "onnxruntime_load", "measured_synthetic_operating_point"],
        "does_not_prove": ["physical_human_wake", "physical_jabra_callback", "release_false_accept_rate"],
        "packaged_at": utc_now(),
    }
    metadata_path = output / "hey_embry_v1.model.json"
    atomic_write_json(metadata_path, metadata)

    actual_names = {item.name for item in output.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(f"package_file_set_invalid:{sorted(actual_names)}")
    package_receipt = {
        "schema": "embry.openwakeword_package_receipt.v1", "status": "PASS", "model_sha256": model_sha,
        "output_dir": str(output), "files": {name: sha256_file(output / name) for name in sorted(expected_names)},
        "onnxruntime_load": True, "physical_human_wake_proven": False,
    }
    if args.receipt:
        atomic_write_json(args.receipt, package_receipt)
    print(json.dumps(package_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
