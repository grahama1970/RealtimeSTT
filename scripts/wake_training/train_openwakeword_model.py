#!/usr/bin/env python3
"""Prepare data and invoke the pinned official OpenWakeWord augmentation/training implementation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import scipy
import yaml

from _common import atomic_write_json, canonical_bytes, copy_or_link, read_json, read_jsonl, run_checked, sha256_bytes, sha256_file

OPENWAKEWORD_COMMIT = "368c03716d1e92591906a84949bc477f3a834455"
PIPER_COMMIT = "f1988a4d54eddb23d99e86f0adfef6226a85acc7"
OPENWAKEWORD_COMPAT_PATCH = Path("/tmp/openwakeword-argparse-defaults.patch")
OPENWAKEWORD_COMPAT_PATCH_SHA256 = "13618f9cdf97f0da4cd47539409a8166452873ba5435b9b869da6976a49e08f9"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_environment(openwakeword_root: Path, piper_root: Path) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"python_minor_mismatch:{platform.python_version()}")
    if scipy.__version__ != "1.14.1":
        raise RuntimeError(f"scipy_version_mismatch:{scipy.__version__}")
    oww_commit = git_head(openwakeword_root)
    piper_commit = git_head(piper_root)
    if oww_commit != OPENWAKEWORD_COMMIT:
        raise RuntimeError(f"openwakeword_commit_mismatch:{oww_commit}")
    if piper_commit != PIPER_COMMIT:
        raise RuntimeError(f"piper_commit_mismatch:{piper_commit}")
    if not OPENWAKEWORD_COMPAT_PATCH.is_file() or sha256_file(OPENWAKEWORD_COMPAT_PATCH) != OPENWAKEWORD_COMPAT_PATCH_SHA256:
        raise RuntimeError("openwakeword_compat_patch_mismatch")
    if 'default="False"' in (openwakeword_root / "openwakeword" / "train.py").read_text(encoding="utf-8"):
        raise RuntimeError("openwakeword_argparse_patch_not_applied")
    return {
        "python": platform.python_version(), "scipy": scipy.__version__,
        "numpy": np.__version__, "openwakeword_commit": oww_commit, "piper_commit": piper_commit,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "compatibility_patches": [{
            "id": "openwakeword_argparse_store_true_defaults_v1",
            "sha256": OPENWAKEWORD_COMPAT_PATCH_SHA256,
        }],
    }




def require_audio_directory(path: Path, label: str) -> None:
    wavs = sorted(Path(path).glob("**/*.wav"))
    if not wavs:
        raise RuntimeError(f"{label}_wav_directory_empty:{path}")
    for wav in wavs[:32]:
        if wav.stat().st_size <= 44:
            raise RuntimeError(f"{label}_wav_invalid:{wav}")


def require_feature_array(path: Path, label: str) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    if array.ndim not in {2, 3} or array.shape[0] <= 0:
        raise RuntimeError(f"{label}_feature_shape_invalid:{array.shape}")
    sample = np.asarray(array[: min(128, array.shape[0])])
    if not np.isfinite(sample).all() or np.all(sample == 0):
        raise RuntimeError(f"{label}_features_invalid:{path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}

def materialize_dataset(dataset_manifest: Path, workspace_model_dir: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    summary = read_json(dataset_manifest)
    records_path = Path(summary["records_path"])
    records = read_jsonl(records_path)
    mapping = {
        "positive_train": "positive_train", "positive_validation": "positive_test",
        "negative_train": "negative_train", "negative_validation": "negative_test",
    }
    counts = {key: 0 for key in mapping}
    used: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for record in records:
        split = record.get("split")
        if split not in mapping or record.get("status") != "accepted":
            continue
        if (
            record.get("synthetic") is True
            and record.get("label") == "positive"
            and (record.get("semantic_qc") or {}).get("status") != "PASS"
        ):
            raise RuntimeError(
                f"training_positive_semantic_qc_missing:{record.get('record_id')}"
            )
        source = Path(record["normalized_wav_path"])
        digest = sha256_file(source)
        if digest in seen_hashes:
            raise RuntimeError(f"training_duplicate_audio_hash:{digest}")
        seen_hashes.add(digest)
        destination = workspace_model_dir / mapping[split] / f"{record['record_id']}.wav"
        copy_or_link(source, destination)
        counts[split] += 1
        used.append({"record_id": record["record_id"], "split": split, "sha256": digest, "path": str(destination)})
    for split in mapping:
        if counts[split] <= 0:
            raise RuntimeError(f"training_split_empty:{split}")
    return counts, used


def consolidate_onnx(source: Path, destination: Path) -> None:
    model = onnx.load(str(source), load_external_data=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.onnx")
    onnx.save_model(model, str(temporary), save_as_external_data=False)
    onnx.checker.check_model(str(temporary), full_check=True)
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = [1 if not isinstance(value, int) or value <= 0 else value for value in input_meta.shape]
    output = session.run(None, {input_meta.name: np.zeros(shape, dtype=np.float32)})
    if not output or not all(np.isfinite(item).all() for item in output):
        raise RuntimeError("onnx_runtime_output_nonfinite")
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openwakeword-root", type=Path, default=Path("/opt/openWakeWord"))
    parser.add_argument("--piper-root", type=Path, default=Path("/opt/piper-sample-generator"))
    parser.add_argument("--rir-dir", type=Path, action="append", required=True)
    parser.add_argument("--background-dir", type=Path, action="append", required=True)
    parser.add_argument("--negative-features", type=Path, required=True)
    parser.add_argument("--false-positive-validation-features", type=Path, required=True)
    parser.add_argument("--model-name", default="hey_embry_v1")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--augmentation-rounds", type=int, default=1)
    parser.add_argument("--augmentation-batch-size", type=int, default=16)
    parser.add_argument("--max-negative-weight", type=float, default=50.0)
    parser.add_argument("--target-fp-per-hour", type=float, default=0.5)
    parser.add_argument("--generic-negative-batch", type=int, default=1024)
    parser.add_argument("--hard-negative-batch", type=int, default=50)
    parser.add_argument("--positive-batch", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-augment", action="store_true")
    args = parser.parse_args()

    for path in [args.dataset_manifest, args.negative_features, args.false_positive_validation_features, *args.rir_dir, *args.background_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.augmentation_rounds != 1:
        raise RuntimeError("augmentation_rounds_must_equal_one_for_reproducible_feature_counts")
    for path in args.rir_dir:
        require_audio_directory(path, "rir")
    for path in args.background_dir:
        require_audio_directory(path, "background")
    generic_negative_metadata = require_feature_array(args.negative_features, "generic_negative")
    false_positive_metadata = require_feature_array(args.false_positive_validation_features, "false_positive_validation")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = require_environment(args.openwakeword_root.resolve(), args.piper_root.resolve())
    dataset_summary = read_json(args.dataset_manifest)
    if dataset_summary.get("status") != "PASS":
        raise RuntimeError("dataset_manifest_not_complete")
    if (dataset_summary.get("semantic_qc") or {}).get("status") != "PASS":
        raise RuntimeError("dataset_semantic_qc_not_passed")

    model_dir = output / args.model_name
    counts, used_records = materialize_dataset(args.dataset_manifest, model_dir)
    config = {
        "model_name": args.model_name, "target_phrase": ["hey embry"],
        "custom_negative_phrases": ["embry", "embryo", "emery", "henry", "hey emory", "hey henry", "hey emily", "hey empty", "hey memory"],
        "n_samples": max(counts["positive_train"], counts["negative_train"]),
        "n_samples_val": max(counts["positive_validation"], counts["negative_validation"]),
        "tts_batch_size": 1, "augmentation_batch_size": args.augmentation_batch_size,
        "piper_sample_generator_path": str(args.piper_root.resolve()), "output_dir": str(output),
        "rir_paths": [str(path.resolve()) for path in args.rir_dir],
        "background_paths": [str(path.resolve()) for path in args.background_dir],
        "background_paths_duplication_rate": [1 for _ in args.background_dir],
        "false_positive_validation_data_path": str(args.false_positive_validation_features.resolve()),
        "augmentation_rounds": args.augmentation_rounds,
        "feature_data_files": {"ACAV100M_sample": str(args.negative_features.resolve())},
        "batch_n_per_class": {"ACAV100M_sample": args.generic_negative_batch, "adversarial_negative": args.hard_negative_batch, "positive": args.positive_batch},
        "model_type": "dnn", "layer_size": args.layer_size, "steps": args.steps,
        "max_negative_weight": args.max_negative_weight, "target_false_positives_per_hour": args.target_fp_per_hour,
    }
    config_path = output / "resolved-training-config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    input_contract = {
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest), "config_sha256": sha256_file(config_path),
        "negative_features": generic_negative_metadata,
        "false_positive_validation_features": false_positive_metadata,
        "environment": environment, "counts": counts,
    }
    input_hash = sha256_bytes(canonical_bytes(input_contract))
    manifest_path = output / "training-manifest.json"
    if manifest_path.exists() and args.resume:
        prior = read_json(manifest_path)
        prior_model = Path((prior.get("output") or {}).get("model_path") or "")
        if prior.get("status") == "PASS" and prior.get("input_contract_sha256") == input_hash and prior_model.is_file() and sha256_file(prior_model) == (prior.get("output") or {}).get("model_sha256"):
            print(manifest_path)
            return 0

    logs = output / "logs"
    train_py = args.openwakeword_root.resolve() / "openwakeword" / "train.py"
    augment_cmd = [sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"]
    if args.force_augment or not all((model_dir / name).exists() for name in ["positive_features_train.npy", "positive_features_test.npy", "negative_features_train.npy", "negative_features_test.npy"]):
        augment_cmd.append("--overwrite")
        augment_receipt = run_checked(augment_cmd, cwd=args.openwakeword_root, stdout_path=logs / "augment.stdout.log", stderr_path=logs / "augment.stderr.log")
        atomic_write_json(output / "augment-receipt.json", {"schema": "embry.openwakeword_augment_receipt.v1", "status": "PASS", **augment_receipt})
    elif not args.resume:
        raise RuntimeError("feature_arrays_exist_use_resume_or_force_augment")

    feature_files = [model_dir / name for name in ["positive_features_train.npy", "positive_features_test.npy", "negative_features_train.npy", "negative_features_test.npy"]]
    feature_metadata: dict[str, Any] = {}
    for path in feature_files:
        if not path.is_file():
            raise RuntimeError(f"feature_file_missing:{path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 3 or array.shape[0] <= 0 or not np.isfinite(array[: min(32, array.shape[0])]).all():
            raise RuntimeError(f"feature_array_invalid:{path}")
        feature_metadata[path.name] = {"path": str(path), "sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}

    train_cmd = [sys.executable, str(train_py), "--training_config", str(config_path), "--train_model"]
    training_receipt = run_checked(train_cmd, cwd=args.openwakeword_root, stdout_path=logs / "train.stdout.log", stderr_path=logs / "train.stderr.log")
    source_model = output / f"{args.model_name}.onnx"
    if not source_model.is_file():
        raise RuntimeError("official_training_onnx_missing")
    standalone_model = output / "standalone" / "hey_embry_v1.onnx"
    consolidate_onnx(source_model, standalone_model)
    session = ort.InferenceSession(str(standalone_model), providers=["CPUExecutionProvider"])
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    manifest = {
        "schema": "embry.openwakeword_training_manifest.v1", "status": "PASS", "live": True, "mocked": False,
        "model_id": "hey_embry_v1", "phrase": "Hey Embry", "framework": "openwakeword_onnx",
        "input_contract_sha256": input_hash, "environment": environment,
        "openwakeword_commit": OPENWAKEWORD_COMMIT, "piper_sample_generator_commit": PIPER_COMMIT,
        "compatibility_patches": environment["compatibility_patches"],
        "dataset_manifest_path": str(args.dataset_manifest.resolve()), "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "dataset_counts": counts, "used_records": used_records, "training_config_path": str(config_path), "training_config_sha256": sha256_file(config_path),
        "training_command": train_cmd, "augmentation_command": augment_cmd, "feature_arrays": feature_metadata,
        "commands": {"training": training_receipt},
        "output": {"model_path": str(standalone_model), "model_sha256": sha256_file(standalone_model), "model_bytes": standalone_model.stat().st_size,
                   "input_name": input_meta.name, "input_shape": input_meta.shape, "output_name": output_meta.name, "output_shape": output_meta.shape},
        "qualification_state": "TRAINED_UNQUALIFIED", "synthetic_build_physical_wake_proven": False,
        "generated_at": utc_now(), "failed_gates": [],
    }
    atomic_write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
