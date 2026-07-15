#!/usr/bin/env python3
"""Quarantine synthetic wake positives that do not contain the complete phrase."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import requests

from _common import atomic_write_json, read_json, read_jsonl, sha256_bytes, sha256_file


ACCEPTED_TARGETS = {
    ("hey", "embry"),
    ("hey", "embree"),
    ("hey", "embrie"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def contains_complete_target(value: str) -> bool:
    tokens = normalized_tokens(value)
    return any(
        tuple(tokens[index:index + 2]) in ACCEPTED_TARGETS
        for index in range(len(tokens) - 1)
    )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def transcribe(
    path: Path,
    *,
    asr_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    api_key = os.environ.get("WHISPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("whisper_api_key_missing")
    with Path(path).open("rb") as handle:
        response = requests.post(
            asr_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(path).name, handle, "audio/wav")},
            data={
                "model": model,
                "language": "en",
                "response_format": "json",
                "temperature": "0",
            },
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("asr_response_not_object")
    text = (
        payload.get("text")
        or payload.get("transcript")
        or (
            (payload.get("result") or {}).get("text")
            if isinstance(payload.get("result"), dict)
            else None
        )
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("asr_transcript_missing")
    return {
        "transcript": text.strip(),
        "normalized_transcript": " ".join(normalized_tokens(text)),
        "complete_target_present": contains_complete_target(text),
        "response_sha256": sha256_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def preserve_artifacts(
    record: dict[str, Any],
    quarantine_dir: Path,
) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    artifact_paths = {
        "raw_wav": record.get("raw_wav_path"),
        "normalized_wav": record.get("normalized_wav_path"),
        "synthesis_receipt": record.get("synthesis_receipt_path"),
    }
    destination = quarantine_dir / "artifacts" / str(record["record_id"])
    destination.mkdir(parents=True, exist_ok=True)
    for label, value in artifact_paths.items():
        if not value:
            continue
        source = Path(str(value))
        if not source.is_file():
            continue
        suffix = source.suffix or ".bin"
        target = destination / f"{label}{suffix}"
        shutil.copy2(source, target)
        preserved[label] = {
            "source_path": str(source),
            "quarantine_path": str(target),
            "sha256": sha256_file(target),
        }
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--asr-url", required=True)
    parser.add_argument("--asr-model", default="whisper-1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh-calibration",
        action="store_true",
        help="Quarantine all calibration records exposed during model selection.",
    )
    args = parser.parse_args()

    dataset = read_json(args.dataset_manifest)
    if dataset.get("schema") != "embry.horus_wake_dataset_manifest.v1":
        raise RuntimeError("dataset_manifest_schema_invalid")
    records_path = Path(str(dataset["records_path"]))
    records = read_jsonl(records_path)
    source_manifest_sha = sha256_file(args.dataset_manifest)
    source_records_sha = sha256_file(records_path)

    quarantine_path = args.quarantine_dir / "records.jsonl"
    quarantine_rows = read_jsonl(quarantine_path)
    active_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    newly_quarantined: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") != "accepted":
            active_records.append(record)
            continue

        split = str(record.get("split") or "")
        label = str(record.get("label") or "")
        is_synthetic_positive = (
            record.get("synthetic") is True and label == "positive"
        )

        semantic_qc: dict[str, Any] | None = None
        if is_synthetic_positive:
            path = Path(str(record.get("normalized_wav_path") or ""))
            if not path.is_file():
                raise FileNotFoundError(path)
            result = transcribe(
                path,
                asr_url=args.asr_url,
                model=args.asr_model,
                timeout=args.timeout,
            )
            semantic_qc = {
                "schema": "embry.wake_dataset_semantic_qc.v1",
                "status": "PASS" if result["complete_target_present"] else "FAIL",
                "authority": "local_whisper_dataset_audit_only",
                "production_wake_authority": False,
                "asr_url": args.asr_url,
                "asr_model": args.asr_model,
                "normalized_wav_sha256": sha256_file(path),
                **result,
                "audited_at": utc_now(),
            }
            audit_rows.append({
                "record_id": record["record_id"],
                "split": split,
                "prompt": record.get("prompt"),
                "synthesis_prompt": record.get("synthesis_prompt") or record.get("prompt"),
                "semantic_qc": semantic_qc,
            })

        reasons: list[str] = []
        if semantic_qc is not None and semantic_qc["status"] != "PASS":
            reasons.append("complete_hey_embry_phrase_missing")
        if args.refresh_calibration and split in {
            "positive_calibration",
            "negative_calibration",
        }:
            reasons.append("calibration_audio_exposed_by_prior_model_selection")

        if reasons:
            quarantine = {
                "schema": "embry.wake_dataset_quarantine.v1",
                "record_id": record["record_id"],
                "split": split,
                "label": label,
                "reason_codes": reasons,
                "original_record": record,
                "semantic_qc": semantic_qc,
                "preserved_artifacts": preserve_artifacts(record, args.quarantine_dir),
                "quarantined_at": utc_now(),
            }
            quarantine_rows.append(quarantine)
            newly_quarantined.append(quarantine)
            continue

        if semantic_qc is not None:
            record = {**record, "semantic_qc": semantic_qc}
        active_records.append(record)

    atomic_write_jsonl(records_path, active_records)
    atomic_write_jsonl(quarantine_path, quarantine_rows)

    active_accepted = [
        item for item in active_records if item.get("status") == "accepted"
    ]
    counts = Counter(str(item.get("split") or "") for item in active_accepted)
    active_positive_failures = [
        item["record_id"]
        for item in active_accepted
        if item.get("synthetic") is True
        and item.get("label") == "positive"
        and (item.get("semantic_qc") or {}).get("status") != "PASS"
    ]

    receipt = {
        "schema": "embry.wake_dataset_semantic_audit_receipt.v1",
        "status": "PASS",
        "live": True,
        "mocked": False,
        "authority": "local_whisper_dataset_audit_only",
        "production_wake_authority": False,
        "source_dataset_manifest_sha256": source_manifest_sha,
        "source_records_sha256": source_records_sha,
        "result_records_sha256": sha256_file(records_path),
        "asr_url": args.asr_url,
        "asr_model": args.asr_model,
        "positive_records_audited": len(audit_rows),
        "positive_records_semantically_valid": sum(
            row["semantic_qc"]["status"] == "PASS" for row in audit_rows
        ),
        "newly_quarantined_count": len(newly_quarantined),
        "newly_quarantined_record_ids": [
            item["record_id"] for item in newly_quarantined
        ],
        "refresh_calibration": args.refresh_calibration,
        "active_record_count": len(active_accepted),
        "active_counts": dict(sorted(counts.items())),
        "active_positive_semantic_failure_count": len(active_positive_failures),
        "records": audit_rows,
        "quarantine_records_path": str(quarantine_path),
        "quarantine_records_sha256": sha256_file(quarantine_path),
        "generated_at": utc_now(),
    }
    atomic_write_json(args.output, receipt)

    complete = len(active_accepted) == int(dataset["planned_count"])
    semantic_pass = not active_positive_failures and not newly_quarantined
    dataset.update({
        "status": "PASS" if complete and semantic_pass else "PARTIAL",
        "records_sha256": sha256_file(records_path),
        "accepted_plan_record_count": len(active_accepted),
        "counts": dict(sorted(counts.items())),
        "semantic_qc": {
            "status": "PASS" if semantic_pass else "PENDING",
            "receipt_path": str(args.output.resolve()),
            "receipt_sha256": sha256_file(args.output),
            "authority": "local_whisper_dataset_audit_only",
            "production_wake_authority": False,
        },
        "quarantine_records_path": str(quarantine_path),
        "quarantine_records_sha256": sha256_file(quarantine_path),
        "failed_gates": (
            [] if complete and semantic_pass
            else ["semantic_qc_regeneration_pending"]
        ),
        "generated_at": utc_now(),
    })
    atomic_write_json(args.dataset_manifest, dataset)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
