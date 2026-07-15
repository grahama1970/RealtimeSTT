#!/usr/bin/env python3
"""Generate resumable, receipt-bound Hey Embry and hard-negative WAVs from local Horus Orpheus."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import requests
from scipy.io import wavfile

from _common import append_jsonl, atomic_write_json, canonical_bytes, normalize_wav, read_json, read_jsonl, sha256_bytes, sha256_file, wav_metadata
from piper_positive_source import PiperPositiveSource


POSITIVE_TARGET = "Hey Embry"
ACCEPTED_TARGET_PAIRS = {
    ("hey", "embry"),
    ("hey", "embree"),
    ("hey", "embrie"),
}
PIPER_EXACT_HARD_NEGATIVE_TRANSCRIPTS = {
    "embry",
    "embryo",
    "emory",
    "emery",
    "henry",
    "hey emory",
    "hey emery",
    "hey emily",
    "hey empty",
}
POSITIVE_SYNTHESIS_TARGET = "Hey Embree"
POSITIVE_PRONUNCIATION_STRATEGY = "orthographic_alias_embree_v1"
POSITIVE_GENERATION_STRATEGY = "carrier_embree_word_timestamp_crop_v2"
POSITIVE_CARRIER_TEMPLATES = (
    "Okay. {target}. Ready.",
    "Hello. {target}. Listen.",
    "Testing. {target}. Begin.",
)
POSITIVE_RETRY_PROFILES = (
    {"temperature": 0.35, "top_p": 0.55, "repetition_penalty": 1.10},
    {"temperature": 0.45, "top_p": 0.65, "repetition_penalty": 1.08},
    {"temperature": 0.55, "top_p": 0.75, "repetition_penalty": 1.06},
    {"temperature": 0.65, "top_p": 0.85, "repetition_penalty": 1.04},
)


class CandidateRejected(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def exact_target_transcript(value: str) -> bool:
    tokens = normalized_tokens(value)
    return len(tokens) == 2 and tuple(tokens) in ACCEPTED_TARGET_PAIRS


def piper_positive_admission(
    *,
    transcript: str,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized = " ".join(normalized_tokens(transcript))
    phoneme = source_metadata.get("phoneme_provenance")
    if not isinstance(phoneme, dict):
        return {"status": "FAIL", "reason": "piper_phoneme_provenance_missing"}
    if phoneme.get("status") != "PASS":
        return {"status": "FAIL", "reason": "piper_phoneme_provenance_invalid"}
    if not str(phoneme.get("phoneme_text") or "").strip():
        return {"status": "FAIL", "reason": "piper_phoneme_text_missing"}
    if not str(phoneme.get("phoneme_ids_sha256") or "").strip():
        return {"status": "FAIL", "reason": "piper_phoneme_hash_missing"}
    if not normalized:
        return {"status": "FAIL", "reason": "unprompted_whisper_empty"}
    if normalized in PIPER_EXACT_HARD_NEGATIVE_TRANSCRIPTS:
        return {
            "status": "FAIL",
            "reason": "unprompted_whisper_exact_hard_negative:" + normalized,
        }
    exact_target = exact_target_transcript(transcript)
    return {
        "status": "PASS",
        "admission_policy": "pinned_piper_phoneme_provenance_v1",
        "admission_authority": "piper_model_and_espeak_phoneme_input",
        "whisper_role": "unprompted_gross_failure_screen_only",
        "whisper_exact_target": exact_target,
        "whisper_observation_class": (
            "EXACT_TARGET" if exact_target else "AMBIGUOUS_NON_AUTHORITATIVE"
        ),
        "normalized_transcript": normalized,
    }


def require_local_url(value: str, label: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{label}_scheme_invalid:{value}")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"{label}_must_be_local:{value}")


def whisper_transcribe(
    path: Path,
    *,
    whisper_url: str,
    whisper_model: str,
    api_key: str,
    timeout: float,
    word_timestamps: bool,
) -> dict[str, Any]:
    data: list[tuple[str, str]] = [
        ("model", whisper_model),
        ("language", "en"),
        ("temperature", "0"),
        ("response_format", "verbose_json"),
    ]
    if word_timestamps:
        data.append(("timestamp_granularities[]", "word"))
    with path.open("rb") as handle:
        response = requests.post(
            whisper_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (path.name, handle, "audio/wav")},
            data=data,
            timeout=timeout,
        )
    if not response.ok:
        raise RuntimeError(f"whisper_http_error:{response.status_code}:{response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("whisper_response_not_object")
    transcript = payload.get("text") or payload.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise RuntimeError("whisper_transcript_missing")
    return payload


def whisper_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        raw_words = []
        for segment in payload.get("segments") or []:
            if isinstance(segment, dict) and isinstance(segment.get("words"), list):
                raw_words.extend(segment["words"])
    words: list[dict[str, Any]] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        token = normalized_tokens(str(raw.get("word") or raw.get("text") or ""))
        start = raw.get("start")
        end = raw.get("end")
        if len(token) != 1 or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start = float(start)
        end = float(end)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            words.append({"token": token[0], "start": start, "end": end, "raw": raw})
    if not words:
        raise CandidateRejected("whisper_word_timestamps_missing")
    return words


def target_word_span(words: list[dict[str, Any]]) -> tuple[int, int]:
    for index in range(len(words) - 1):
        if (words[index]["token"], words[index + 1]["token"]) in ACCEPTED_TARGET_PAIRS:
            return index, index + 1
    raise CandidateRejected("complete_target_pair_missing")


def extract_target_segment(
    carrier_path: Path,
    destination: Path,
    *,
    words: list[dict[str, Any]],
    first_index: int,
    second_index: int,
    pre_roll_ms: float,
    post_roll_ms: float,
) -> dict[str, Any]:
    sample_rate, data = wavfile.read(carrier_path)
    audio = np.asarray(data)
    if sample_rate != 16000:
        raise CandidateRejected(f"carrier_rate_invalid:{sample_rate}")
    if audio.ndim != 1 or audio.dtype != np.int16 or audio.size == 0:
        raise CandidateRejected(f"carrier_pcm_invalid:{audio.shape}:{audio.dtype}")
    first = words[first_index]
    second = words[second_index]
    duration = audio.size / sample_rate
    start = max(0.0, first["start"] - pre_roll_ms / 1000.0)
    end = min(duration, second["end"] + post_roll_ms / 1000.0)
    if first_index > 0:
        previous = words[first_index - 1]
        start = max(start, (previous["end"] + first["start"]) / 2.0)
    if second_index + 1 < len(words):
        following = words[second_index + 1]
        end = min(end, (second["end"] + following["start"]) / 2.0)
    if start > first["start"] + 0.020 or end < second["end"] - 0.020:
        raise CandidateRejected("carrier_boundary_overlaps_target")
    start_sample = max(0, int(math.floor(start * sample_rate)))
    end_sample = min(audio.size, int(math.ceil(end * sample_rate)))
    segment = audio[start_sample:end_sample]
    segment_duration = segment.size / sample_rate
    if not 0.30 <= segment_duration <= 2.00:
        raise CandidateRejected(f"extracted_segment_duration_invalid:{segment_duration:.6f}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, sample_rate, segment)
    return {
        "source_start_sec": round(start, 6),
        "source_end_sec": round(end, 6),
        "source_start_sample": start_sample,
        "source_end_sample": end_sample,
        "duration_sec": round(segment_duration, 6),
        "pre_roll_ms": pre_roll_ms,
        "post_roll_ms": post_roll_ms,
    }


def generation_strategy(*, pre_roll_ms: float, post_roll_ms: float) -> dict[str, Any]:
    value = {
        "id": POSITIVE_GENERATION_STRATEGY,
        "target": POSITIVE_TARGET,
        "planned_wake_phrase": POSITIVE_TARGET,
        "synthesis_pronunciation": POSITIVE_SYNTHESIS_TARGET,
        "pronunciation_strategy": POSITIVE_PRONUNCIATION_STRATEGY,
        "accepted_target_pairs": sorted(" ".join(pair) for pair in ACCEPTED_TARGET_PAIRS),
        "carrier_templates": list(POSITIVE_CARRIER_TEMPLATES),
        "retry_profiles": list(POSITIVE_RETRY_PROFILES),
        "pre_roll_ms": pre_roll_ms,
        "post_roll_ms": post_roll_ms,
        "carrier_asr_requires_word_timestamps": True,
        "segment_asr_requires_exact_two_token_target": True,
    }
    return {**value, "sha256": sha256_bytes(canonical_bytes(value))}


def positive_candidate_request(
    *, item: dict[str, Any], speaker: str, candidate_ordinal: int, min_duration_sec: float,
) -> tuple[str, str, dict[str, Any]]:
    template = POSITIVE_CARRIER_TEMPLATES[candidate_ordinal % len(POSITIVE_CARRIER_TEMPLATES)]
    profile_index = (candidate_ordinal // len(POSITIVE_CARRIER_TEMPLATES)) % len(POSITIVE_RETRY_PROFILES)
    request_body = {
        "speaker": speaker,
        "prompt": template.format(target=POSITIVE_SYNTHESIS_TARGET),
        **POSITIVE_RETRY_PROFILES[profile_index],
        "min_duration_sec": min_duration_sec,
    }
    seed = {
        "schema": "embry.wake_positive_candidate_seed.v1",
        "record_id": item["record_id"],
        "strategy": POSITIVE_GENERATION_STRATEGY,
        "candidate_ordinal": candidate_ordinal,
        "planned_wake_phrase": POSITIVE_TARGET,
        "synthesis_pronunciation": POSITIVE_SYNTHESIS_TARGET,
        "pronunciation_strategy": POSITIVE_PRONUNCIATION_STRATEGY,
        "carrier_template": template,
        "request": request_body,
    }
    return sha256_bytes(canonical_bytes(seed))[:24], template, request_body


def piper_generation_strategy(source: PiperPositiveSource) -> dict[str, Any]:
    value = source.strategy()
    return {**value, "sha256": sha256_bytes(canonical_bytes(value))}


def http_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 180.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers={"content-type": "application/json"}, method="POST" if body is not None else "GET")
    with urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("service_response_not_object")
    return value


def download_local_wav(base_url: str, location: str, destination: Path, *, timeout: float) -> str:
    source_url = urljoin(base_url.rstrip("/") + "/", location)
    parsed = urlparse(source_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(f"nonlocal_audio_url_forbidden:{source_url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url, timeout=timeout) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return source_url


def iter_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = plan.get("parameter_profiles") or [{}]
    records: list[dict[str, Any]] = []
    for split, entries in (plan.get("splits") or {}).items():
        if split.startswith("positive_"):
            label = "positive"
        elif split.startswith("negative_"):
            label = "hard_negative"
        else:
            raise ValueError(f"unsupported_split:{split}")
        if not isinstance(entries, list):
            raise ValueError(f"split_entries_not_list:{split}")
        for entry_number, entry in enumerate(entries):
            prompt = str(entry.get("prompt") or "").strip()
            synthesis_prompt = str(entry.get("synthesis_prompt") or prompt).strip()
            count = int(entry.get("count") or 0)
            if not prompt or not synthesis_prompt or count <= 0:
                raise ValueError(f"invalid_plan_entry:{split}:{entry_number}")
            if label != "positive" and synthesis_prompt != prompt:
                raise ValueError(f"negative_synthesis_prompt_override_forbidden:{split}:{entry_number}")
            for index in range(count):
                profile = profiles[index % len(profiles)]
                record_seed = {
                    "model_id": plan["model_id"], "split": split, "label": label,
                    "prompt": prompt, "entry": entry_number, "index": index, "profile": profile,
                }
                record_id = sha256_bytes(canonical_bytes(record_seed))[:24]
                records.append({
                    **record_seed,
                    "synthesis_prompt": synthesis_prompt,
                    "record_id": record_id,
                })
    return records


def import_records(path: Path | None, output_dir: Path, manifest_path: Path, completed: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    for source in read_jsonl(path):
        record_id = str(source.get("record_id") or sha256_bytes(canonical_bytes(source))[:24])
        if record_id in completed:
            continue
        split = str(source.get("split") or "")
        label = str(source.get("label") or "")
        if split not in {"positive_train", "positive_validation", "positive_calibration", "negative_train", "negative_validation", "negative_calibration"}:
            raise ValueError(f"import_split_invalid:{split}")
        expected_label = "positive" if split.startswith("positive_") else "hard_negative"
        if label != expected_label:
            raise ValueError(f"import_label_mismatch:{record_id}")
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        normalized = output_dir / "normalized" / split / f"{record_id}.wav"
        metadata = normalize_wav(source_path, normalized)
        record = {
            "schema": "embry.wake_dataset_record.v1", "status": "accepted", "record_id": record_id,
            "source_class": str(source.get("source_class") or "physical_human"), "synthetic": False,
            "split": split, "label": label, "prompt": source.get("prompt"),
            "source_path": str(source_path), "source_sha256": sha256_file(source_path),
            "normalized_wav_path": str(normalized), "normalized_wav": metadata, "accepted_at": utc_now(),
        }
        append_jsonl(manifest_path, record)
        completed[record_id] = record


def generate_positive_record(
    *,
    item: dict[str, Any],
    plan: dict[str, Any],
    output: Path,
    attempts_path: Path,
    prior_attempts: list[dict[str, Any]],
    service_url: str,
    whisper_url: str,
    whisper_model: str,
    whisper_api_key: str,
    timeout: float,
    whisper_timeout: float,
    min_duration_sec: float,
    max_candidates: int,
    pre_roll_ms: float,
    post_roll_ms: float,
    strategy: dict[str, Any],
) -> dict[str, Any] | None:
    record_id = str(item["record_id"])
    prior_ordinals = [
        int(row["candidate_ordinal"])
        for row in prior_attempts
        if row.get("record_id") == record_id
        and row.get("generation_strategy_id") == strategy["id"]
        and isinstance(row.get("candidate_ordinal"), int)
    ]
    start_ordinal = max(prior_ordinals, default=-1) + 1
    quarantine_path = output / "quarantine" / "positive-candidates.jsonl"

    for candidate_ordinal in range(start_ordinal, max_candidates):
        candidate_id, carrier_template, request_body = positive_candidate_request(
            item=item,
            speaker=str(plan.get("speaker") or "horus"),
            candidate_ordinal=candidate_ordinal,
            min_duration_sec=min_duration_sec,
        )
        candidate_dir = output / "candidates" / item["split"] / record_id / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        synthesis_path = candidate_dir / "synthesis.json"
        raw_carrier_path = candidate_dir / "carrier-raw.wav"
        normalized_carrier_path = candidate_dir / "carrier-16k.wav"
        carrier_asr_path = candidate_dir / "carrier-asr.json"
        extracted_path = candidate_dir / "hey-embry-extracted.wav"
        segment_asr_path = candidate_dir / "segment-asr.json"
        receipt_path = candidate_dir / "candidate-receipt.json"
        attempt = {
            "schema": "embry.wake_positive_candidate_attempt.v1",
            "record_id": record_id,
            "candidate_id": candidate_id,
            "candidate_ordinal": candidate_ordinal,
            "generation_strategy_id": strategy["id"],
            "generation_strategy_sha256": strategy["sha256"],
            "planned_prompt": item["prompt"],
            "planned_wake_phrase": POSITIVE_TARGET,
            "synthesis_pronunciation": POSITIVE_SYNTHESIS_TARGET,
            "pronunciation_strategy": POSITIVE_PRONUNCIATION_STRATEGY,
            "carrier_template": carrier_template,
            "request": request_body,
            "started_at": utc_now(),
        }
        candidate_receipt: dict[str, Any] = {
            **attempt,
            "authority": "local_whisper_dataset_audit_only",
            "production_wake_authority": False,
            "artifacts": {},
        }
        try:
            response = http_json(service_url.rstrip("/") + "/v1/synthesize", body=request_body, timeout=timeout)
            atomic_write_json(synthesis_path, response)
            candidate_receipt["artifacts"]["synthesis_receipt"] = {
                "path": str(synthesis_path), "sha256": sha256_file(synthesis_path),
            }
            location = str(response.get("download_url") or "")
            if not location:
                raise RuntimeError("synthesis_download_url_missing")
            source_url = download_local_wav(service_url, location, raw_carrier_path, timeout=timeout)
            raw_carrier = wav_metadata(raw_carrier_path)
            normalized_carrier = normalize_wav(raw_carrier_path, normalized_carrier_path)
            candidate_receipt["artifacts"]["carrier_raw"] = {"path": str(raw_carrier_path), **raw_carrier}
            candidate_receipt["artifacts"]["carrier_normalized"] = {
                "path": str(normalized_carrier_path), **normalized_carrier,
            }

            carrier_asr = whisper_transcribe(
                normalized_carrier_path,
                whisper_url=whisper_url,
                whisper_model=whisper_model,
                api_key=whisper_api_key,
                timeout=whisper_timeout,
                word_timestamps=True,
            )
            atomic_write_json(carrier_asr_path, carrier_asr)
            carrier_transcript = str(carrier_asr.get("text") or carrier_asr.get("transcript") or "").strip()
            candidate_receipt["artifacts"]["carrier_asr"] = {
                "path": str(carrier_asr_path),
                "sha256": sha256_file(carrier_asr_path),
                "transcript": carrier_transcript,
            }
            words = whisper_words(carrier_asr)
            first_index, second_index = target_word_span(words)
            extraction = extract_target_segment(
                normalized_carrier_path,
                extracted_path,
                words=words,
                first_index=first_index,
                second_index=second_index,
                pre_roll_ms=pre_roll_ms,
                post_roll_ms=post_roll_ms,
            )
            extracted_metadata = wav_metadata(extracted_path)
            candidate_receipt["artifacts"]["extracted_segment"] = {
                "path": str(extracted_path), **extracted_metadata, "extraction": extraction,
            }
            segment_asr = whisper_transcribe(
                extracted_path,
                whisper_url=whisper_url,
                whisper_model=whisper_model,
                api_key=whisper_api_key,
                timeout=whisper_timeout,
                word_timestamps=False,
            )
            atomic_write_json(segment_asr_path, segment_asr)
            segment_transcript = str(segment_asr.get("text") or segment_asr.get("transcript") or "").strip()
            candidate_receipt["artifacts"]["segment_asr"] = {
                "path": str(segment_asr_path),
                "sha256": sha256_file(segment_asr_path),
                "transcript": segment_transcript,
                "normalized_transcript": " ".join(normalized_tokens(segment_transcript)),
            }
            if not exact_target_transcript(segment_transcript):
                raise CandidateRejected(
                    "extracted_segment_not_exact_target:" + " ".join(normalized_tokens(segment_transcript))
                )

            normalized_path = output / "normalized" / item["split"] / f"{record_id}.wav"
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted_path, normalized_path)
            normalized_metadata = wav_metadata(normalized_path)
            candidate_receipt.update({"status": "accepted", "completed_at": utc_now(), "source_url": source_url})
            atomic_write_json(receipt_path, candidate_receipt)
            record = {
                "schema": "embry.wake_dataset_record.v1",
                "status": "accepted",
                "record_id": record_id,
                "model_id": plan["model_id"],
                "source_class": "horus_orpheus_local",
                "synthetic": True,
                "speaker": request_body["speaker"],
                "split": item["split"],
                "label": item["label"],
                "prompt": item["prompt"],
                "planned_prompt": item["prompt"],
                "target_phrase": POSITIVE_TARGET,
                "planned_wake_phrase": POSITIVE_TARGET,
                "synthesis_pronunciation": POSITIVE_SYNTHESIS_TARGET,
                "pronunciation_strategy": POSITIVE_PRONUNCIATION_STRATEGY,
                "synthesis_prompt": request_body["prompt"],
                "request": request_body,
                "candidate_id": candidate_id,
                "candidate_ordinal": candidate_ordinal,
                "generation_strategy": strategy,
                "service_url": service_url,
                "source_url": source_url,
                "service_request_id": (response.get("receipt") or {}).get("request_id") or response.get("request_id"),
                "candidate_receipt_path": str(receipt_path),
                "candidate_receipt_sha256": sha256_file(receipt_path),
                "raw_wav_path": str(raw_carrier_path),
                "raw_wav": raw_carrier,
                "carrier_normalized_wav_path": str(normalized_carrier_path),
                "carrier_normalized_wav": normalized_carrier,
                "normalized_wav_path": str(normalized_path),
                "normalized_wav": normalized_metadata,
                "semantic_qc": {
                    "schema": "embry.wake_dataset_semantic_qc.v1",
                    "status": "PASS",
                    "authority": "local_whisper_dataset_audit_only",
                    "production_wake_authority": False,
                    "accepted_transcript": segment_transcript,
                    "accepted_normalized_transcript": " ".join(normalized_tokens(segment_transcript)),
                    "carrier_asr_receipt_path": str(carrier_asr_path),
                    "carrier_asr_receipt_sha256": sha256_file(carrier_asr_path),
                    "segment_asr_receipt_path": str(segment_asr_path),
                    "segment_asr_receipt_sha256": sha256_file(segment_asr_path),
                },
                "accepted_at": utc_now(),
            }
            attempt.update({
                "status": "accepted",
                "completed_at": utc_now(),
                "candidate_receipt_path": str(receipt_path),
                "candidate_receipt_sha256": sha256_file(receipt_path),
                "record_sha256": sha256_bytes(canonical_bytes(record)),
            })
            append_jsonl(attempts_path, attempt)
            return record
        except CandidateRejected as exc:
            status = "semantic_rejected"
            reason_key = "reason"
            reason = str(exc)
        except Exception as exc:
            status = "failed"
            reason_key = "error"
            reason = f"{type(exc).__name__}:{exc}"

        candidate_receipt.update({"status": status, "completed_at": utc_now(), reason_key: reason})
        atomic_write_json(receipt_path, candidate_receipt)
        attempt.update({
            "status": status,
            "completed_at": utc_now(),
            reason_key: reason,
            "candidate_receipt_path": str(receipt_path),
            "candidate_receipt_sha256": sha256_file(receipt_path),
        })
        append_jsonl(attempts_path, attempt)
        append_jsonl(quarantine_path, {
            **candidate_receipt,
            "candidate_receipt_path": str(receipt_path),
            "candidate_receipt_sha256": sha256_file(receipt_path),
        })
        time.sleep(0.5)
    return None


def generate_piper_positive_record(
    *,
    item: dict[str, Any],
    plan: dict[str, Any],
    output: Path,
    attempts_path: Path,
    prior_attempts: list[dict[str, Any]],
    source: PiperPositiveSource,
    whisper_url: str,
    whisper_model: str,
    whisper_api_key: str,
    whisper_timeout: float,
    max_candidates: int,
    existing_hashes: set[str],
    strategy: dict[str, Any],
) -> dict[str, Any] | None:
    record_id = str(item["record_id"])
    attempted_ordinals = {
        int(row["candidate_ordinal"])
        for row in prior_attempts
        if row.get("record_id") == record_id
        and row.get("generation_strategy_id") == strategy["id"]
        and isinstance(row.get("candidate_ordinal"), int)
    }
    quarantine_path = output / "quarantine" / "piper-positive-candidates.jsonl"
    for candidate_ordinal in range(max_candidates):
        if candidate_ordinal in attempted_ordinals:
            continue
        candidate_dir = output / "candidates" / "piper" / item["split"] / record_id / f"candidate-{candidate_ordinal:02d}"
        wav_path = candidate_dir / "candidate.wav"
        asr_path = candidate_dir / "whisper.json"
        receipt_path = candidate_dir / "candidate-receipt.json"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        attempt: dict[str, Any] = {
            "record_id": record_id,
            "candidate_ordinal": candidate_ordinal,
            "generation_strategy_id": strategy["id"],
            "generation_strategy_sha256": strategy["sha256"],
            "started_at": utc_now(),
        }
        receipt: dict[str, Any] = {
            "schema": "embry.piper_wake_candidate.v1",
            "status": "STARTED",
            **attempt,
            "planned_wake_phrase": POSITIVE_TARGET,
            "dataset_qc_authority": "local_whisper_only",
            "production_wake_authority": False,
        }
        try:
            source_metadata = source.synthesize(
                record_id=record_id,
                candidate_ordinal=candidate_ordinal,
                output_path=wav_path,
            )
            attempt["candidate_id"] = source_metadata["candidate_id"]
            receipt["candidate_id"] = source_metadata["candidate_id"]
            receipt["source"] = source_metadata
            if source_metadata["wav"]["sha256"] in existing_hashes:
                raise CandidateRejected("duplicate_active_audio_hash")
            asr = whisper_transcribe(
                wav_path,
                whisper_url=whisper_url,
                whisper_model=whisper_model,
                api_key=whisper_api_key,
                timeout=whisper_timeout,
                word_timestamps=False,
            )
            atomic_write_json(asr_path, asr)
            transcript = str(asr.get("text") or asr.get("transcript") or "").strip()
            admission = piper_positive_admission(
                transcript=transcript,
                source_metadata=source_metadata,
            )
            receipt["dataset_qc"] = {
                "schema": "embry.piper_positive_dataset_qc.v1",
                "authority": "piper_phoneme_provenance_with_unprompted_whisper_screen",
                "production_wake_authority": False,
                "asr_model": whisper_model,
                "asr_receipt_path": str(asr_path),
                "asr_receipt_sha256": sha256_file(asr_path),
                "whisper_observation": transcript,
                **admission,
            }
            if admission["status"] != "PASS":
                raise CandidateRejected(str(admission["reason"]))
            normalized_path = output / "normalized" / item["split"] / f"{record_id}.wav"
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, normalized_path)
            normalized_metadata = wav_metadata(normalized_path)
            receipt.update({"status": "ACCEPTED", "completed_at": utc_now()})
            atomic_write_json(receipt_path, receipt)
            record = {
                "schema": "embry.wake_dataset_record.v1",
                "status": "accepted",
                "record_id": record_id,
                "model_id": plan["model_id"],
                "source_class": "piper_libritts_synthetic",
                "synthetic": True,
                "split": item["split"],
                "label": item["label"],
                "prompt": item["prompt"],
                "planned_wake_phrase": POSITIVE_TARGET,
                "synthesis_pronunciation": source_metadata["synthesis_text"],
                "pronunciation_strategy": source_metadata["pronunciation_strategy"],
                "generation_strategy": strategy,
                "piper": source_metadata,
                "candidate_receipt_path": str(receipt_path),
                "candidate_receipt_sha256": sha256_file(receipt_path),
                "normalized_wav_path": str(normalized_path),
                "normalized_wav": normalized_metadata,
                "semantic_qc": {
                    "schema": "embry.wake_dataset_semantic_qc.v1",
                    "status": "PASS",
                    **receipt["dataset_qc"],
                },
                "accepted_at": utc_now(),
            }
            attempt.update({
                "status": "accepted",
                "completed_at": utc_now(),
                "candidate_receipt_path": str(receipt_path),
                "candidate_receipt_sha256": sha256_file(receipt_path),
                "record_sha256": sha256_bytes(canonical_bytes(record)),
            })
            append_jsonl(attempts_path, attempt)
            return record
        except CandidateRejected as exc:
            receipt.update({"status": "SEMANTIC_REJECTED", "reason": str(exc), "completed_at": utc_now()})
        except Exception as exc:
            receipt.update({"status": "FAILED", "error": f"{type(exc).__name__}:{exc}", "completed_at": utc_now()})
        atomic_write_json(receipt_path, receipt)
        attempt.update({
            "status": receipt["status"].lower(),
            "reason": receipt.get("reason"),
            "error": receipt.get("error"),
            "completed_at": receipt["completed_at"],
            "candidate_receipt_path": str(receipt_path),
            "candidate_receipt_sha256": sha256_file(receipt_path),
        })
        append_jsonl(attempts_path, attempt)
        append_jsonl(quarantine_path, {
            **receipt,
            "candidate_receipt_path": str(receipt_path),
            "candidate_receipt_sha256": sha256_file(receipt_path),
        })
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", default="http://127.0.0.1:8767")
    parser.add_argument("--whisper-url", default="http://127.0.0.1:9000/v1/audio/transcriptions")
    parser.add_argument("--whisper-model", default="whisper-1")
    parser.add_argument("--whisper-api-key-env", default="WHISPER_API_KEY")
    parser.add_argument("--whisper-timeout", type=float, default=120.0)
    parser.add_argument("--positive-source", choices=("horus-carrier", "piper"), default="horus-carrier")
    parser.add_argument("--piper-root", type=Path, default=Path("/opt/piper-sample-generator"))
    parser.add_argument(
        "--piper-model", type=Path,
        default=Path("/opt/piper-sample-generator/models/en-us-libritts-high.pt"),
    )
    parser.add_argument("--piper-model-sha256")
    parser.add_argument("--piper-speaker-count", type=int, default=256)
    parser.add_argument("--piper-seed-base", type=int, default=20260715)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--import-manifest", type=Path)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--positive-max-candidates", type=int, default=12)
    parser.add_argument("--positive-pilot-slots", type=int, default=20)
    parser.add_argument("--positive-min-pilot-acceptance", type=float, default=0.90)
    parser.add_argument("--positive-max-consecutive-exhausted", type=int, default=2)
    parser.add_argument("--segment-pre-roll-ms", type=float, default=60.0)
    parser.add_argument("--segment-post-roll-ms", type=float, default=100.0)
    parser.add_argument("--max-records", type=int, help="Bound this invocation without changing the immutable plan")
    parser.add_argument(
        "--min-duration-sec", type=float, default=0.25,
        help="Wake-dataset-only Orpheus minimum output duration",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_local_url(args.service_url, "orpheus_url")
    require_local_url(args.whisper_url, "whisper_url")
    if not 0.1 <= args.min_duration_sec <= 30.0:
        parser.error("--min-duration-sec must be between 0.1 and 30.0")
    if args.positive_max_candidates < 1:
        parser.error("--positive-max-candidates must be positive")
    if args.positive_pilot_slots < 0:
        parser.error("--positive-pilot-slots must not be negative")
    if not 0.0 <= args.positive_min_pilot_acceptance <= 1.0:
        parser.error("--positive-min-pilot-acceptance must be between 0 and 1")
    if args.positive_max_consecutive_exhausted < 1:
        parser.error("--positive-max-consecutive-exhausted must be positive")
    if args.positive_source == "piper" and not args.piper_model_sha256:
        parser.error("--piper-model-sha256 is required for Piper")

    plan = read_json(args.plan)
    if plan.get("schema") != "embry.horus_wake_dataset_plan.v1":
        raise ValueError("dataset_plan_schema_invalid")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "records.jsonl"
    attempts_path = output / "attempts.jsonl"
    quarantine_path = output / "quarantine" / "records.jsonl"
    existing = read_jsonl(manifest_path)
    attempt_history = read_jsonl(attempts_path)
    quarantine_history = read_jsonl(quarantine_path)
    if existing and not args.resume:
        raise RuntimeError("dataset_exists_use_resume")
    completed = {str(item["record_id"]): item for item in existing if item.get("status") == "accepted"}
    prior_attempt_counts = Counter(
        str(item.get("record_id") or "") for item in attempt_history
    )
    generation_revisions = Counter(
        str(item.get("record_id") or "") for item in quarantine_history
    )
    import_records(args.import_manifest, output, manifest_path, completed)

    health = http_json(args.service_url.rstrip("/") + "/health", timeout=args.timeout)
    if health.get("status") != "ok" or "horus" not in (health.get("speaker_checkpoints") or {"horus": None}):
        raise RuntimeError("horus_service_not_ready")

    planned = iter_plan(plan)
    pending = [record for record in planned if record["record_id"] not in completed]
    if args.max_records is not None:
        pending = pending[: args.max_records]
    positive_pending = [item for item in pending if item["label"] == "positive"]
    piper_source = None
    if positive_pending and args.positive_source == "piper":
        piper_source = PiperPositiveSource(
            root=args.piper_root,
            model_path=args.piper_model,
            expected_model_sha256=args.piper_model_sha256,
            speaker_count=args.piper_speaker_count,
            seed_base=args.piper_seed_base,
        )
    whisper_api_key = os.environ.get(args.whisper_api_key_env, "")
    if positive_pending and not whisper_api_key:
        raise RuntimeError(f"whisper_api_key_missing:{args.whisper_api_key_env}")
    strategy = (
        piper_generation_strategy(piper_source)
        if piper_source is not None
        else generation_strategy(
            pre_roll_ms=args.segment_pre_roll_ms,
            post_roll_ms=args.segment_post_roll_ms,
        )
    )
    existing_hashes = {
        str((record.get("normalized_wav") or {}).get("sha256") or "")
        for record in completed.values()
    }
    positive_slots_attempted = 0
    positive_slots_accepted = 0
    positive_slots_exhausted = 0
    consecutive_exhausted = 0
    for ordinal, item in enumerate(pending, start=1):
        record_id = item["record_id"]
        if item["label"] == "positive":
            positive_slots_attempted += 1
            record = (
                generate_piper_positive_record(
                    item=item,
                    plan=plan,
                    output=output,
                    attempts_path=attempts_path,
                    prior_attempts=attempt_history,
                    source=piper_source,
                    whisper_url=args.whisper_url,
                    whisper_model=args.whisper_model,
                    whisper_api_key=whisper_api_key,
                    whisper_timeout=args.whisper_timeout,
                    max_candidates=args.positive_max_candidates,
                    existing_hashes=existing_hashes,
                    strategy=strategy,
                )
                if piper_source is not None
                else generate_positive_record(
                    item=item,
                    plan=plan,
                    output=output,
                    attempts_path=attempts_path,
                    prior_attempts=attempt_history,
                    service_url=args.service_url,
                    whisper_url=args.whisper_url,
                    whisper_model=args.whisper_model,
                    whisper_api_key=whisper_api_key,
                    timeout=args.timeout,
                    whisper_timeout=args.whisper_timeout,
                    min_duration_sec=args.min_duration_sec,
                    max_candidates=args.positive_max_candidates,
                    pre_roll_ms=args.segment_pre_roll_ms,
                    post_roll_ms=args.segment_post_roll_ms,
                    strategy=strategy,
                )
            )
            if record is None:
                positive_slots_exhausted += 1
                consecutive_exhausted += 1
                if consecutive_exhausted >= args.positive_max_consecutive_exhausted:
                    raise RuntimeError(
                        f"positive_generation_stop:consecutive_slots_exhausted:{consecutive_exhausted}"
                    )
            else:
                append_jsonl(manifest_path, record)
                completed[record_id] = record
                attempt_history = read_jsonl(attempts_path)
                existing_hashes.add(str(record["normalized_wav"]["sha256"]))
                positive_slots_accepted += 1
                consecutive_exhausted = 0
                print(f"accepted {ordinal}/{len(pending)} {item['split']} {record_id}", flush=True)
            if args.positive_pilot_slots > 0 and positive_slots_attempted == args.positive_pilot_slots:
                pilot_rate = positive_slots_accepted / positive_slots_attempted
                if pilot_rate < args.positive_min_pilot_acceptance:
                    raise RuntimeError(
                        "positive_generation_stop:"
                        f"pilot_acceptance_rate:{pilot_rate:.6f}:"
                        f"required:{args.positive_min_pilot_acceptance:.6f}"
                    )
            continue
        generation_revision = generation_revisions[record_id] + 1
        request_body = {
            "speaker": plan.get("speaker", "horus"),
            "prompt": item["synthesis_prompt"],
            **item["profile"],
            "min_duration_sec": args.min_duration_sec,
        }
        last_error: str | None = None
        for local_attempt in range(1, args.max_attempts + 1):
            attempt = prior_attempt_counts[record_id] + local_attempt
            attempt_record = {
                "record_id": record_id,
                "attempt": attempt,
                "generation_revision": generation_revision,
                "planned_prompt": item["prompt"],
                "synthesis_prompt": item["synthesis_prompt"],
                "started_at": utc_now(),
                "request": request_body,
            }
            try:
                response = http_json(args.service_url.rstrip("/") + "/v1/synthesize", body=request_body, timeout=args.timeout)
                receipt_dir = output / "synthesis_receipts"
                receipt_dir.mkdir(parents=True, exist_ok=True)
                receipt_path = receipt_dir / f"{record_id}.json"
                atomic_write_json(receipt_path, response)
                location = str(response.get("download_url") or "")
                if not location:
                    raise RuntimeError("synthesis_download_url_missing")
                raw_path = output / "raw" / item["split"] / f"{record_id}.wav"
                source_url = download_local_wav(args.service_url, location, raw_path, timeout=args.timeout)
                raw_meta = wav_metadata(raw_path)
                normalized_path = output / "normalized" / item["split"] / f"{record_id}.wav"
                normalized_meta = normalize_wav(raw_path, normalized_path)
                record = {
                    "schema": "embry.wake_dataset_record.v1", "status": "accepted", "record_id": record_id,
                    "model_id": plan["model_id"], "source_class": "horus_orpheus_local", "synthetic": True,
                    "generation_revision": generation_revision,
                    "speaker": request_body["speaker"], "split": item["split"], "label": item["label"],
                    "prompt": item["prompt"],
                    "planned_prompt": item["prompt"],
                    "synthesis_prompt": item["synthesis_prompt"],
                    "prompt_transform": (
                        "positive_prompt_controls_removed_v1"
                        if item["synthesis_prompt"] != item["prompt"]
                        else "identity"
                    ),
                    "semantic_qc": (
                        {"status": "PENDING", "authority": "dataset_audit_only"}
                        if item["label"] == "positive"
                        else None
                    ),
                    "request": request_body,
                    "service_url": args.service_url, "source_url": source_url,
                    "service_request_id": (response.get("receipt") or {}).get("request_id") or response.get("request_id"),
                    "synthesis_receipt_path": str(receipt_path), "synthesis_receipt_sha256": sha256_file(receipt_path),
                    "raw_wav_path": str(raw_path), "raw_wav": raw_meta,
                    "normalized_wav_path": str(normalized_path), "normalized_wav": normalized_meta,
                    "accepted_at": utc_now(),
                }
                append_jsonl(manifest_path, record)
                completed[record_id] = record
                attempt_record.update({"status": "accepted", "completed_at": utc_now(), "record_sha256": sha256_bytes(canonical_bytes(record))})
                append_jsonl(attempts_path, attempt_record)
                print(f"accepted {ordinal}/{len(pending)} {item['split']} {record_id}", flush=True)
                break
            except Exception as exc:  # preserve every failed live attempt
                last_error = f"{type(exc).__name__}:{exc}"
                attempt_record.update({"status": "failed", "completed_at": utc_now(), "error": last_error})
                append_jsonl(attempts_path, attempt_record)
                if local_attempt < args.max_attempts:
                    time.sleep(min(8.0, 2.0 ** local_attempt))
        else:
            raise RuntimeError(f"synthesis_failed:{record_id}:{last_error}")

    records = read_jsonl(manifest_path)
    expected_ids = {item["record_id"] for item in planned}
    accepted_ids = {item["record_id"] for item in records if item.get("status") == "accepted" and item["record_id"] in expected_ids}
    counts: dict[str, int] = {}
    for item in records:
        if item.get("status") == "accepted":
            counts[item["split"]] = counts.get(item["split"], 0) + 1
    accepted_records = [item for item in records if item.get("status") == "accepted"]
    accepted_hashes = [str((item.get("normalized_wav") or {}).get("sha256") or "") for item in accepted_records]
    hash_counts = Counter(digest for digest in accepted_hashes if digest)
    duplicate_hashes = sorted(digest for digest, count in hash_counts.items() if count > 1)
    if any(not digest for digest in accepted_hashes):
        raise RuntimeError("accepted_record_missing_normalized_hash")
    complete = expected_ids.issubset(accepted_ids)
    pending_semantic_qc = sorted(
        str(item["record_id"])
        for item in accepted_records
        if item.get("synthetic") is True
        and item.get("label") == "positive"
        and (item.get("semantic_qc") or {}).get("status") != "PASS"
    )
    failed_gates: list[str] = []
    if not complete:
        failed_gates.append("dataset_plan_incomplete")
    if duplicate_hashes:
        failed_gates.append("duplicate_normalized_audio_hash")
    if pending_semantic_qc:
        failed_gates.append("semantic_positive_qc_pending")
    required_counts = {
        "positive_train": 200,
        "positive_validation": 75,
        "positive_calibration": 40,
        "negative_train": 260,
        "negative_validation": 90,
        "negative_calibration": 40,
    }
    semantic_positive_count = sum(
        1
        for item in accepted_records
        if item.get("label") == "positive"
        and (item.get("semantic_qc") or {}).get("status") == "PASS"
    )
    if any(counts.get(split, 0) != expected for split, expected in required_counts.items()):
        failed_gates.append("dataset_split_counts_invalid")
    if semantic_positive_count != 315:
        failed_gates.append("semantic_positive_corpus_incomplete")
    summary = {
        "schema": "embry.horus_wake_dataset_manifest.v1",
        "status": "PASS" if not failed_gates else "PARTIAL",
        "live": True, "mocked": False, "model_id": plan["model_id"],
        "plan_path": str(args.plan.resolve()), "plan_sha256": sha256_file(args.plan),
        "records_path": str(manifest_path), "records_sha256": sha256_file(manifest_path),
        "attempts_path": str(attempts_path), "attempts_sha256": sha256_file(attempts_path) if attempts_path.exists() else None,
        "planned_count": len(planned), "accepted_plan_record_count": len(accepted_ids), "counts": counts,
        "unique_normalized_audio_hash_count": len(hash_counts),
        "duplicate_normalized_audio_hashes": duplicate_hashes,
        "required_counts": required_counts,
        "semantic_positive_count": semantic_positive_count,
        "semantic_positive_required_count": 315,
        "positive_generation": {
            "strategy": strategy,
            "slots_attempted_this_run": positive_slots_attempted,
            "slots_accepted_this_run": positive_slots_accepted,
            "slots_exhausted_this_run": positive_slots_exhausted,
            "pilot_slots": args.positive_pilot_slots,
            "pilot_minimum_acceptance": args.positive_min_pilot_acceptance,
            "max_candidates_per_slot": args.positive_max_candidates,
            "max_consecutive_exhausted": args.positive_max_consecutive_exhausted,
        },
        "dataset_asr_role": "semantic_quality_control_only",
        "production_wake_authority": "openwakeword_native_callback",
        "semantic_positive_qc_pending_count": len(pending_semantic_qc),
        "semantic_positive_qc_pending_record_ids": pending_semantic_qc,
        "semantic_qc": {
            "status": "PASS" if complete and not pending_semantic_qc else "PENDING",
            "authority": "local_whisper_dataset_audit_only",
            "production_wake_authority": False,
        },
        "quarantine_records_path": str(quarantine_path),
        "quarantine_records_sha256": sha256_file(quarantine_path) if quarantine_path.exists() else None,
        "physical_inputs_optional": True, "synthetic_build_physical_wake_proven": False,
        "generated_at": utc_now(), "failed_gates": failed_gates,
    }
    atomic_write_json(output / "dataset-manifest.json", summary)
    print(output / "dataset-manifest.json")
    return 0 if not failed_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
