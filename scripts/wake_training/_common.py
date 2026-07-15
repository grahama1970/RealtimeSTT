#!/usr/bin/env python3
"""Shared fail-closed utilities for the Hey Embry wake-model gate."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not Path(path).exists():
        return records
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_record_not_object:{path}:{line_number}")
        records.append(value)
    return records


def run_checked(command: list[str], *, cwd: Path | None = None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, stdout=stdout, stderr=stderr, check=False)
    receipt = {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    if result.returncode != 0:
        raise RuntimeError(f"command_failed:{result.returncode}:{' '.join(command)}")
    return receipt


def _decoded_wav(path: Path, sample_rate: int, data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    array = np.asarray(data)
    if sample_rate <= 0:
        raise ValueError(f"wav_sample_rate_invalid:{path}:{sample_rate}")
    if array.size == 0:
        raise ValueError(f"wav_empty:{path}")
    if array.ndim == 1:
        frames, channels = int(array.shape[0]), 1
    elif array.ndim == 2 and array.shape[1] > 0:
        frames, channels = int(array.shape[0]), int(array.shape[1])
    else:
        raise ValueError(f"wav_shape_unsupported:{path}:{array.shape}")
    if frames <= 0:
        raise ValueError(f"wav_empty:{path}")
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize in {4, 8}:
        compression = "IEEE_FLOAT"
        encoding = f"ieee_float{array.dtype.itemsize * 8}"
    elif np.issubdtype(array.dtype, np.signedinteger) and array.dtype.itemsize in {1, 2, 4, 8}:
        compression = "PCM"
        encoding = f"pcm_s{array.dtype.itemsize * 8}"
    elif np.issubdtype(array.dtype, np.unsignedinteger) and array.dtype.itemsize == 1:
        compression = "PCM"
        encoding = "pcm_u8"
    else:
        raise ValueError(f"wav_dtype_unsupported:{path}:{array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"wav_nonfinite:{path}")
    return array, {
        "channels": channels,
        "sample_rate_hz": int(sample_rate),
        "sample_width_bytes": int(array.dtype.itemsize),
        "frame_count": frames,
        "duration_ms": round(frames * 1000.0 / sample_rate, 3),
        "compression": compression,
        "encoding": encoding,
    }


def wav_metadata(path: Path) -> dict[str, Any]:
    path = Path(path)
    sample_rate, data = wavfile.read(path)
    _, metadata = _decoded_wav(path, int(sample_rate), data)
    return {
        **metadata,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_wav(source: Path, destination: Path, *, target_rate: int = 16000) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    rate, data = wavfile.read(source)
    array, _ = _decoded_wav(source, int(rate), data)
    if array.ndim == 2:
        array = array.astype(np.float64).mean(axis=1)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(array.dtype).max + 1) / 2.0
        signal = (array.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(array.dtype, np.signedinteger):
        maximum = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        signal = array.astype(np.float64) / float(maximum)
    else:
        signal = array.astype(np.float64)
    if rate != target_rate:
        divisor = math.gcd(int(rate), int(target_rate))
        signal = resample_poly(signal, target_rate // divisor, rate // divisor, window=("kaiser", 8.6))
    if signal.size == 0:
        raise ValueError(f"wav_empty:{source}")
    if not np.isfinite(signal).all():
        raise ValueError(f"wav_nonfinite:{source}")
    peak = float(np.max(np.abs(signal)))
    if peak <= 1e-6:
        raise ValueError(f"wav_silent:{source}")
    if peak > 1.0:
        signal = signal / peak
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype(np.int16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".wav", delete=False) as handle:
        temporary = Path(handle.name)
    wavfile.write(temporary, target_rate, pcm)
    os.replace(temporary, destination)
    metadata = wav_metadata(destination)
    if metadata["channels"] != 1 or metadata["sample_rate_hz"] != target_rate or metadata["sample_width_bytes"] != 2:
        raise ValueError(f"normalized_wav_contract_failed:{destination}")
    metadata["pcm_sha256"] = sha256_bytes(pcm.tobytes())
    return metadata


def copy_or_link(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"existing_file_hash_conflict:{destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def ensure_unique_hashes(paths: Iterable[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(f"duplicate_audio_hash:{seen[digest]}:{path}")
        seen[digest] = path
