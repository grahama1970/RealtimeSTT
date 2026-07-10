#!/usr/bin/env python3
"""Prove local PipeWire audio capture can feed RealtimeSTT.

This runner deliberately avoids UX Lab, Chatterbox, browser automation, typed
transcripts, and mocked STT events. It plays a known spoken WAV through a local
PipeWire sink, captures a PipeWire source as real PCM, feeds the captured PCM to
RealtimeSTT with use_microphone=False, and writes a replayable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import uuid
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RealtimeSTT import AudioToTextRecorder  # noqa: E402


EXPECTED_PHRASE = "embry ingress proof alpha seven"
PROOF_RUNG = "01_pipewire_or_pulse_audio_ingress_to_realtimestt"


@dataclass
class WavAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: int


def utc_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def run_command(command: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav(path: Path) -> WavAudio:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM, got sample width {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1).astype(np.int16)
    duration_ms = int(round((samples.size / float(sample_rate)) * 1000))
    return WavAudio(samples=samples, sample_rate=sample_rate, channels=channels, duration_ms=duration_ms)


def write_mono_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    samples = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def write_raw(path: Path, samples: np.ndarray) -> None:
    path.write_bytes(np.asarray(samples, dtype=np.int16).tobytes())


def audio_stats(path: Path) -> dict[str, Any]:
    audio = read_wav(path)
    samples = audio.samples.astype(np.float32) / 32768.0
    if samples.size == 0:
        rms = 0.0
        peak = 0.0
        non_silent_ratio = 0.0
    else:
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        frame = max(1, int(audio.sample_rate * 0.02))
        non_silent = 0
        total = 0
        for start in range(0, samples.size, frame):
            chunk = samples[start:start + frame]
            if chunk.size == 0:
                continue
            total += 1
            chunk_rms = float(np.sqrt(np.mean(np.square(chunk))))
            if chunk_rms > 0.0025:
                non_silent += 1
        non_silent_ratio = non_silent / max(1, total)
    rms_dbfs = -120.0 if rms <= 0 else 20.0 * math.log10(rms)
    peak_dbfs = -120.0 if peak <= 0 else 20.0 * math.log10(peak)
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
        "duration_ms": audio.duration_ms,
        "sample_rate": audio.sample_rate,
        "channels": 1,
        "format": "s16le",
        "rms_dbfs": round(rms_dbfs, 2),
        "peak_dbfs": round(peak_dbfs, 2),
        "non_silent_frame_ratio": round(non_silent_ratio, 4),
        "captured_audio_non_silent": bool(rms_dbfs > -50 and peak > 0.005 and non_silent_ratio > 0.05),
    }


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def transcript_matches(expected: str, actual: str, max_wer: float) -> tuple[bool, dict[str, Any]]:
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)
    expected_words = expected_norm.split()
    actual_words = actual_norm.split()
    distance = edit_distance(expected_words, actual_words)
    wer = distance / max(1, len(expected_words))
    return wer <= max_wer, {
        "expected": expected,
        "final": actual,
        "normalized_expected": expected_norm,
        "normalized_final": actual_norm,
        "edit_distance": distance,
        "wer": round(wer, 4),
        "matches_expected": wer <= max_wer,
    }


def repo_info(path: Path) -> dict[str, Any]:
    head = run_command(["git", "-C", str(path), "rev-parse", "--short", "HEAD"]).stdout.strip()
    status = run_command(["git", "-C", str(path), "status", "--short"]).stdout.splitlines()
    return {"path": str(path), "head": head, "dirty": bool(status), "dirty_count": len(status)}


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def event_id(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}.{digest}"


def build_event_publisher(
    *, service_url: str, session_id: str, turn_id: str, run_id: str
) -> tuple[Any, dict[str, Any]]:
    """Return a synchronous live event publisher and mutable delivery receipt."""
    delivery: dict[str, Any] = {
        "attempted": 0,
        "accepted": 0,
        "errors": [],
        "event_ids": [],
        "assigned_events": [],
    }
    previous_event_id: str | None = None
    client = httpx.Client(base_url=service_url.rstrip("/"), timeout=httpx.Timeout(5.0, connect=2.0))

    def publish(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal previous_event_id
        created_at = datetime.now(timezone.utc).isoformat()
        seed = f"{session_id}:{turn_id}:{event_type}:{run_id}:{uuid.uuid4().hex}"
        artifact_hashes = {
            key: value
            for key, value in payload.items()
            if key.endswith("sha256") and isinstance(value, str) and value
        }
        event = {
            "schema": "embry.voice_event.v1",
            "event_id": event_id(event_type, {"seed": seed}),
            "session_id": session_id,
            "turn_id": turn_id,
            "type": event_type,
            "created_at": created_at,
            "causation_id": previous_event_id or f"run.{run_id}",
            "correlation_id": session_id,
            "producer": "RealtimeSTT.pipewire_ingress",
            "mocked": False,
            "live": True,
            "artifact_hashes": artifact_hashes,
            "payload": {"run_id": run_id, **payload},
        }
        event["receipt_hash"] = hashlib.sha256(
            json.dumps(event, sort_keys=True).encode("utf-8")
        ).hexdigest()
        delivery["attempted"] += 1
        try:
            response = client.post("/v1/listener/events", json=event)
            response.raise_for_status()
            response_json = response.json()
            assigned_event_id = response_json.get("event_id")
            assigned_sequence = response_json.get("sequence")
            if assigned_event_id != event["event_id"]:
                raise RuntimeError("journal_response_event_id_mismatch")
            if not isinstance(assigned_sequence, int) or assigned_sequence < 1:
                raise RuntimeError("journal_response_sequence_missing")
            delivery["accepted"] += 1
            delivery["event_ids"].append(assigned_event_id)
            delivery["assigned_events"].append({
                "event_id": assigned_event_id,
                "sequence": assigned_sequence,
            })
            previous_event_id = assigned_event_id
            return {**event, "assigned_sequence": assigned_sequence}
        except Exception as exc:
            delivery["errors"].append({"event_id": event["event_id"], "error": str(exc)})
            raise RuntimeError(f"journal_event_publish_failed:{event['event_id']}:{exc}") from exc

    return publish, delivery


def generate_source_wav(path: Path, phrase: str, commands: list[str], source_wav: str | None = None) -> None:
    if source_wav:
        source = Path(source_wav).expanduser().resolve()
        if not source.exists():
            raise RuntimeError(f"--source-wav does not exist: {source}")
        shutil.copy2(source, path)
        commands.append(f"cp {source} {path}")
        return
    espeak = shutil.which("espeak-ng")
    if not espeak:
        raise RuntimeError("espeak-ng is required to generate a real spoken source WAV")
    command = [espeak, "-v", "en-us", "-s", "135", "-a", "200", "-w", str(path), phrase]
    commands.append(" ".join(command))
    result = run_command(command, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {result.stderr.strip()}")


def capture_pipewire_audio(
    run_dir: Path,
    source_wav: Path,
    captured_wav: Path,
    playback_target: str,
    capture_target: str,
    capture_seconds: float,
    commands: list[str],
) -> dict[str, Any]:
    record_out = run_dir / "pw-record.stdout.log"
    record_err = run_dir / "pw-record.stderr.log"
    play_out = run_dir / "pw-play.stdout.log"
    play_err = run_dir / "pw-play.stderr.log"

    record_command = [
        "pw-record",
        "--target",
        capture_target,
        "--rate",
        "16000",
        "--channels",
        "1",
        "--format",
        "s16",
        str(captured_wav),
    ]
    play_command = ["pw-play", "--target", playback_target, str(source_wav)]
    commands.append(" ".join(record_command))
    commands.append(" ".join(play_command))

    with record_out.open("wb") as stdout, record_err.open("wb") as stderr:
        capture_proc = subprocess.Popen(record_command, stdout=stdout, stderr=stderr)
    capture_started = time.monotonic()
    time.sleep(0.8)
    with play_out.open("wb") as stdout, play_err.open("wb") as stderr:
        play_proc = subprocess.Popen(play_command, stdout=stdout, stderr=stderr)

    play_proc.wait(timeout=max(10, int(capture_seconds) + 5))
    elapsed = time.monotonic() - capture_started
    if elapsed < capture_seconds:
        time.sleep(capture_seconds - elapsed)
    if capture_proc.poll() is None:
        capture_proc.send_signal(signal.SIGINT)
        try:
            capture_proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            capture_proc.kill()
            capture_proc.wait(timeout=4)

    return {
        "backend": "pipewire",
        "strategy": "existing-pipewire-sink-to-existing-pipewire-source",
        "created_sink": None,
        "created_monitor": None,
        "module_id": None,
        "playback_target": playback_target,
        "capture_target": capture_target,
        "playback_command": " ".join(play_command),
        "capture_command": " ".join(record_command),
        "playback_pid": play_proc.pid,
        "capture_pid": capture_proc.pid,
        "playback_returncode": play_proc.returncode,
        "capture_returncode": capture_proc.returncode,
        "record_stdout_log": str(record_out),
        "record_stderr_log": str(record_err),
        "play_stdout_log": str(play_out),
        "play_stderr_log": str(play_err),
    }


def feed_to_realtimestt(
    captured: WavAudio,
    callback_log: Path,
    args: argparse.Namespace,
    publish: Any | None = None,
) -> dict[str, Any]:
    events: dict[str, list[dict[str, Any]]] = {
        "vad_events": [],
        "realtime_transcript_events": [],
        "final_transcript_events": [],
    }
    started = time.monotonic()

    def t_ms() -> int:
        return int(round((time.monotonic() - started) * 1000))

    def log_event(event: dict[str, Any]) -> None:
        append_jsonl(callback_log, event)
        if publish is None:
            return
        event_type = {
            "vad_start": "listener.audio_started",
            "vad_stop": "listener.audio_stopped",
            "recording_start": "listener.recording_started",
            "recording_stop": "listener.recording_stopped",
            "realtime": "listener.partial_transcript",
            "final": "listener.final_transcript",
        }.get(str(event.get("type")))
        if event_type:
            publish(event_type, {key: value for key, value in event.items() if key != "type"})

    def on_vad_start() -> None:
        event = {"type": "vad_start", "t_ms": t_ms()}
        events["vad_events"].append(event)
        log_event(event)

    def on_vad_stop() -> None:
        event = {"type": "vad_stop", "t_ms": t_ms()}
        events["vad_events"].append(event)
        log_event(event)

    def on_recording_start() -> None:
        event = {"type": "recording_start", "t_ms": t_ms()}
        events["vad_events"].append(event)
        log_event(event)

    def on_recording_stop() -> None:
        event = {"type": "recording_stop", "t_ms": t_ms()}
        events["vad_events"].append(event)
        log_event(event)

    def on_realtime(text: str) -> None:
        event = {"type": "realtime", "t_ms": t_ms(), "text": text}
        events["realtime_transcript_events"].append(event)
        log_event(event)

    config = {
        "use_microphone": False,
        "spinner": False,
        "model": args.model,
        "realtime_model_type": args.realtime_model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": args.realtime_processing_pause,
        "min_length_of_recording": 0,
        "min_gap_between_recordings": 0,
        "post_speech_silence_duration": args.post_speech_silence_duration,
        "no_log_file": True,
        "faster_whisper_vad_filter": False,
        "on_vad_start": on_vad_start,
        "on_vad_stop": on_vad_stop,
        "on_recording_start": on_recording_start,
        "on_recording_stop": on_recording_stop,
        "on_realtime_transcription_update": on_realtime,
        "on_realtime_transcription_stabilized": on_realtime,
    }

    recorder = AudioToTextRecorder(**config)
    chunks_fed = 0
    bytes_fed = 0
    try:
        recorder.start()
        chunk_size = max(1, int(round(captured.sample_rate * args.chunk_ms / 1000.0)))
        lead = np.zeros(int(captured.sample_rate * args.lead_silence), dtype=np.int16)
        tail = np.zeros(int(captured.sample_rate * args.tail_silence), dtype=np.int16)
        feed_samples = np.concatenate([lead, captured.samples.astype(np.int16), tail])
        for start in range(0, feed_samples.size, chunk_size):
            chunk = feed_samples[start:start + chunk_size]
            recorder.feed_audio(chunk, original_sample_rate=captured.sample_rate)
            chunks_fed += 1
            bytes_fed += int(chunk.size * 2)
            time.sleep(chunk.size / float(captured.sample_rate))
        recorder.stop()
        time.sleep(args.post_stop_wait)
        final_text = recorder.text()
        final_event = {"type": "final", "t_ms": t_ms(), "text": final_text}
        events["final_transcript_events"].append(final_event)
        log_event(final_event)
    finally:
        recorder.shutdown()

    return {
        "use_microphone": False,
        "model": args.model,
        "realtime_model": args.realtime_model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "config": config | {
            "on_vad_start": "callback",
            "on_vad_stop": "callback",
            "on_recording_start": "callback",
            "on_recording_stop": "callback",
            "on_realtime_transcription_update": "callback",
            "on_realtime_transcription_stabilized": "callback",
        },
        "chunks_fed": chunks_fed,
        "bytes_fed": bytes_fed,
        "callback_log_path": str(callback_log),
        **events,
    }


def write_live_session_events(
    path: Path,
    *,
    run_id: str,
    source_audio: dict[str, Any],
    captured_audio: dict[str, Any],
    realtimestt_meta: dict[str, Any],
    transcript_meta: dict[str, Any],
) -> dict[str, Any]:
    session_id = f"embry-ingress-{run_id}"
    turn_id = f"turn-{uuid.uuid5(uuid.NAMESPACE_URL, session_id + transcript_meta['normalized_final']).hex[:16]}"
    events: list[dict[str, Any]] = []

    def add(kind: str, payload: dict[str, Any], parent_ids: list[str] | None = None) -> str:
        base = {
            "schema": "embry.live_session_event.v1",
            "session_id": session_id,
            "turn_id": turn_id,
            "run_id": run_id,
            "kind": kind,
            "parent_event_ids": parent_ids or [],
            "payload": payload,
        }
        base["event_id"] = event_id(kind, base)
        events.append(base)
        return base["event_id"]

    ingress_id = add("audio.ingress.captured", {
        "source_audio_sha256": source_audio["sha256"],
        "captured_audio_sha256": captured_audio["sha256"],
        "captured_audio_path": captured_audio["path"],
        "sample_rate": captured_audio["sample_rate"],
        "channels": captured_audio["channels"],
        "duration_ms": captured_audio["duration_ms"],
    })
    stt_final = realtimestt_meta["final_transcript_events"][-1]
    stt_id = add("stt.final", {
        "text": stt_final["text"],
        "normalized_text": transcript_meta["normalized_final"],
        "callback_t_ms": stt_final["t_ms"],
        "engine": "RealtimeSTT",
        "use_microphone": False,
        "callback_log_path": realtimestt_meta["callback_log_path"],
    }, [ingress_id])
    candidate_id = add("turn.input_candidate.created", {
        "text": transcript_meta["final"],
        "normalized_text": transcript_meta["normalized_final"],
        "source_event_id": stt_id,
        "accepted_by_stt": True,
        "accepted_for_tau": False,
        "routing_status": "pending_speaker_gate",
        "speaker_gate_status": "not_run",
        "tau_called": False,
        "chatterbox_called": False,
        "ui_used": False,
    }, [stt_id])

    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    return {
        "schema": "embry.live_session_event_receipt.v1",
        "session_id": session_id,
        "turn_id": turn_id,
        "events_path": str(path),
        "event_count": len(events),
        "event_kinds": [event["kind"] for event in events],
        "audio_ingress_event_id": ingress_id,
        "stt_final_event_id": stt_id,
        "turn_input_candidate_event_id": candidate_id,
        "accepted_by_stt": True,
        "accepted_for_tau": False,
        "routing_status": "pending_speaker_gate",
        "speaker_gate_status": "not_run",
        "tau_called": False,
        "chatterbox_called": False,
        "ui_used": False,
    }


def run_speaker_gate(
    path: Path,
    *,
    run_id: str,
    session_id: str,
    turn_id: str,
    captured_wav: Path,
    stt_final_event_id: str,
    token: str | None,
    device: str,
) -> dict[str, Any]:
    started = time.monotonic()
    events: list[dict[str, Any]] = []

    def add(kind: str, payload: dict[str, Any], parent_ids: list[str] | None = None) -> str:
        base = {
            "schema": "embry.speaker_gate_event.v1",
            "session_id": session_id,
            "turn_id": turn_id,
            "run_id": run_id,
            "kind": kind,
            "parent_event_ids": parent_ids or [],
            "payload": payload,
        }
        base["event_id"] = event_id(kind, base)
        events.append(base)
        return base["event_id"]

    try:
        from pyannote.audio import Pipeline
        import torch
    except Exception as exc:
        rejected_id = add("speaker_gate.rejected.pyannote_unavailable", {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "accepted_for_tau": False,
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }, [stt_final_event_id])
        path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n", encoding="utf-8")
        return {
            "schema": "embry.speaker_gate_receipt.v1",
            "events_path": str(path),
            "event_count": len(events),
            "status": "rejected",
            "decision": "speaker_gate_rejected_pyannote_unavailable",
            "rejection_event_id": rejected_id,
            "pyannote_available": False,
            "accepted_for_tau": False,
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }

    pipeline_loaded = False
    diarization_segments: list[dict[str, Any]] = []
    speaker_labels: list[str] = []
    device_used = "cpu"
    error: dict[str, str] | None = None

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
        pipeline_loaded = True
        if device == "cuda":
            try:
                pipeline.to(torch.device("cuda"))
                device_used = "cuda"
            except Exception as exc:
                error = {"cuda_error_type": type(exc).__name__, "cuda_error": str(exc)}
                device_used = "cpu"
        output = pipeline(str(captured_wav))
        for turn, speaker in output.speaker_diarization:
            diarization_segments.append({
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            })
        speaker_labels = sorted({segment["speaker"] for segment in diarization_segments})
        diarization_id = add("speaker_gate.diarization.completed", {
            "pipeline": "pyannote/speaker-diarization-community-1",
            "device": device_used,
            "captured_wav": str(captured_wav),
            "segment_count": len(diarization_segments),
            "speaker_labels": speaker_labels,
            "segments": diarization_segments,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            **(error or {}),
        }, [stt_final_event_id])
        rejected_id = add("speaker_gate.rejected.not_enrolled", {
            "reason": "no enrolled primary speaker profile was provided",
            "speaker_labels": speaker_labels,
            "segment_count": len(diarization_segments),
            "accepted_for_tau": False,
            "routing_status": "speaker_gate_rejected_not_enrolled",
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }, [diarization_id])
        status = "rejected"
        decision = "speaker_gate_rejected_not_enrolled"
    except Exception as exc:
        rejected_id = add("speaker_gate.rejected.pyannote_runtime_error", {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pipeline_loaded": pipeline_loaded,
            "accepted_for_tau": False,
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }, [stt_final_event_id])
        status = "rejected"
        decision = "speaker_gate_rejected_pyannote_runtime_error"

    path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n", encoding="utf-8")
    return {
        "schema": "embry.speaker_gate_receipt.v1",
        "events_path": str(path),
        "event_count": len(events),
        "event_kinds": [event["kind"] for event in events],
        "status": status,
        "decision": decision,
        "rejection_event_id": rejected_id,
        "pyannote_available": True,
        "pipeline": "pyannote/speaker-diarization-community-1",
        "pipeline_loaded": pipeline_loaded,
        "device": device_used,
        "hf_token_present": bool(token),
        "diarization_segments": diarization_segments,
        "speaker_labels": speaker_labels,
        "segment_count": len(diarization_segments),
        "enrolled_primary_speaker_profile_present": False,
        "accepted_for_tau": False,
        "routing_status": decision,
        "tau_called": False,
        "chatterbox_called": False,
        "ui_used": False,
    }


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-phrase", default=EXPECTED_PHRASE)
    parser.add_argument(
        "--source-wav",
        default="",
        help="Optional real speech WAV to copy into the run as source.wav instead of generating espeak-ng audio.",
    )
    parser.add_argument("--playback-target", default="64")
    parser.add_argument("--capture-target", default="67")
    parser.add_argument("--capture-seconds", type=float, default=5.0)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--realtime-model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--lead-silence", type=float, default=0.25)
    parser.add_argument("--tail-silence", type=float, default=1.0)
    parser.add_argument("--post-stop-wait", type=float, default=0.8)
    parser.add_argument("--post-speech-silence-duration", type=float, default=0.35)
    parser.add_argument("--realtime-processing-pause", type=float, default=0.08)
    parser.add_argument("--max-wer", type=float, default=0.5)
    parser.add_argument("--output-root", default="/tmp/embry-realtimestt-ingress")
    parser.add_argument("--speaker-gate-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-speaker-gate", action="store_true")
    parser.add_argument("--event-service-url", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--turn-id", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = utc_run_id()
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    source_wav = run_dir / "source.wav"
    captured_wav = run_dir / "captured.wav"
    captured_raw = run_dir / "captured.raw"
    callback_log = run_dir / "realtime_stt_callbacks.jsonl"
    live_session_events = run_dir / "live_session_events.jsonl"
    speaker_gate_events = run_dir / "speaker_gate_events.jsonl"
    receipt_path = run_dir / "receipt.json"
    commands_path = run_dir / "commands.txt"
    environment_path = run_dir / "environment.txt"
    commands: list[str] = []
    session_id = args.session_id or f"embry-listener-{run_id}"
    turn_id = args.turn_id or f"turn-{uuid.uuid4().hex[:16]}"
    publish = None
    event_delivery: dict[str, Any] = {"attempted": 0, "accepted": 0, "errors": [], "event_ids": []}
    if args.event_service_url:
        publish, event_delivery = build_event_publisher(
            service_url=args.event_service_url,
            session_id=session_id,
            turn_id=turn_id,
            run_id=run_id,
        )
        publish("listener.ready", {"listener_authority": "unix_pipewire_realtimestt"})

    receipt: dict[str, Any] = {
        "run_id": run_id,
        "proof_rung": PROOF_RUNG,
        "status": "fail",
        "used_ui": False,
        "used_mock_transcript": False,
        "used_typed_prompt": False,
    }

    try:
        generate_source_wav(source_wav, args.expected_phrase, commands, args.source_wav or None)
        source = read_wav(source_wav)
        audio_graph = capture_pipewire_audio(
            run_dir,
            source_wav,
            captured_wav,
            args.playback_target,
            args.capture_target,
            args.capture_seconds,
            commands,
        )
        captured = read_wav(captured_wav)
        write_raw(captured_raw, captured.samples)
        captured_meta = audio_stats(captured_wav)
        captured_meta["raw_path"] = str(captured_raw)
        captured_meta["raw_sha256"] = sha256_file(captured_raw)
        if publish is not None:
            publish("listener.audio_captured", {
                "captured_audio_path": str(captured_wav),
                "captured_audio_sha256": captured_meta["sha256"],
                "captured_audio_non_silent": captured_meta["captured_audio_non_silent"],
            })

        realtimestt_meta = feed_to_realtimestt(captured, callback_log, args, publish)
        final_text = realtimestt_meta["final_transcript_events"][-1]["text"] if realtimestt_meta["final_transcript_events"] else ""
        matches, transcript_meta = transcript_matches(args.expected_phrase, final_text, args.max_wer)
        source_audio_meta = {
            "path": str(source_wav),
            "sha256": sha256_file(source_wav),
            "duration_ms": source.duration_ms,
            "sample_rate": source.sample_rate,
            "channels": source.channels,
            "expected_phrase": args.expected_phrase,
        }
        live_session_meta = write_live_session_events(
            live_session_events,
            run_id=run_id,
            source_audio=source_audio_meta,
            captured_audio=captured_meta,
            realtimestt_meta=realtimestt_meta,
            transcript_meta=transcript_meta,
        ) if final_text.strip() else {
            "schema": "embry.live_session_event_receipt.v1",
            "events_path": str(live_session_events),
            "event_count": 0,
            "accepted_by_stt": False,
            "accepted_for_tau": False,
            "routing_status": "no_final_transcript",
            "speaker_gate_status": "not_run",
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }
        speaker_gate_meta = run_speaker_gate(
            speaker_gate_events,
            run_id=run_id,
            session_id=live_session_meta.get("session_id", f"embry-ingress-{run_id}"),
            turn_id=live_session_meta.get("turn_id", ""),
            captured_wav=captured_wav,
            stt_final_event_id=live_session_meta.get("stt_final_event_id", ""),
            token=os.environ.get("HF_TOKEN"),
            device=args.speaker_gate_device,
        ) if final_text.strip() and not args.skip_speaker_gate else {
            "schema": "embry.speaker_gate_receipt.v1",
            "events_path": str(speaker_gate_events),
            "event_count": 0,
            "status": "skipped",
            "decision": "speaker_gate_skipped",
            "accepted_for_tau": False,
            "routing_status": "speaker_gate_skipped",
            "tau_called": False,
            "chatterbox_called": False,
            "ui_used": False,
        }
        acceptance = {
            "audio_was_played_through_local_audio_graph": audio_graph["playback_returncode"] == 0,
            "audio_was_captured_from_local_audio_graph": captured_wav.exists() and captured.duration_ms > 0,
            "captured_audio_non_silent": bool(captured_meta["captured_audio_non_silent"]),
            "captured_audio_fed_to_realtimestt": realtimestt_meta["chunks_fed"] > 0 and realtimestt_meta["bytes_fed"] > 0,
            "realtimestt_realtime_events_seen": len(realtimestt_meta["realtime_transcript_events"]) > 0,
            "realtimestt_final_transcript_seen": bool(final_text.strip()),
            "final_transcript_matches_expected_phrase": bool(matches),
        }
        if args.event_service_url:
            acceptance["live_event_service_delivery"] = (
                event_delivery["attempted"] > 0
                and event_delivery["attempted"] == event_delivery["accepted"]
                and not event_delivery["errors"]
            )
        acceptance["pass"] = all(acceptance.values())

        receipt.update({
            "status": "pass" if acceptance["pass"] else "fail",
            "ok": acceptance["pass"],
            "mocked": False,
            "live": True,
            "listener_authority": "unix_pipewire_realtimestt",
            "session_id": session_id,
            "turn_id": turn_id,
            "repos": {
                "realtimestt": repo_info(ROOT),
                "pi_mono": repo_info(Path("/home/graham/workspace/experiments/pi-mono")),
                "agent_skills": repo_info(Path("/home/graham/workspace/experiments/agent-skills")),
            },
            "source_audio": source_audio_meta,
            "audio_graph": audio_graph,
            "captured_audio": captured_meta,
            "realtimestt": realtimestt_meta,
            "live_session": live_session_meta,
            "speaker_gate": speaker_gate_meta,
            "event_service": {
                "url": args.event_service_url or None,
                "session_id": session_id,
                "turn_id": turn_id,
                "delivery": event_delivery,
            },
            "transcript": transcript_meta,
            "acceptance": acceptance,
        })
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        receipt.setdefault("acceptance", {
            "audio_was_played_through_local_audio_graph": False,
            "audio_was_captured_from_local_audio_graph": False,
            "captured_audio_non_silent": False,
            "captured_audio_fed_to_realtimestt": False,
            "realtimestt_realtime_events_seen": False,
            "realtimestt_final_transcript_seen": False,
            "final_transcript_matches_expected_phrase": False,
            "pass": False,
        })
    finally:
        commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
        env_parts = [
            "$ wpctl status\n" + run_command(["wpctl", "status"]).stdout,
            "$ pw-cli ls Node\n" + run_command(["pw-cli", "ls", "Node"]).stdout,
            "$ git status --short\n" + run_command(["git", "-C", str(ROOT), "status", "--short"]).stdout,
        ]
        write_text(environment_path, "\n\n".join(env_parts))
        receipt["commands_path"] = str(commands_path)
        receipt["environment_path"] = str(environment_path)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    print(str(receipt_path))
    print(json.dumps({
        "status": receipt.get("status"),
        "final": receipt.get("transcript", {}).get("final", ""),
        "acceptance": receipt.get("acceptance", {}),
    }, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
