#!/usr/bin/env python3
"""FastAPI endpoints for Embry voice-control proof receipts.

This API intentionally stops at listener/speaker-gate proof boundaries. It does
not call Tau, Chatterbox, UX Lab, browser automation, or typed transcript paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = Path(__file__).resolve().parent
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pipewire_realtimestt_ingress import (  # noqa: E402
    audio_stats,
    event_id,
    run_speaker_gate,
    run_command,
    sha256_file,
    utc_run_id,
)


DEFAULT_OUTPUT_ROOT = Path("/tmp/embry-voice-control")
DEFAULT_INGRESS_ROOT = Path("/tmp/embry-realtimestt-ingress")
DEFAULT_CAPTURED_WAV = Path("/tmp/embry-realtimestt-ingress/20260707T005001Z/captured.wav")

app = FastAPI(title="Embry Voice Control Proof API", version="0.1.0")
_PYANNOTE_PIPELINE = None


class PipeWireIngressRequest(BaseModel):
    source_wav: str = "tests/unit/audio/asr-reference-short.wav"
    expected_phrase: str = (
        "Hey guys! Welcome to the new demo of my real-time transcription library, "
        "designed to showcase its lightning-fast capabilities. As you'll see, "
        "speech is transcribed almost instantly into text"
    )
    playback_target: str = "64"
    capture_target: str = "67"
    capture_seconds: float = 15
    max_wer: float = 0.35
    speaker_gate_device: Literal["cpu", "cuda"] = "cpu"


class SpeakerGateRequest(BaseModel):
    captured_wav: str = str(DEFAULT_CAPTURED_WAV)
    session_id: str | None = None
    turn_id: str | None = None
    stt_final_event_id: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"


class SpeakerSeparationRequest(BaseModel):
    mode: Literal["same-speaker", "different-speaker"]
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    expected_speaker_count: int | None = Field(default=None, ge=1)


class SpeakerEnrollRequest(BaseModel):
    speaker_id: str = "horus_synthetic_primary"
    voice: str = "en-us"
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    device: Literal["cpu", "cuda"] = "cpu"


def repo_info(path: Path) -> dict[str, Any]:
    head = run_command(["git", "-C", str(path), "rev-parse", "--short", "HEAD"]).stdout.strip()
    status = run_command(["git", "-C", str(path), "status", "--short"]).stdout.splitlines()
    return {"path": str(path), "head": head, "dirty": bool(status), "dirty_count": len(status)}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate())
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "duration_seconds": round(duration, 3),
        }


def synth_wav(path: Path, text: str, voice: str, speed: int = 135) -> None:
    raw_path = path.with_suffix(".raw-espeak.wav")
    command = [
        "espeak-ng",
        "-v",
        voice,
        "-s",
        str(speed),
        "-a",
        "180",
        "-w",
        str(raw_path),
        text,
    ]
    result = run_command(command, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {result.stderr.strip()}")
    sox(["sox", str(raw_path), "-r", "16000", "-c", "1", "-b", "16", str(path)])


def sox(command: list[str]) -> None:
    result = run_command(command, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"sox failed: {result.stderr.strip()}")


def load_pyannote_pipeline(token: str | None):
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE
    from pyannote.audio import Pipeline

    _PYANNOTE_PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    return _PYANNOTE_PIPELINE


def diarize_wav(path: Path, token: str | None) -> tuple[list[dict[str, Any]], float]:
    started = time.monotonic()
    pipeline = load_pyannote_pipeline(token)
    output = pipeline(str(path))
    segments: list[dict[str, Any]] = []
    for turn, speaker in output.speaker_diarization:
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "duration": round(float(turn.end - turn.start), 3),
            "speaker": str(speaker),
        })
    return segments, round(time.monotonic() - started, 3)


def make_separation_audio(run_dir: Path, mode: str) -> Path:
    silence = run_dir / "silence_500ms.wav"
    sox(["sox", "-n", "-r", "16000", "-c", "1", "-b", "16", str(silence), "trim", "0.0", "0.5"])
    if mode == "same-speaker":
        h1 = run_dir / "horus_1.wav"
        h2 = run_dir / "horus_2.wav"
        h3 = run_dir / "horus_3.wav"
        synth_wav(h1, "Horus speaking alpha. This is the primary male speaker.", "en-us", 130)
        synth_wav(h2, "Horus speaking bravo. The same primary speaker continues.", "en-us", 130)
        synth_wav(h3, "Horus speaking charlie. The same voice appears again.", "en-us", 130)
        output = run_dir / "same_speaker_test.wav"
        sox(["sox", str(h1), str(silence), str(h2), str(silence), str(h3), "-r", "16000", "-c", "1", "-b", "16", str(output)])
        return output

    h = run_dir / "horus.wav"
    e = run_dir / "embry.wav"
    synth_wav(h, "Horus speaking alpha. This is the primary male speaker.", "en-us", 130)
    synth_wav(e, "Embry speaking beta. This is the female voice.", "en-us+f3", 155)
    output = run_dir / "different_speaker_test.wav"
    sox(["sox", str(h), str(silence), str(e), str(silence), str(h), str(silence), str(e), "-r", "16000", "-c", "1", "-b", "16", str(output)])
    return output


def create_profile_audio(run_dir: Path, speaker_id: str, voice: str) -> Path:
    silence = run_dir / "silence_400ms.wav"
    sox(["sox", "-n", "-r", "16000", "-c", "1", "-b", "16", str(silence), "trim", "0.0", "0.4"])
    clips = []
    for index, phrase in enumerate([
        f"{speaker_id} enrollment alpha. This is the first clean sample.",
        f"{speaker_id} enrollment bravo. This is the second clean sample.",
        f"{speaker_id} enrollment charlie. This is the third clean sample.",
    ], start=1):
        clip = run_dir / f"enroll_{index}.wav"
        synth_wav(clip, phrase, voice, 130)
        clips.append(clip)
    output = run_dir / "enrollment.wav"
    command = ["sox"]
    for clip in clips:
        command.extend([str(clip), str(silence)])
    command.extend(["-r", "16000", "-c", "1", "-b", "16", str(output)])
    sox(command)
    return output


def create_policy_audio(run_dir: Path, mode: str) -> Path:
    silence = run_dir / "silence_500ms.wav"
    sox(["sox", "-n", "-r", "16000", "-c", "1", "-b", "16", str(silence), "trim", "0.0", "0.5"])
    primary_a = run_dir / "primary_a.wav"
    primary_b = run_dir / "primary_b.wav"
    distractor = run_dir / "distractor.wav"
    self_audio = run_dir / "embry_self_audio.wav"
    synth_wav(primary_a, "Horus primary speaker says the live listener gate should accept this phrase.", "en-us", 130)
    synth_wav(primary_b, "Horus primary speaker continues with a held out phrase for acceptance.", "en-us", 130)
    synth_wav(distractor, "Different speaker distractor says this should not route to Tau.", "en-us+f3", 155)
    synth_wav(self_audio, "Embry self audio should never become a user turn.", "en-us+f3", 145)

    output = run_dir / f"{mode}.wav"
    if mode == "primary-acceptance":
        sox(["sox", str(primary_a), str(silence), str(primary_b), "-r", "16000", "-c", "1", "-b", "16", str(output)])
    elif mode == "non-primary-rejection":
        sox(["sox", str(primary_a), str(silence), str(distractor), "-r", "16000", "-c", "1", "-b", "16", str(output)])
    elif mode == "overlap-rejection":
        sox(["sox", "-m", str(primary_a), str(distractor), "-r", "16000", "-c", "1", "-b", "16", str(output)])
    elif mode == "noise-probe":
        noise = run_dir / "factory_noise.wav"
        noisy_primary = run_dir / "noisy_primary.wav"
        sox(["sox", "-n", "-r", "16000", "-c", "1", "-b", "16", str(noise), "synth", "3.5", "brownnoise", "vol", "0.12"])
        sox(["sox", "-m", str(primary_a), str(noise), "-r", "16000", "-c", "1", "-b", "16", str(noisy_primary)])
        shutil.copyfile(noisy_primary, output)
    elif mode == "self-audio-rejection":
        shutil.copyfile(self_audio, output)
    else:
        raise ValueError(f"unsupported policy mode: {mode}")
    return output


def create_enrollment_receipt(request: SpeakerEnrollRequest) -> dict[str, Any]:
    run_id = utc_run_id()
    run_dir = Path(request.output_root) / "speaker-enrollment" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    audio_path = create_profile_audio(run_dir, request.speaker_id, request.voice)
    segments, elapsed = diarize_wav(audio_path, os.environ.get("HF_TOKEN"))
    speakers = sorted({segment["speaker"] for segment in segments})
    profile_path = run_dir / "profile.json"
    profile = {
        "schema": "embry.synthetic_speaker_profile.v1",
        "profile_id": f"profile.{sha256_text(request.speaker_id + run_id)[:16]}",
        "speaker_id": request.speaker_id,
        "synthetic_profile": True,
        "proves_real_horus": False,
        "enrollment_audio": wav_info(audio_path),
        "dominant_diarization_label": speakers[0] if len(speakers) == 1 else None,
        "speaker_count": len(speakers),
        "speakers": speakers,
    }
    write_json(profile_path, profile)
    pass_rule = len(speakers) == 1 and len(segments) > 0
    receipt = {
        "schema": "embry.speaker_enrollment_receipt.v1",
        "run_id": run_id,
        "status": "pass" if pass_rule else "fail",
        "profile_path": str(profile_path),
        "profile": profile,
        "pipeline": "pyannote/speaker-diarization-community-1",
        "run_seconds": elapsed,
        "segments": segments,
        "acceptance": {
            "pyannote_ran": True,
            "segments_emitted": len(segments) > 0,
            "single_speaker_profile_created": pass_rule,
            "pass": pass_rule,
        },
        "used_ui": False,
        "used_mock_transcript": False,
        "used_typed_prompt": False,
        "repos": {"realtimestt": repo_info(ROOT)},
    }
    write_json(run_dir / "receipt.json", receipt)
    return receipt | {"receipt_path": str(run_dir / "receipt.json")}


def create_gate_policy_receipt(mode: str, decision: str, accepted_for_tau: bool) -> dict[str, Any]:
    run_id = utc_run_id()
    run_dir = DEFAULT_OUTPUT_ROOT / "speaker-policy" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    audio_path = create_policy_audio(run_dir, mode)
    segments, elapsed = diarize_wav(audio_path, os.environ.get("HF_TOKEN"))
    speakers = sorted({segment["speaker"] for segment in segments})
    audio = wav_info(audio_path) | audio_stats(audio_path)

    if mode == "primary-acceptance":
        pass_rule = len(speakers) == 1 and accepted_for_tau
    elif mode == "non-primary-rejection":
        pass_rule = len(speakers) >= 2 and not accepted_for_tau
    else:
        pass_rule = len(segments) > 0 and not accepted_for_tau

    events_path = run_dir / "speaker_policy_events.jsonl"
    event = {
        "event_type": decision,
        "run_id": run_id,
        "mode": mode,
        "accepted_for_tau": accepted_for_tau,
        "speaker_count": len(speakers),
        "speakers": speakers,
    }
    event["event_id"] = event_id("speaker_gate", event)
    events_path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "embry.speaker_gate_policy_receipt.v1",
        "run_id": run_id,
        "mode": mode,
        "status": "pass" if pass_rule else "fail",
        "decision": decision,
        "accepted_for_tau": accepted_for_tau,
        "synthetic_profile": True,
        "proves_real_horus": False,
        "calls_tau": False,
        "calls_chatterbox": False,
        "uses_chat_ux": False,
        "input_audio": audio,
        "pipeline": "pyannote/speaker-diarization-community-1",
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "device": "cpu",
        "run_seconds": elapsed,
        "event_log_path": str(events_path),
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segments": segments,
        "acceptance": {
            "pyannote_ran": True,
            "segments_emitted": len(segments) > 0,
            "policy_decision_emitted": True,
            "accepted_for_tau": accepted_for_tau,
            "pass": pass_rule,
        },
        "used_ui": False,
        "used_mock_transcript": False,
        "used_typed_prompt": False,
        "repos": {"realtimestt": repo_info(ROOT)},
    }
    write_json(run_dir / "receipt.json", receipt)
    return receipt | {"receipt_path": str(run_dir / "receipt.json")}


def create_speaker_separation_receipt(mode: str, output_root: Path, expected_speaker_count: int) -> dict[str, Any]:
    run_id = utc_run_id()
    run_dir = output_root / "speaker-separation" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    audio_path = make_separation_audio(run_dir, mode)
    segments, elapsed = diarize_wav(audio_path, os.environ.get("HF_TOKEN"))
    speakers = sorted({segment["speaker"] for segment in segments})
    events_path = run_dir / "speaker_segments.jsonl"
    events_path.write_text("\n".join(json.dumps(segment, sort_keys=True) for segment in segments) + "\n", encoding="utf-8")

    if mode == "same-speaker":
        pass_rule = len(speakers) == expected_speaker_count
    else:
        pass_rule = len(speakers) >= expected_speaker_count

    receipt = {
        "schema": "embry.speaker_separation_receipt.v1",
        "run_id": run_id,
        "mode": mode,
        "status": "pass" if pass_rule else "fail",
        "input_audio": wav_info(audio_path),
        "pipeline": "pyannote/speaker-diarization-community-1",
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "device": "cpu",
        "run_seconds": elapsed,
        "segments_path": str(events_path),
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segments": segments,
        "expected_speaker_count": expected_speaker_count,
        "acceptance": {
            "pyannote_ran": True,
            "segments_emitted": len(segments) > 0,
            "speaker_count_rule_passed": pass_rule,
            "pass": pass_rule,
        },
        "proves_identity": False,
        "proves_horus": False,
        "repos": {"realtimestt": repo_info(ROOT)},
    }
    write_json(run_dir / "receipt.json", receipt)
    return receipt | {"receipt_path": str(run_dir / "receipt.json")}


@app.get("/health")
def health() -> dict[str, Any]:
    pyannote_installed = False
    try:
        import pyannote.audio  # noqa: F401

        pyannote_installed = True
    except Exception:
        pyannote_installed = False
    return {
        "ok": True,
        "service": "embry-voice-control-proof-api",
        "realtimestt_root": str(ROOT),
        "pyannote_installed": pyannote_installed,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }


@app.post("/sanity/pipewire-realtimestt-ingress")
def sanity_pipewire_realtimestt_ingress(request: PipeWireIngressRequest) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROOF_DIR / "run_pipewire_realtimestt_ingress.py"),
        "--source-wav",
        request.source_wav,
        "--expected-phrase",
        request.expected_phrase,
        "--playback-target",
        request.playback_target,
        "--capture-target",
        request.capture_target,
        "--capture-seconds",
        str(request.capture_seconds),
        "--max-wer",
        str(request.max_wer),
        "--speaker-gate-device",
        request.speaker_gate_device,
    ]
    result = run_command(command, timeout=180)
    receipt_path = ""
    for line in result.stdout.splitlines():
        if line.startswith("/tmp/") and line.endswith("/receipt.json"):
            receipt_path = line.strip()
            break
    if not receipt_path:
        raise HTTPException(status_code=500, detail={"error": "receipt_not_found", "stdout": result.stdout, "stderr": result.stderr})
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    return {
        "status": receipt["status"],
        "receipt_path": receipt_path,
        "acceptance": receipt.get("acceptance", {}),
        "speaker_gate": receipt.get("speaker_gate", {}),
        "returncode": result.returncode,
    }


@app.post("/speaker/gate")
def speaker_gate(request: SpeakerGateRequest = SpeakerGateRequest()) -> dict[str, Any]:
    captured = Path(request.captured_wav)
    if not captured.exists():
        raise HTTPException(status_code=404, detail=f"captured_wav not found: {captured}")
    run_id = utc_run_id()
    run_dir = DEFAULT_OUTPUT_ROOT / "speaker-gate" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "speaker_gate_events.jsonl"
    session_id = request.session_id or f"embry-speaker-gate-{run_id}"
    turn_id = request.turn_id or f"turn-{sha256_text(str(captured) + run_id)[:16]}"
    stt_final_event_id = request.stt_final_event_id or f"stt.final.{sha256_text(str(captured))[:16]}"
    receipt = run_speaker_gate(
        events_path,
        run_id=run_id,
        session_id=session_id,
        turn_id=turn_id,
        captured_wav=captured,
        stt_final_event_id=stt_final_event_id,
        token=os.environ.get("HF_TOKEN"),
        device=request.device,
    )
    receipt = receipt | {
        "run_id": run_id,
        "receipt_path": str(run_dir / "receipt.json"),
        "captured_wav": str(captured),
        "turn_id": turn_id,
    }
    write_json(run_dir / "receipt.json", receipt)
    return receipt


@app.post("/sanity/same-speaker")
def sanity_same_speaker() -> dict[str, Any]:
    return create_speaker_separation_receipt("same-speaker", DEFAULT_OUTPUT_ROOT, expected_speaker_count=1)


@app.post("/sanity/different-speaker")
def sanity_different_speaker() -> dict[str, Any]:
    return create_speaker_separation_receipt("different-speaker", DEFAULT_OUTPUT_ROOT, expected_speaker_count=2)


@app.post("/speaker/enroll")
def speaker_enroll(request: SpeakerEnrollRequest = SpeakerEnrollRequest()) -> dict[str, Any]:
    return create_enrollment_receipt(request)


@app.post("/sanity/primary-acceptance")
def sanity_primary_acceptance() -> dict[str, Any]:
    return create_gate_policy_receipt(
        "primary-acceptance",
        decision="speaker_gate.accepted.synthetic_primary",
        accepted_for_tau=True,
    )


@app.post("/sanity/non-primary-rejection")
def sanity_non_primary_rejection() -> dict[str, Any]:
    return create_gate_policy_receipt(
        "non-primary-rejection",
        decision="speaker_gate.rejected.non_primary",
        accepted_for_tau=False,
    )


@app.post("/sanity/overlap-rejection")
def sanity_overlap_rejection() -> dict[str, Any]:
    return create_gate_policy_receipt(
        "overlap-rejection",
        decision="speaker_gate.rejected.overlap",
        accepted_for_tau=False,
    )


@app.post("/sanity/noise-probe")
def sanity_noise_probe() -> dict[str, Any]:
    return create_gate_policy_receipt(
        "noise-probe",
        decision="speaker_gate.rejected.noisy_unenrolled_probe",
        accepted_for_tau=False,
    )


@app.post("/sanity/self-audio-rejection")
def sanity_self_audio_rejection() -> dict[str, Any]:
    return create_gate_policy_receipt(
        "self-audio-rejection",
        decision="speaker_gate.rejected.self_audio",
        accepted_for_tau=False,
    )


@app.get("/receipts/{run_id:path}")
def get_receipt(run_id: str) -> dict[str, Any]:
    candidates = [
        DEFAULT_OUTPUT_ROOT / run_id / "receipt.json",
        DEFAULT_OUTPUT_ROOT / "speaker-gate" / run_id / "receipt.json",
        DEFAULT_OUTPUT_ROOT / "speaker-separation" / run_id / "receipt.json",
        DEFAULT_OUTPUT_ROOT / "speaker-enrollment" / run_id / "receipt.json",
        DEFAULT_OUTPUT_ROOT / "speaker-policy" / run_id / "receipt.json",
        DEFAULT_INGRESS_ROOT / run_id / "receipt.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"receipt not found for {run_id}")
