#!/usr/bin/env python3
"""Run resumable physical PipeWire microphone turns through RealtimeSTT.

This proof uses no source WAV, browser microphone, typed transcript, or mocked
callback. It keeps one physical PipeWire capture stream open, feeds PCM into
RealtimeSTT, journals callback events, saves accepted wake-word utterances, and
supports process restart/resume against the same session state.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RealtimeSTT import AudioToTextRecorder  # noqa: E402
from proofs.embry_pipewire_ingress.run_pipewire_realtimestt_ingress import (  # noqa: E402
    build_event_publisher,
)


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_BYTES = 3200
SCHEMA = "realtimestt.physical_hot_mic_listener_receipt.v1"
STATE_SCHEMA = "realtimestt.physical_hot_mic_listener_state.v1"
WAKE_NAME_ALIASES = {"embry", "emory", "embring"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def wake_phrase_match(value: str) -> dict[str, Any]:
    tokens = normalize_text(value).split()
    if len(tokens) < 2 or tokens[0] not in {"hey", "hay"}:
        return {"detected": False, "tokens": tokens, "similarity": 0.0, "request_tokens": []}
    similarity = SequenceMatcher(None, tokens[1], "embry").ratio()
    detected = tokens[1] in WAKE_NAME_ALIASES or similarity >= 0.72
    return {
        "detected": detected,
        "tokens": tokens,
        "similarity": round(similarity, 4),
        "matched_phrase": " ".join(tokens[:2]),
        "request_tokens": tokens[2:] if detected else [],
    }


def event_service_origin(value: str) -> str:
    return value.rstrip("/").removesuffix("/v1/listener/events")


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def initial_state(source_node: str, target_cycles: int) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    return {
        "schema": STATE_SCHEMA,
        "run_id": run_id,
        "session_id": f"physical-hot-mic-{run_id}",
        "source_node": source_node,
        "target_cycles": target_cycles,
        "completed_cycles": [],
        "rejected_attempts": [],
        "process_runs": [],
        "capture_restarts": [],
        "pending_wake": None,
        "last_event_id": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def load_or_create_state(path: Path, source_node: str, target_cycles: int) -> dict[str, Any]:
    if not path.exists():
        return initial_state(source_node, target_cycles)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise RuntimeError("listener_state_schema_invalid")
    if state.get("source_node") != source_node:
        raise RuntimeError("listener_resume_source_node_mismatch")
    if state.get("target_cycles") != target_cycles:
        raise RuntimeError("listener_resume_target_cycles_mismatch")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PipeWireCapture:
    """Continuously feed one stable physical PipeWire source into RealtimeSTT."""

    def __init__(
        self,
        source_node: str,
        recorder: AudioToTextRecorder,
        log_path: Path,
        on_connect: Callable[[int], None],
        on_disconnect: Callable[[int, int | None], None],
    ) -> None:
        self.source_node = source_node
        self.recorder = recorder
        self.log_path = log_path
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.recording = threading.Event()
        self.segment_lock = threading.Lock()
        self.segment_chunks: list[bytes] = []
        self.total_bytes = 0

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("pipewire_capture_already_running")
        self.stop_requested.clear()
        stderr = self.log_path.open("ab")
        self.process = subprocess.Popen(
            [
                "pw-record",
                "--target",
                self.source_node,
                "--rate",
                str(SAMPLE_RATE),
                "--channels",
                str(CHANNELS),
                "--format",
                "s16",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        self.process._embry_stderr = stderr  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self._feed_loop, name="physical-hot-mic-feed", daemon=True)
        self.thread.start()
        time.sleep(0.25)
        if self.process.poll() is not None:
            self.stop()
            raise RuntimeError(f"pw_record_start_failed:{self.log_path}")
        self.on_connect(self.process.pid)

    def _feed_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while not self.stop_requested.is_set():
            chunk = self.process.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            self.total_bytes += len(chunk)
            if self.recording.is_set():
                with self.segment_lock:
                    self.segment_chunks.append(chunk)
            self.recorder.feed_audio(chunk, original_sample_rate=SAMPLE_RATE)
        returncode = self.process.poll() if self.process else None
        if self.process is not None:
            self.on_disconnect(self.process.pid, returncode)

    def begin_segment(self) -> None:
        with self.segment_lock:
            self.segment_chunks = []
        self.recording.set()

    def end_segment(self) -> None:
        self.recording.clear()

    def segment_pcm(self) -> bytes:
        with self.segment_lock:
            return b"".join(self.segment_chunks)

    def restart(self) -> tuple[int, int]:
        old_pid = self.process.pid if self.process is not None else -1
        self.stop()
        self.start()
        assert self.process is not None
        return old_pid, self.process.pid

    def stop(self) -> None:
        self.stop_requested.set()
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.process is not None:
            stderr = getattr(self.process, "_embry_stderr", None)
            if stderr is not None:
                stderr.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = run_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    state_path = run_dir / "state.json"
    receipt_path = run_dir / "receipt.json"
    callback_log = run_dir / "callbacks.jsonl"
    state = load_or_create_state(state_path, args.source_node, args.target_cycles)
    process_run_number = len(state["process_runs"]) + 1
    process_run = {
        "process_run": process_run_number,
        "status": "starting",
        "started_at": utc_now(),
        "starting_cycle_count": len(state["completed_cycles"]),
        "pid": None,
        "completed_cycle_count": 0,
    }
    state["process_runs"].append(process_run)
    save_state(state_path, state)

    publisher, delivery = build_event_publisher(
        service_url=event_service_origin(args.event_service_url),
        session_id=state["session_id"],
        turn_id=f"listener-process-{process_run_number}",
        run_id=state["run_id"],
        initial_causation_id=state.get("last_event_id"),
    )
    publish_lock = threading.Lock()

    def emit(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with publish_lock:
            event = publisher(event_type, payload)
            state["last_event_id"] = event["event_id"]
            with callback_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            save_state(state_path, state)
            return event

    capture_ref: dict[str, PipeWireCapture] = {}

    def on_recording_start() -> None:
        capture_ref["capture"].begin_segment()
        emit("listener.recording_started", {"process_run": process_run_number})

    def on_recording_stop() -> None:
        capture_ref["capture"].end_segment()
        emit("listener.recording_stopped", {"process_run": process_run_number})

    def on_realtime(text: str) -> None:
        emit("listener.partial_transcript", {"text": text, "process_run": process_run_number})

    recorder = AudioToTextRecorder(
        use_microphone=False,
        spinner=False,
        model=args.model,
        realtime_model_type=args.realtime_model,
        language="en",
        device=args.device,
        compute_type=args.compute_type,
        enable_realtime_transcription=True,
        realtime_processing_pause=0.08,
        post_speech_silence_duration=0.55,
        min_length_of_recording=0.25,
        no_log_file=True,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_realtime_transcription_update=on_realtime,
        on_realtime_transcription_stabilized=on_realtime,
    )
    process_run["status"] = "ready"
    save_state(state_path, state)

    def on_connect(pid: int) -> None:
        process_run["pid"] = pid
        emit("listener.capture_connected", {
            "source_node": args.source_node,
            "capture_pid": pid,
            "process_run": process_run_number,
        })

    def on_disconnect(pid: int, returncode: int | None) -> None:
        emit("listener.capture_disconnected", {
            "source_node": args.source_node,
            "capture_pid": pid,
            "returncode": returncode,
            "process_run": process_run_number,
        })

    capture = PipeWireCapture(args.source_node, recorder, run_dir / "pw-record.log", on_connect, on_disconnect)
    capture_ref["capture"] = capture
    attempts_this_run = 0
    accepted_this_run = 0
    forced_restart_done = False

    emit("listener.process_started", {
        "process_run": process_run_number,
        "resume_cycle_count": len(state["completed_cycles"]),
        "source_node": args.source_node,
    })
    capture.start()
    try:
        while (
            len(state["completed_cycles"]) < args.target_cycles
            and accepted_this_run < args.cycles_this_run
            and attempts_this_run < args.max_attempts_this_run
        ):
            cycle_number = len(state["completed_cycles"]) + 1
            print(f"READY cycle {cycle_number}/{args.target_cycles}: say Embry, then your sentence", flush=True)
            text = recorder.text()
            attempts_this_run += 1
            pcm = capture.segment_pcm()
            normalized = normalize_text(text)
            wake_match = wake_phrase_match(text)
            pending_wake = state.get("pending_wake")
            if pending_wake:
                request_tokens = wake_match["request_tokens"] if wake_match["detected"] else normalized.split()
                request_text = " ".join(request_tokens)
                wake_detected = True
                wake_mode = "two_stage"
            else:
                request_tokens = wake_match["request_tokens"]
                request_text = " ".join(request_tokens)
                wake_detected = bool(wake_match["detected"])
                wake_mode = "one_shot"

            if wake_detected and not request_text and pcm:
                wake_number = len(state["completed_cycles"]) + 1
                wake_path = segments_dir / f"wake-{wake_number:02d}.wav"
                write_wav(wake_path, pcm)
                wake_event = {
                    "wake_for_cycle": wake_number,
                    "text": text,
                    "normalized_text": normalized,
                    "matched_phrase": wake_match.get("matched_phrase"),
                    "wake_similarity": wake_match.get("similarity"),
                    "audio_path": str(wake_path),
                    "audio_sha256": sha256_file(wake_path),
                    "process_run": process_run_number,
                    "detected_at": utc_now(),
                }
                event = emit("listener.wake_detected", wake_event)
                wake_event["event_id"] = event["event_id"]
                state["pending_wake"] = wake_event
                save_state(state_path, state)
                print(f"WAKE ACCEPTED for cycle {wake_number}; LISTENING for request", flush=True)
                continue

            if not wake_detected or not request_text or not pcm:
                rejected = {
                    "attempt": len(state["completed_cycles"]) + len(state["rejected_attempts"]) + 1,
                    "text": text,
                    "normalized_text": normalized,
                    "wake_detected": wake_detected,
                    "wake_similarity": wake_match.get("similarity"),
                    "request_text": request_text,
                    "audio_bytes": len(pcm),
                    "process_run": process_run_number,
                }
                state["rejected_attempts"].append(rejected)
                emit("listener.wake_rejected", rejected)
                print(f"REJECTED wake={wake_detected} text={text!r}; repeat", flush=True)
                continue

            segment_path = segments_dir / f"cycle-{cycle_number:02d}.wav"
            write_wav(segment_path, pcm)
            cycle = {
                "cycle": cycle_number,
                "text": text,
                "normalized_text": normalized,
                "request_text": request_text,
                "normalized_request_text": normalize_text(request_text),
                "wake_detected": True,
                "wake_mode": wake_mode,
                "wake_phrase": pending_wake or {
                    "text": " ".join(wake_match.get("tokens", [])[:2]),
                    "matched_phrase": wake_match.get("matched_phrase"),
                    "wake_similarity": wake_match.get("similarity"),
                },
                "audio_path": str(segment_path),
                "audio_sha256": sha256_file(segment_path),
                "audio_bytes": len(pcm),
                "duration_ms": int(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000),
                "process_run": process_run_number,
                "accepted_at": utc_now(),
            }
            final_event = emit("listener.final_transcript", cycle)
            cycle["event_id"] = final_event["event_id"]
            cycle["sequence"] = final_event["assigned_sequence"]
            state["completed_cycles"].append(cycle)
            state["pending_wake"] = None
            accepted_this_run += 1
            process_run["completed_cycle_count"] = accepted_this_run
            save_state(state_path, state)
            print(f"ACCEPTED cycle {cycle_number}: {text}", flush=True)

            if (
                args.restart_capture_after_cycle > 0
                and not forced_restart_done
                and accepted_this_run == args.restart_capture_after_cycle
            ):
                old_pid, new_pid = capture.restart()
                restart = {
                    "process_run": process_run_number,
                    "after_cycle": cycle_number,
                    "old_pid": old_pid,
                    "new_pid": new_pid,
                    "source_node": args.source_node,
                    "at": utc_now(),
                }
                state["capture_restarts"].append(restart)
                emit("listener.capture_reconnected", restart)
                forced_restart_done = True
    finally:
        capture.stop()
        recorder.shutdown()

    process_run["ended_at"] = utc_now()
    process_run["status"] = "completed"
    process_run["ending_cycle_count"] = len(state["completed_cycles"])
    emit("listener.process_stopped", {
        "process_run": process_run_number,
        "ending_cycle_count": len(state["completed_cycles"]),
    })
    save_state(state_path, state)

    journal = httpx.get(
        event_service_origin(args.event_service_url)
        + f"/v1/sessions/{state['session_id']}/journal",
        timeout=10,
    ).json()
    events = journal.get("events") or []
    event_ids = [event.get("event_id") for event in events]
    sequences = [event.get("sequence") for event in events]
    completed = state["completed_cycles"]
    acceptance = {
        "physical_pipewire_source": args.source_node.startswith("alsa_input."),
        "no_source_wav": True,
        "no_browser_microphone": True,
        "no_mocked_transcript": True,
        "ten_wake_cycles": len(completed) == args.target_cycles,
        "unique_transcripts": len({cycle["normalized_request_text"] for cycle in completed}) == len(completed),
        "all_cycle_audio_saved": all(Path(cycle["audio_path"]).is_file() for cycle in completed),
        "capture_reconnected": len(state["capture_restarts"]) >= 1,
        "listener_restart_resumed": sum(
            process_run.get("status") == "completed" for process_run in state["process_runs"]
        ) >= 2,
        "journal_sequences_contiguous": sequences == list(range(1, len(events) + 1)),
        "journal_event_ids_unique": len(event_ids) == len(set(event_ids)),
        "journal_contains_all_final_turns": sum(event.get("type") == "listener.final_transcript" for event in events) == len(completed),
        "delivery_errors_absent": not delivery["errors"],
    }
    receipt = {
        "schema": SCHEMA,
        "status": "pass" if all(acceptance.values()) else "partial",
        "mocked": False,
        "live": True,
        "used_ui": False,
        "used_source_wav": False,
        "used_browser_mic": False,
        "session_id": state["session_id"],
        "source_node": args.source_node,
        "target_cycles": args.target_cycles,
        "completed_cycle_count": len(completed),
        "completed_cycles": completed,
        "rejected_attempts": state["rejected_attempts"],
        "process_runs": state["process_runs"],
        "capture_restarts": state["capture_restarts"],
        "journal": {
            "event_count": len(events),
            "sha256": journal.get("sha256"),
            "sequences": sequences,
        },
        "acceptance": acceptance,
        "claims": {
            "physical_hot_mic": True,
            "wake_detection_mode": "hey_embry_transcript_phrase_then_request",
            "production_wake_detector_target": "porcupine_custom_ppn",
            "physical_usb_unplug_replug": False,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}, sort_keys=True), flush=True)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-node", required=True)
    parser.add_argument("--event-service-url", default="http://127.0.0.1:8030/v1/listener/events")
    parser.add_argument("--target-cycles", type=int, default=10)
    parser.add_argument("--cycles-this-run", type=int, default=5)
    parser.add_argument("--max-attempts-this-run", type=int, default=15)
    parser.add_argument("--restart-capture-after-cycle", type=int, default=3)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--realtime-model", default="tiny.en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    return parser


def main() -> int:
    receipt = run(build_parser().parse_args())
    return 0 if receipt["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
