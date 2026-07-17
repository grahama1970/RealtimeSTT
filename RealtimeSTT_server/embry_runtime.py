"""Fail-closed Embry RealtimeSTT runtime with native OpenWakeWord authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
from uuid import uuid4

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from RealtimeSTT_server.embry_pcm import PcmIngress
from RealtimeSTT_server.event_outbox import EventOutbox


WAKE_AUTHORITY = "openwakeword_native_callback"
MODEL_ID_DEFAULT = "hey_embry_v1"


class RuntimeStartupError(RuntimeError):
    """Categorized startup failure exposed through readiness."""

    def __init__(self, category: str, detail: str | None = None) -> None:
        super().__init__(detail or category)
        self.category = category
        self.detail = detail or category


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: str) -> str:
    normalized = str(value or "").strip().lower().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise RuntimeStartupError("wake_model_sha256_required")
    return normalized


def _default_recorder_factory(**kwargs: Any) -> Any:
    from RealtimeSTT import AudioToTextRecorder

    return AudioToTextRecorder(**kwargs)


@dataclass(frozen=True)
class RuntimeConfig:
    socket_path: Path
    outbox_path: Path
    journal_url: str
    session_id: str
    correlation_id: str
    wake_model: Path
    wake_model_sha256: str
    wake_model_id: str = MODEL_ID_DEFAULT
    wake_sensitivity: float = 0.6
    wake_timeout_seconds: float = 5.0
    wake_buffer_seconds: float = 0.1
    stt_model: str = "small.en"
    realtime_stt_model: str = "tiny.en"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    runtime_commit: str | None = None
    pcm_bind_timeout_seconds: float = 5.0
    outbox_retry_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        session_id = os.environ.get("EMBRY_SESSION_ID", "embry-container-listener")
        model_value = os.environ.get("EMBRY_OPENWAKEWORD_MODEL", "")
        return cls(
            socket_path=Path(
                os.environ.get(
                    "EMBRY_PCM_SOCKET",
                    "/run/embry/audio/realtimestt-pcm.sock",
                )
            ),
            outbox_path=Path(
                os.environ.get(
                    "EMBRY_EVENT_OUTBOX",
                    "/var/lib/embry-realtimestt/outbox.sqlite3",
                )
            ),
            journal_url=os.environ.get(
                "EMBRY_JOURNAL_URL",
                "http://voice-control:8019/v1/listener/events",
            ),
            session_id=session_id,
            correlation_id=os.environ.get("EMBRY_CORRELATION_ID", session_id),
            wake_model=Path(model_value),
            wake_model_sha256=os.environ.get(
                "EMBRY_OPENWAKEWORD_MODEL_SHA256",
                "",
            ),
            wake_model_id=os.environ.get(
                "EMBRY_OPENWAKEWORD_MODEL_ID",
                MODEL_ID_DEFAULT,
            ),
            wake_sensitivity=float(os.environ.get("EMBRY_WAKE_SENSITIVITY", "0.6")),
            wake_timeout_seconds=float(
                os.environ.get("EMBRY_WAKE_TIMEOUT_SECONDS", "5.0")
            ),
            wake_buffer_seconds=float(
                os.environ.get("EMBRY_WAKE_BUFFER_SECONDS", "0.1")
            ),
            stt_model=os.environ.get("EMBRY_STT_MODEL", "small.en"),
            realtime_stt_model=os.environ.get(
                "EMBRY_REALTIME_STT_MODEL",
                "tiny.en",
            ),
            stt_device=os.environ.get("EMBRY_STT_DEVICE", "cuda"),
            stt_compute_type=os.environ.get(
                "EMBRY_STT_COMPUTE_TYPE",
                "float16",
            ),
            runtime_commit=os.environ.get("EMBRY_REALTIMESTT_COMMIT") or None,
            pcm_bind_timeout_seconds=float(
                os.environ.get("EMBRY_PCM_BIND_TIMEOUT_SECONDS", "5.0")
            ),
            outbox_retry_seconds=float(
                os.environ.get("EMBRY_OUTBOX_RETRY_SECONDS", "1.0")
            ),
        )


class EmbryRuntime:
    """Own recorder, PCM ingress, native-wake turn state, and event delivery."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        recorder_factory: Callable[..., Any] = _default_recorder_factory,
        ingress_factory: Callable[..., Any] = PcmIngress,
        outbox: EventOutbox | None = None,
        journal_deliver: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        turn_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self.recorder_factory = recorder_factory
        self.ingress_factory = ingress_factory
        self.outbox = outbox or EventOutbox(config.outbox_path)
        self.journal_deliver = journal_deliver or self._deliver_http
        self.turn_id_factory = turn_id_factory or self._new_turn_id

        self._state_lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._turn_lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._turn_counter = 0
        self._armed_turn: dict[str, Any] | None = None
        self._active_turn: dict[str, Any] | None = None
        self._recorder: Any = None
        self._ingress: Any = None
        self._ingress_thread: threading.Thread | None = None
        self._transcriber_thread: threading.Thread | None = None
        self._retry_thread: threading.Thread | None = None

        self.state: dict[str, Any] = {
            "alive": True,
            "initialization_started": False,
            "configuration_valid": False,
            "model_verified": False,
            "recorder_ready": False,
            "wake_model_runtime_key": None,
            "pcm_ingress_ready": False,
            "outbox_storage_ready": True,
            "startup_error_category": None,
            "startup_error_detail": None,
            "background_error_category": None,
            "background_error_detail": None,
            "last_delivery_error": self.outbox.last_error(),
            "last_transcript": None,
            "last_event_id": None,
            "last_assigned_sequence": None,
            "ignored_transcript_without_wake_count": 0,
            "duplicate_wake_callback_count": 0,
            "completed_turn_count": 0,
            "wake_detection_started": False,
            "last_wake_timeout_at": None,
        }

    def _new_turn_id(self) -> str:
        with self._turn_lock:
            self._turn_counter += 1
            return f"listener-turn-{self._turn_counter:06d}-{uuid4().hex[:12]}"

    def _set_state(self, **values: Any) -> None:
        with self._state_lock:
            self.state.update(values)

    def _capture_startup_error(self, exc: Exception) -> None:
        category = (
            exc.category
            if isinstance(exc, RuntimeStartupError)
            else f"{type(exc).__name__}"
        )
        self._set_state(
            startup_error_category=category,
            startup_error_detail=str(exc),
            recorder_ready=False,
            pcm_ingress_ready=False,
        )

    def _capture_background_error(self, category: str, exc: Exception) -> None:
        self._set_state(
            background_error_category=category,
            background_error_detail=f"{type(exc).__name__}:{exc}",
        )

    def validate_model(self) -> str:
        path = self.config.wake_model
        if not path.is_absolute():
            raise RuntimeStartupError("wake_model_absolute_path_required")
        if path.suffix.lower() != ".onnx":
            raise RuntimeStartupError("wake_model_onnx_required")
        if not path.is_file():
            raise RuntimeStartupError("wake_model_missing")
        if self.config.wake_model_id != path.stem:
            raise RuntimeStartupError(
                "wake_model_id_path_mismatch",
                f"configured={self.config.wake_model_id}:path_stem={path.stem}",
            )
        expected = _normalize_sha256(self.config.wake_model_sha256)
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeStartupError("wake_model_sha256_mismatch")
        self._set_state(
            configuration_valid=True,
            model_verified=True,
            wake_model_sha256=f"sha256:{actual}",
        )
        return actual

    def _recorder_args(self, model_sha256: str) -> dict[str, Any]:
        return {
            "use_microphone": False,
            "spinner": False,
            "model": self.config.stt_model,
            "realtime_model_type": self.config.realtime_stt_model,
            "language": "en",
            "device": self.config.stt_device,
            "compute_type": self.config.stt_compute_type,
            "enable_realtime_transcription": True,
            "no_log_file": True,
            "wake_words": self.config.wake_model_id,
            "wakeword_backend": "openwakeword",
            "openwakeword_model_paths": str(self.config.wake_model),
            "openwakeword_inference_framework": "onnx",
            "wake_words_sensitivity": self.config.wake_sensitivity,
            "wake_word_timeout": self.config.wake_timeout_seconds,
            "wake_word_buffer_duration": self.config.wake_buffer_seconds,
            "on_wakeword_detection_start": self.on_wakeword_detection_start,
            "on_wakeword_detected": self.on_wakeword_detected,
            "on_wakeword_timeout": self.on_wakeword_timeout,
            "on_recording_start": self.on_recording_start,
            "on_realtime_transcription_update": self.on_partial_transcript,
        }

    def start(self) -> None:
        self._set_state(initialization_started=True)
        try:
            self._initialize()
        except Exception as exc:
            self._capture_startup_error(exc)

    def _initialize(self) -> None:
        model_sha256 = self.validate_model()

        try:
            self._flush_outbox()
        except Exception:
            # The pending events remain durable and readiness stays false. Recorder
            # construction may continue so the failure is observable and retryable.
            pass

        try:
            self._recorder = self.recorder_factory(
                **self._recorder_args(model_sha256)
            )
        except Exception as exc:
            raise RuntimeStartupError(
                "recorder_construction_failed",
                f"{type(exc).__name__}:{exc}",
            ) from exc
        model_keys = sorted(
            str(key)
            for key in getattr(self._recorder.owwModel, "models", {}).keys()
        )
        if model_keys != [self.config.wake_model_id]:
            raise RuntimeStartupError(
                "wake_model_runtime_key_mismatch",
                f"expected={self.config.wake_model_id}:loaded={','.join(model_keys)}",
            )
        self._set_state(wake_model_runtime_key=self.config.wake_model_id)
        self._set_state(recorder_ready=True)

        try:
            self._ingress = self.ingress_factory(
                self.config.socket_path,
                lambda pcm: self._recorder.feed_audio(
                    pcm,
                    original_sample_rate=16000,
                ),
                on_first_frame=self.on_first_pcm_frame,
            )
        except Exception as exc:
            raise RuntimeStartupError(
                "pcm_ingress_construction_failed",
                f"{type(exc).__name__}:{exc}",
            ) from exc

        self._ingress_thread = threading.Thread(
            target=self._run_ingress,
            daemon=True,
            name="embry-pcm-ingress",
        )
        self._ingress_thread.start()
        if not self._ingress.wait_until_bound(
            self.config.pcm_bind_timeout_seconds
        ):
            detail = getattr(self._ingress, "last_error", None)
            raise RuntimeStartupError(
                "pcm_ingress_bind_failed",
                str(detail or "pcm_ingress_bind_timeout"),
            )
        self._set_state(pcm_ingress_ready=True)

        self._transcriber_thread = threading.Thread(
            target=self._run_transcriber,
            daemon=True,
            name="embry-transcriber",
        )
        self._transcriber_thread.start()
        self._retry_thread = threading.Thread(
            target=self._run_outbox_retry,
            daemon=True,
            name="embry-event-outbox-retry",
        )
        self._retry_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._ingress is not None and hasattr(self._ingress, "stop"):
            self._ingress.stop()
        if self._recorder is not None:
            try:
                self._recorder.shutdown()
            except Exception:
                pass
        for thread in (
            self._ingress_thread,
            self._transcriber_thread,
            self._retry_thread,
        ):
            if thread is not None:
                thread.join(timeout=1.0)
        self._set_state(
            recorder_ready=False,
            pcm_ingress_ready=False,
            alive=False,
        )

    def _run_ingress(self) -> None:
        try:
            self._ingress.serve_one()
            if not self._stop_event.is_set():
                raise RuntimeError("pcm_ingress_stopped")
        except Exception as exc:
            if not self._stop_event.is_set():
                self._capture_background_error("pcm_ingress_failed", exc)
        finally:
            self._set_state(pcm_ingress_ready=False)

    def _run_transcriber(self) -> None:
        try:
            while not self._stop_event.is_set():
                text = self._recorder.text()
                if text and str(text).strip():
                    self.accept_final_transcript(str(text))
                else:
                    time.sleep(0.02)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._capture_background_error("transcriber_failed", exc)

    def _run_outbox_retry(self) -> None:
        while not self._stop_event.wait(self.config.outbox_retry_seconds):
            if self.outbox.pending_count() <= 0:
                continue
            try:
                self._flush_outbox()
            except Exception:
                continue

    def _deliver_http(self, event: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            self.config.journal_url,
            json=event,
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("journal_response_not_object")
        return payload

    def _flush_outbox(self) -> int:
        with self._flush_lock:
            try:
                delivered = self.outbox.flush(self.journal_deliver)
            except Exception as exc:
                self._set_state(last_delivery_error=f"{type(exc).__name__}:{exc}")
                raise
            latest = self.outbox.latest_delivered()
            self._set_state(
                last_delivery_error=self.outbox.last_error(),
                last_event_id=(latest or {}).get("event_id")
                or self.state.get("last_event_id"),
                last_assigned_sequence=(latest or {}).get("assigned_sequence")
                or self.state.get("last_assigned_sequence"),
            )
            return delivered

    def make_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        turn_id: str,
        causation_id: str | None,
        producer: str = "realtimestt.container",
    ) -> dict[str, Any]:
        created_at = _utc_now()
        identity = {
            "session_id": self.config.session_id,
            "turn_id": turn_id,
            "type": event_type,
            "created_at": created_at,
            "payload": payload,
        }
        event_id = (
            f"{event_type}."
            f"{hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16]}"
        )
        event: dict[str, Any] = {
            "schema": "embry.voice_event.v2",
            "event_id": event_id,
            "session_id": self.config.session_id,
            "turn_id": turn_id,
            "type": event_type,
            "created_at": created_at,
            "causation_id": causation_id or event_id,
            "correlation_id": self.config.correlation_id,
            "producer": producer,
            "mocked": False,
            "live": True,
            "artifact_hashes": payload.get("artifact_hashes", {}),
            "payload": payload,
        }
        event["receipt_hash"] = "sha256:" + hashlib.sha256(
            _canonical_bytes(event)
        ).hexdigest()
        if "sequence" in event:
            raise AssertionError("producer_sequence_must_not_exist")
        return event

    def publish_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        turn_id: str,
        causation_id: str | None,
        producer: str = "realtimestt.container",
    ) -> dict[str, Any]:
        with self._event_lock:
            event = self.make_event(
                event_type,
                payload,
                turn_id=turn_id,
                causation_id=causation_id,
                producer=producer,
            )
            self.outbox.put(event)
            self._set_state(last_event_id=event["event_id"])
            try:
                self._flush_outbox()
            except Exception:
                # The exact event remains pending and readiness becomes false.
                pass
            return event

    def on_wakeword_detection_start(self) -> None:
        self._set_state(wake_detection_started=True)

    def on_wakeword_timeout(self) -> None:
        with self._turn_lock:
            self._active_turn = None
            self._set_state(last_wake_timeout_at=_utc_now())

    def on_wakeword_detected(self) -> dict[str, Any]:
        """The only method allowed to create listener.wake_detected."""
        with self._turn_lock:
            if self._active_turn is not None:
                self._set_state(
                    duplicate_wake_callback_count=(
                        int(self.state["duplicate_wake_callback_count"]) + 1
                    )
                )
                return self._active_turn["wake_event"]
            armed_turn = self._armed_turn
            turn_id = (
                armed_turn["turn_id"]
                if armed_turn is not None
                else self.turn_id_factory()
            )
            payload = {
                "wake_phrase": "Hey Embry",
                "wake_authority": WAKE_AUTHORITY,
                "native_callback": True,
                "asr_text_used_for_wake": False,
                "substring_matcher_used": False,
                "wake_model_id": self.config.wake_model_id,
                "wake_model_sha256": self.state.get("wake_model_sha256"),
                "wake_sensitivity": self.config.wake_sensitivity,
            }
            event = self.publish_event(
                "listener.wake_detected",
                payload,
                turn_id=turn_id,
                causation_id=(
                    armed_turn["last_event_id"] if armed_turn is not None else None
                ),
                producer="realtimestt.openwakeword",
            )
            self._armed_turn = None
            self._active_turn = {
                "turn_id": turn_id,
                "wake_event": event,
                "wake_event_id": event["event_id"],
                "last_event_id": event["event_id"],
            }
            return event

    def _active_turn_snapshot(self) -> dict[str, Any] | None:
        with self._turn_lock:
            return dict(self._active_turn) if self._active_turn else None

    def _advance_turn_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        with self._turn_lock:
            if self._active_turn is None:
                return None
            event = self.publish_event(
                event_type,
                payload,
                turn_id=self._active_turn["turn_id"],
                causation_id=self._active_turn["last_event_id"],
            )
            self._active_turn["last_event_id"] = event["event_id"]
            return event

    def on_recording_start(self) -> dict[str, Any] | None:
        return self._advance_turn_event(
            "listener.audio_turn_started",
            {"wake_authority": WAKE_AUTHORITY},
        )

    def on_partial_transcript(self, text: str) -> dict[str, Any] | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        return self._advance_turn_event(
            "listener.partial_transcript",
            {"text": normalized},
        )

    def on_first_pcm_frame(self, header: dict[str, Any]) -> None:
        self._set_state(
            pcm_source_node=header.get("source_node"),
            pcm_stream_id=header.get("stream_id"),
        )
        with self._turn_lock:
            if self._armed_turn is not None or self._active_turn is not None:
                return
            turn_id = self.turn_id_factory()
            device_event = self.publish_event(
                "listener.device_selected",
                {
                    "source_node": header.get("source_node"),
                    "stream_id": header.get("stream_id"),
                    "sample_rate_hz": header.get("sample_rate_hz"),
                    "channels": header.get("channels"),
                    "sample_format": header.get("format"),
                },
                turn_id=turn_id,
                causation_id=None,
            )
            armed_event = self.publish_event(
                "listener.wake_armed",
                {
                    "wake_phrase": "Hey Embry",
                    "wake_model_id": self.config.wake_model_id,
                    "wake_model_sha256": self.state.get("wake_model_sha256"),
                    "wake_sensitivity": self.config.wake_sensitivity,
                    "native_callback_required": True,
                },
                turn_id=turn_id,
                causation_id=device_event["event_id"],
            )
            self._armed_turn = {
                "turn_id": turn_id,
                "last_event_id": armed_event["event_id"],
            }

    def accept_final_transcript(self, text: str) -> dict[str, Any] | None:
        """Publish a final transcript only after a native wake callback."""
        normalized = re.sub(
            r"^\s*hey[\s,]+(?:embry|embree)\b[\s,.:;!?-]*",
            "",
            str(text or ""),
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if not normalized:
            return None
        with self._turn_lock:
            if self._active_turn is None:
                self._set_state(
                    ignored_transcript_without_wake_count=(
                        int(self.state["ignored_transcript_without_wake_count"]) + 1
                    )
                )
                return None
            event = self.publish_event(
                "listener.final_transcript",
                {
                    "text": normalized,
                    "wake_event_id": self._active_turn["wake_event_id"],
                    "wake_authority": WAKE_AUTHORITY,
                    "asr_text_used_for_wake": False,
                },
                turn_id=self._active_turn["turn_id"],
                causation_id=self._active_turn["last_event_id"],
            )
            receipt_event = self.publish_event(
                "listener.receipt_written",
                {
                    "final_transcript_event_id": event["event_id"],
                    "event_schema": event["schema"],
                    "source_node": self.state.get("pcm_source_node"),
                    "stream_id": self.state.get("pcm_stream_id"),
                },
                turn_id=self._active_turn["turn_id"],
                causation_id=event["event_id"],
            )
            self._active_turn["last_event_id"] = receipt_event["event_id"]
            self._active_turn = None
            self._set_state(
                last_transcript=normalized,
                completed_turn_count=int(self.state["completed_turn_count"]) + 1,
            )
            return event

    def snapshot(self) -> dict[str, Any]:
        ingress_snapshot = (
            self._ingress.snapshot()
            if self._ingress is not None and hasattr(self._ingress, "snapshot")
            else {
                "socket_path": str(self.config.socket_path),
                "bound": False,
                "connected": False,
                "frame_count": 0,
                "last_sequence": 0,
                "gap_count": 0,
                "sample_gap_count": 0,
                "last_error": None,
            }
        )
        outbox_snapshot = self.outbox.snapshot()
        with self._state_lock:
            current = dict(self.state)
        current["last_delivery_error"] = (
            current.get("last_delivery_error") or outbox_snapshot["last_error"]
        )
        recorder = self._recorder
        score_lock = getattr(recorder, "oww_score_lock", None)
        if score_lock is None:
            score_values = ({}, {}, 0)
        else:
            with score_lock:
                score_values = (
                    dict(getattr(recorder, "oww_last_scores", {})),
                    dict(getattr(recorder, "oww_rolling_max_scores", {})),
                    int(getattr(recorder, "oww_inference_count", 0)),
                )
        wakeword_scores = {
            "frame_samples": 1280,
            "pending_samples": int(getattr(recorder, "oww_pending_samples", 0)),
            "inference_count": score_values[2],
            "model_key": current.get("wake_model_runtime_key"),
            "configured_sensitivity": self.config.wake_sensitivity,
            "last": score_values[0],
            "rolling_max": score_values[1],
        }
        ready = bool(
            current["configuration_valid"]
            and current["model_verified"]
            and current["recorder_ready"]
            and current["pcm_ingress_ready"]
            and ingress_snapshot.get("bound")
            and current["startup_error_category"] is None
            and current["background_error_category"] is None
            and outbox_snapshot["pending_count"] == 0
            and current["last_delivery_error"] is None
        )
        return {
            "schema": "embry.realtimestt_runtime_status.v1",
            "alive": bool(current["alive"]),
            "ready": ready,
            "wake_authority": WAKE_AUTHORITY,
            "asr_text_used_for_wake": False,
            "transcript_alias_matcher_enabled": False,
            "wake_model": {
                "id": self.config.wake_model_id,
                "path": str(self.config.wake_model),
                "sha256": current.get("wake_model_sha256"),
                "hash_verified": bool(current["model_verified"]),
                "inference_framework": "onnx",
                "runtime_key": current.get("wake_model_runtime_key"),
            },
            "wakeword_scores": wakeword_scores,
            "runtime_commit": self.config.runtime_commit,
            "session_id": self.config.session_id,
            "correlation_id": self.config.correlation_id,
            "startup_error": {
                "category": current.get("startup_error_category"),
                "detail": current.get("startup_error_detail"),
            },
            "background_error": {
                "category": current.get("background_error_category"),
                "detail": current.get("background_error_detail"),
            },
            "outbox": outbox_snapshot,
            "pcm": ingress_snapshot,
            "state": current,
            "active_turn": self._active_turn_snapshot(),
        }


app = FastAPI(title="Embry RealtimeSTT Runtime")
runtime: EmbryRuntime | None = None
boot_error: dict[str, str | None] | None = None


def _boot_failure_snapshot() -> dict[str, Any]:
    return {
        "schema": "embry.realtimestt_runtime_status.v1",
        "alive": True,
        "ready": False,
        "wake_authority": WAKE_AUTHORITY,
        "asr_text_used_for_wake": False,
        "transcript_alias_matcher_enabled": False,
        "wake_model": None,
        "runtime_commit": os.environ.get("EMBRY_REALTIMESTT_COMMIT") or None,
        "startup_error": boot_error
        or {"category": "runtime_not_started", "detail": "runtime_not_started"},
        "background_error": {"category": None, "detail": None},
        "outbox": {"pending_count": None, "last_error": None},
        "pcm": {"bound": False, "frame_count": 0, "gap_count": 0, "sample_gap_count": 0},
        "state": {},
        "active_turn": None,
    }


@app.on_event("startup")
def startup() -> None:
    global runtime, boot_error
    boot_error = None
    try:
        runtime = EmbryRuntime(RuntimeConfig.from_env())
        runtime.start()
    except Exception as exc:
        runtime = None
        boot_error = {
            "category": f"{type(exc).__name__}",
            "detail": str(exc),
        }


@app.on_event("shutdown")
def shutdown() -> None:
    if runtime is not None:
        runtime.shutdown()


@app.get("/health")
def health() -> dict[str, Any]:
    payload = runtime.snapshot() if runtime is not None else _boot_failure_snapshot()
    return {
        **payload,
        "schema": "embry.realtimestt_container_health.v2",
        "ok": True,
        "alive": True,
    }


@app.get("/readiness")
def readiness() -> JSONResponse:
    payload = runtime.snapshot() if runtime is not None else _boot_failure_snapshot()
    status_code = 200 if payload.get("ready") is True else 503
    return JSONResponse(
        status_code=status_code,
        content={
            **payload,
            "schema": "embry.realtimestt_container_readiness.v2",
        },
    )
