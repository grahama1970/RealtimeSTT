from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from RealtimeSTT_server.embry_runtime import EmbryRuntime, RuntimeConfig


class FakeRecorder:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.owwModel = SimpleNamespace(models={kwargs["wake_words"]: object()})
        self.stop_event = threading.Event()
        self.feed_calls: list[tuple[bytes, int]] = []

    def feed_audio(self, pcm: bytes, *, original_sample_rate: int) -> None:
        self.feed_calls.append((pcm, original_sample_rate))

    def text(self) -> str:
        self.stop_event.wait(0.03)
        return ""

    def shutdown(self) -> None:
        self.stop_event.set()


class FakeIngress:
    def __init__(self, _path: Path, _on_pcm: Any, *, on_first_frame: Any = None) -> None:
        self.bound_event = threading.Event()
        self.stop_event = threading.Event()
        self.bound = False
        self.connected = False
        self.frame_count = 0
        self.last_sequence = 0
        self.gap_count = 0
        self.sample_gap_count = 0
        self.last_error = None
        self.on_first_frame = on_first_frame

    def serve_one(self) -> None:
        self.bound = True
        self.bound_event.set()
        self.stop_event.wait()
        self.bound = False

    def wait_until_bound(self, timeout: float) -> bool:
        return self.bound_event.wait(timeout)

    def snapshot(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "connected": self.connected,
            "frame_count": self.frame_count,
            "last_sequence": self.last_sequence,
            "gap_count": self.gap_count,
            "sample_gap_count": self.sample_gap_count,
            "last_error": self.last_error,
        }

    def stop(self) -> None:
        self.stop_event.set()


class FailingIngress(FakeIngress):
    def serve_one(self) -> None:
        self.bound = True
        self.bound_event.set()
        raise RuntimeError("background_ingress_boom")


def config(tmp_path: Path, model: Path, sha256: str) -> RuntimeConfig:
    return RuntimeConfig(
        socket_path=tmp_path / "pcm.sock",
        outbox_path=tmp_path / "outbox.sqlite3",
        journal_url="http://journal.invalid/events",
        session_id="session-a",
        correlation_id="correlation-a",
        wake_model=model,
        wake_model_sha256=sha256,
        runtime_commit="3a86b6fe96002bf7de06533aa5cdd9cb90ed18c9",
        pcm_bind_timeout_seconds=1.0,
        outbox_retry_seconds=0.05,
    )


def model_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "hey_embry_v1.onnx"
    path.write_bytes(b"test-onnx-boundary")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def recorder_boundary(storage: list[FakeRecorder]):
    def build(**kwargs: Any) -> FakeRecorder:
        recorder = FakeRecorder(**kwargs)
        storage.append(recorder)
        return recorder

    return build


def journal_boundary(storage: list[dict[str, Any]]):
    sequence = 0

    def deliver(event: dict[str, Any]) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        storage.append(event)
        return {"event": {**event, "sequence": sequence}}

    return deliver


def start_valid_runtime(tmp_path: Path, *, ingress_factory=FakeIngress):
    model, digest = model_file(tmp_path)
    recorders: list[FakeRecorder] = []
    delivered: list[dict[str, Any]] = []
    runtime = EmbryRuntime(
        config(tmp_path, model, digest),
        recorder_factory=recorder_boundary(recorders),
        ingress_factory=ingress_factory,
        journal_deliver=journal_boundary(delivered),
        turn_id_factory=iter(["turn-001", "turn-002", "turn-003"]).__next__,
    )
    runtime.start()
    return runtime, recorders, delivered


def test_missing_model_path_fails_readiness(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    runtime = EmbryRuntime(
        config(tmp_path, missing, "a" * 64),
        recorder_factory=lambda **_kwargs: pytest.fail("recorder must not construct"),
        ingress_factory=FakeIngress,
    )
    runtime.start()
    snapshot = runtime.snapshot()
    assert snapshot["ready"] is False
    assert snapshot["startup_error"]["category"] == "wake_model_missing"


def test_relative_model_path_is_rejected(tmp_path: Path) -> None:
    runtime = EmbryRuntime(
        config(tmp_path, Path("relative.onnx"), "a" * 64),
        recorder_factory=lambda **_kwargs: pytest.fail("recorder must not construct"),
        ingress_factory=FakeIngress,
    )
    runtime.start()
    assert runtime.snapshot()["startup_error"]["category"] == "wake_model_absolute_path_required"


def test_missing_model_hash_fails_readiness(tmp_path: Path) -> None:
    model, _digest = model_file(tmp_path)
    runtime = EmbryRuntime(
        config(tmp_path, model, ""),
        recorder_factory=lambda **_kwargs: pytest.fail("recorder must not construct"),
        ingress_factory=FakeIngress,
    )
    runtime.start()
    assert runtime.snapshot()["startup_error"]["category"] == "wake_model_sha256_required"


def test_model_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    model, _digest = model_file(tmp_path)
    runtime = EmbryRuntime(
        config(tmp_path, model, "b" * 64),
        recorder_factory=lambda **_kwargs: pytest.fail("recorder must not construct"),
        ingress_factory=FakeIngress,
    )
    runtime.start()
    assert runtime.snapshot()["startup_error"]["category"] == "wake_model_sha256_mismatch"


def test_valid_model_configuration_reaches_recorder_construction(tmp_path: Path) -> None:
    runtime, recorders, _delivered = start_valid_runtime(tmp_path)
    try:
        assert len(recorders) == 1
        kwargs = recorders[0].kwargs
        assert kwargs["use_microphone"] is False
        assert kwargs["wakeword_backend"] == "openwakeword"
        assert kwargs["openwakeword_inference_framework"] == "onnx"
        assert Path(kwargs["openwakeword_model_paths"]).is_absolute()
        snapshot = runtime.snapshot()
        assert snapshot["ready"] is True
        assert snapshot["wake_authority"] == "openwakeword_native_callback"
        assert snapshot["asr_text_used_for_wake"] is False
    finally:
        runtime.shutdown()


def test_native_callback_creates_one_wake_and_fresh_turn(tmp_path: Path) -> None:
    runtime, recorders, delivered = start_valid_runtime(tmp_path)
    try:
        callback = recorders[0].kwargs["on_wakeword_detected"]
        first = callback()
        duplicate = callback()
        assert duplicate["event_id"] == first["event_id"]
        assert runtime.accept_final_transcript("What evidence should a QRA include?") is not None
        second = callback()

        wakes = [event for event in delivered if event["type"] == "listener.wake_detected"]
        assert len(wakes) == 2
        assert wakes[0]["turn_id"] == "turn-001"
        assert wakes[1]["turn_id"] == "turn-002"
        assert wakes[0]["event_id"] != wakes[1]["event_id"]
        assert first["producer"] == "realtimestt.openwakeword"
        assert first["payload"]["native_callback"] is True
        assert all("sequence" not in event for event in delivered)
    finally:
        runtime.shutdown()


def test_wake_timeout_releases_turn_for_next_native_wake(tmp_path: Path) -> None:
    runtime, recorders, delivered = start_valid_runtime(tmp_path)
    try:
        wake = recorders[0].kwargs["on_wakeword_detected"]
        timeout = recorders[0].kwargs["on_wakeword_timeout"]

        first = wake()
        timeout()
        second = wake()

        assert first["turn_id"] == "turn-001"
        assert second["turn_id"] == "turn-002"
        assert first["event_id"] != second["event_id"]
        assert runtime.snapshot()["active_turn"]["turn_id"] == "turn-002"
        assert runtime.snapshot()["state"]["duplicate_wake_callback_count"] == 0
        assert [event["type"] for event in delivered] == [
            "listener.wake_detected",
            "listener.wake_detected",
        ]
    finally:
        runtime.shutdown()


def test_transcript_text_cannot_promote_wake_without_callback(tmp_path: Path) -> None:
    runtime, _recorders, delivered = start_valid_runtime(tmp_path)
    try:
        result = runtime.accept_final_transcript("Hey Embry what is the capital of France?")
        assert result is None
        assert delivered == []
        snapshot = runtime.snapshot()
        assert snapshot["state"]["ignored_transcript_without_wake_count"] == 1
    finally:
        runtime.shutdown()


def test_journal_assigned_sequence_is_exposed_in_runtime_state(tmp_path: Path) -> None:
    runtime, recorders, _delivered = start_valid_runtime(tmp_path)
    try:
        recorders[0].kwargs["on_wakeword_detected"]()
        snapshot = runtime.snapshot()
        assert snapshot["state"]["last_assigned_sequence"] == 1
        assert snapshot["outbox"]["pending_count"] == 0
    finally:
        runtime.shutdown()


def test_background_startup_exception_appears_in_readiness(tmp_path: Path) -> None:
    runtime, _recorders, _delivered = start_valid_runtime(
        tmp_path,
        ingress_factory=FailingIngress,
    )
    try:
        deadline = time.time() + 2
        while time.time() < deadline:
            snapshot = runtime.snapshot()
            if snapshot["background_error"]["category"]:
                break
            time.sleep(0.01)
        assert snapshot["ready"] is False
        assert snapshot["background_error"]["category"] == "pcm_ingress_failed"
        assert "background_ingress_boom" in snapshot["background_error"]["detail"]
    finally:
        runtime.shutdown()


def test_container_healthcheck_targets_readiness() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1]
        / "docker"
        / "Dockerfile.embry-runtime"
    ).read_text(encoding="utf-8")
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8020/readiness" in dockerfile
    assert "http://127.0.0.1:8020/health" not in dockerfile


def test_health_and_readiness_have_distinct_http_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import RealtimeSTT_server.embry_runtime as module

    class SnapshotRuntime:
        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def snapshot(self) -> dict[str, Any]:
            return {
                "schema": "embry.realtimestt_runtime_status.v1",
                "alive": True,
                "ready": self.ready,
            }

    monkeypatch.setattr(module, "runtime", SnapshotRuntime(False))
    health_payload = module.health()
    readiness_response = module.readiness()
    assert health_payload["schema"] == "embry.realtimestt_container_health.v2"
    assert health_payload["ok"] is True
    assert readiness_response.status_code == 503
    assert json.loads(readiness_response.body)["schema"] == "embry.realtimestt_container_readiness.v2"

    monkeypatch.setattr(module, "runtime", SnapshotRuntime(True))
    assert module.readiness().status_code == 200
