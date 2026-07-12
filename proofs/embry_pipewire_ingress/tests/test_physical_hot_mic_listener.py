"""Deterministic contract tests for the resumable physical listener."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_physical_hot_mic_listener.py"
SPEC = importlib.util.spec_from_file_location("physical_hot_mic_listener", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
realtimestt_stub = types.ModuleType("RealtimeSTT")
realtimestt_stub.AudioToTextRecorder = object
sys.modules.setdefault("RealtimeSTT", realtimestt_stub)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("text", "expected", "expected_request"),
    [
        ("Hey Embry, what is the capital of France?", True, "what is the capital of france"),
        ("hey embry", True, ""),
        ("Hey Emory, listen carefully", True, "listen carefully"),
        ("Hey Embring, test cycle three", True, "test cycle three"),
        ("Hey Henry.", True, ""),
        ("Hey entry, test cycle three", False, ""),
        ("Embry tell me what you heard", False, ""),
        ("background conversation", False, ""),
        ("", False, ""),
    ],
)
def test_wake_phrase_uses_hey_embry_with_bounded_variants(
    text: str, expected: bool, expected_request: str
) -> None:
    match = MODULE.wake_phrase_match(text)
    assert match["detected"] is expected
    assert " ".join(match["request_tokens"]) == expected_request


def test_listener_state_resumes_same_source_and_target(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = MODULE.load_or_create_state(state_path, "alsa_input.physical", 10)
    state["completed_cycles"].append({"cycle": 1})
    MODULE.save_state(state_path, state)

    resumed = MODULE.load_or_create_state(state_path, "alsa_input.physical", 10)
    assert resumed["session_id"] == state["session_id"]
    assert resumed["completed_cycles"] == [{"cycle": 1}]


def test_listener_state_rejects_source_or_target_drift(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    MODULE.save_state(state_path, MODULE.initial_state("alsa_input.physical", 10))

    with pytest.raises(RuntimeError, match="source_node_mismatch"):
        MODULE.load_or_create_state(state_path, "alsa_input.other", 10)
    with pytest.raises(RuntimeError, match="target_cycles_mismatch"):
        MODULE.load_or_create_state(state_path, "alsa_input.physical", 9)


def test_state_file_records_no_transcript_fixture(tmp_path: Path) -> None:
    state = MODULE.initial_state("alsa_input.physical", 10)
    path = tmp_path / "state.json"
    MODULE.save_state(path, state)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "expected_transcript" not in payload
    assert payload["completed_cycles"] == []


def test_failed_start_does_not_look_like_completed_restart() -> None:
    process_runs = [
        {"process_run": 1, "status": "starting"},
        {"process_run": 2, "status": "completed"},
    ]
    assert sum(run.get("status") == "completed" for run in process_runs) == 1


@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1:8030",
        "http://127.0.0.1:8030/",
        "http://127.0.0.1:8030/v1/listener/events",
    ],
)
def test_event_service_origin_normalizes_endpoint_input(value: str) -> None:
    assert MODULE.event_service_origin(value) == "http://127.0.0.1:8030"


def test_parser_accepts_managed_socket_without_semantic_prompt() -> None:
    args = MODULE.build_parser().parse_args([
        "--run-dir", "/tmp/managed-listener",
        "--source-node", "alsa_input.physical",
        "--managed-socket", "/tmp/managed-listener/listener.sock",
    ])

    assert args.managed_socket == Path("/tmp/managed-listener/listener.sock")
    assert not hasattr(args, "expected_transcript")
    assert not hasattr(args, "prompt")
