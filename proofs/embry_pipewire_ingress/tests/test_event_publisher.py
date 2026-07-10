"""Focused contract tests for journal-assigned RealtimeSTT event identity."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import httpx
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_pipewire_realtimestt_ingress.py"
SPEC = importlib.util.spec_from_file_location("pipewire_ingress_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
realtimestt_stub = types.ModuleType("RealtimeSTT")
realtimestt_stub.AudioToTextRecorder = object
sys.modules.setdefault("RealtimeSTT", realtimestt_stub)
SPEC.loader.exec_module(MODULE)


class RecordingClient:
    """Small HTTP client double used only to inspect the producer request contract."""

    response_payload: dict[str, Any] = {}
    posted: list[dict[str, Any]] = []

    def __init__(self, **_: Any) -> None:
        pass

    def post(self, _: str, json: dict[str, Any]) -> httpx.Response:
        self.posted.append(json)
        payload = dict(self.response_payload)
        if payload.get("event_id") == "FROM_REQUEST":
            payload["event_id"] = json["event_id"]
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://journal/v1/listener/events"))


@pytest.fixture(autouse=True)
def reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingClient.posted = []
    RecordingClient.response_payload = {
        "schema": "embry.listener_event_ingest_receipt.v1",
        "accepted": True,
        "event_id": "FROM_REQUEST",
        "sequence": 7,
    }
    monkeypatch.setattr(MODULE.httpx, "Client", RecordingClient)


def publisher() -> tuple[Any, dict[str, Any]]:
    return MODULE.build_event_publisher(
        service_url="http://journal",
        session_id="session-a",
        turn_id="turn-a",
        run_id="run-a",
    )


def test_publisher_omits_sequence_and_records_server_assignment() -> None:
    publish, delivery = publisher()
    stored = publish("listener.ready", {"listener_authority": "unix_pipewire_realtimestt"})

    request = RecordingClient.posted[0]
    assert "sequence" not in request
    assert request["mocked"] is False
    assert request["live"] is True
    assert request["producer"] == "RealtimeSTT.pipewire_ingress"
    assert stored["assigned_sequence"] == 7
    assert delivery["assigned_events"] == [{"event_id": request["event_id"], "sequence": 7}]


def test_publisher_chains_causation_to_previous_accepted_event() -> None:
    publish, _ = publisher()
    first = publish("listener.ready", {})
    publish("listener.audio_started", {})

    assert RecordingClient.posted[1]["causation_id"] == first["event_id"]


@pytest.mark.parametrize("response", [{"event_id": "FROM_REQUEST"}, {"event_id": "wrong", "sequence": 1}])
def test_publisher_fails_closed_on_invalid_journal_identity(response: dict[str, Any]) -> None:
    RecordingClient.response_payload = response
    publish, delivery = publisher()

    with pytest.raises(RuntimeError, match="journal_event_publish_failed"):
        publish("listener.ready", {})
    assert delivery["accepted"] == 0
    assert len(delivery["errors"]) == 1
