from __future__ import annotations

from pathlib import Path

import pytest

from RealtimeSTT_server.event_outbox import EventOutbox, OutboxDeliveryError


def event(event_id: str = "listener.wake_detected.abc") -> dict:
    return {
        "schema": "embry.voice_event.v2",
        "event_id": event_id,
        "session_id": "session-a",
        "turn_id": "turn-a",
        "type": "listener.wake_detected",
        "created_at": "2026-07-15T12:00:00+00:00",
        "causation_id": event_id,
        "correlation_id": "session-a",
        "producer": "realtimestt.openwakeword",
        "mocked": False,
        "live": True,
        "artifact_hashes": {},
        "receipt_hash": "sha256:" + "a" * 64,
        "payload": {"native_callback": True},
    }


def stored_response(value: dict, sequence: int = 7) -> dict:
    return {"event": {**value, "sequence": sequence}}


def test_journal_response_sequence_is_stored_durably(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path / "outbox.sqlite3")
    value = event()
    outbox.put(value)

    assert outbox.flush(lambda original: stored_response(original, 19)) == 1
    record = outbox.get(value["event_id"])
    assert record is not None
    assert record["delivered"] is True
    assert record["assigned_sequence"] == 19
    assert record["response"]["event"]["sequence"] == 19
    assert outbox.pending_count() == 0


def test_missing_journal_sequence_fails_closed_and_remains_pending(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path / "outbox.sqlite3")
    value = event()
    outbox.put(value)

    with pytest.raises(OutboxDeliveryError, match="sequence_missing"):
        outbox.flush(lambda original: {"event": dict(original)})

    record = outbox.get(value["event_id"])
    assert record is not None
    assert record["delivered"] is False
    assert record["assigned_sequence"] is None
    assert "sequence_missing" in record["last_error"]
    assert outbox.pending_count() == 1


def test_conflicting_journal_event_fails_closed_and_remains_pending(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path / "outbox.sqlite3")
    value = event()
    outbox.put(value)

    def conflicting(original: dict) -> dict:
        stored = {**original, "sequence": 3, "turn_id": "wrong-turn"}
        return {"event": stored}

    with pytest.raises(OutboxDeliveryError, match="event_conflict:turn_id"):
        outbox.flush(conflicting)

    record = outbox.get(value["event_id"])
    assert record is not None
    assert record["delivered"] is False
    assert outbox.pending_count() == 1


def test_exact_replay_is_idempotent_and_not_redelivered(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path / "outbox.sqlite3")
    value = event()
    calls: list[str] = []

    def deliver(original: dict) -> dict:
        calls.append(original["event_id"])
        return stored_response(original, 5)

    outbox.put(value)
    assert outbox.flush(deliver) == 1
    outbox.put(value)
    assert outbox.flush(deliver) == 0
    assert calls == [value["event_id"]]


def test_producer_assigned_sequence_is_rejected(tmp_path: Path) -> None:
    outbox = EventOutbox(tmp_path / "outbox.sqlite3")
    value = {**event(), "sequence": 1}
    with pytest.raises(ValueError, match="producer_sequence_forbidden"):
        outbox.put(value)
