from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_turn_protocol import ManagedTurnServer, send_arm_command, validate_arm_command


def command() -> dict:
    return {"schema": "embry.listener_turn_command.v1", "command": "arm", "campaign_id": "campaign-a", "case_id": "case-a", "attempt_id": "attempt-01", "session_id": "session-a", "turn_id": "case-a:turn-001", "source_authority_id": "source-a", "wake_required": True}


def test_protocol_round_trip(tmp_path: Path) -> None:
    server = ManagedTurnServer(tmp_path / "listener.sock")
    received = []
    thread = threading.Thread(target=lambda: received.append(server.next_command()))
    thread.start()
    ack = send_arm_command(server.path, command())
    thread.join(timeout=2)
    server.close()
    assert ack["armed"] is True and ack["turn_id"] == "case-a:turn-001"
    assert received == [command()]


@pytest.mark.parametrize("field", ["transcript", "expected_transcript", "expected_response", "memory_result", "tau_route"])
def test_protocol_rejects_semantic_shortcuts(field: str) -> None:
    value = command(); value[field] = "forbidden"
    with pytest.raises(ValueError, match="forbidden_semantic_input"):
        validate_arm_command(value)
