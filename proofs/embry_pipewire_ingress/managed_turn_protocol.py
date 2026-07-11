"""Unix-socket control protocol for arming one managed Embry listener turn."""

from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any

SCHEMA = "embry.listener_turn_command.v1"
REQUIRED = ("campaign_id", "case_id", "attempt_id", "session_id", "turn_id", "source_authority_id")
FORBIDDEN = ("transcript", "expected_transcript", "expected_response", "memory_result", "tau_route")


def validate_arm_command(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA or value.get("command") != "arm":
        raise ValueError("managed_turn_command_invalid")
    for key in REQUIRED:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"managed_turn_{key}_missing")
    if not isinstance(value.get("wake_required"), bool):
        raise ValueError("managed_turn_wake_required_invalid")
    if any(key in value for key in FORBIDDEN):
        raise ValueError("managed_turn_forbidden_semantic_input")
    return value


class ManagedTurnServer:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.path))
        self.socket.listen(4)

    def next_command(self) -> dict[str, Any]:
        connection, _ = self.socket.accept()
        with connection:
            raw = connection.makefile("rb").readline()
            command = validate_arm_command(json.loads(raw))
            ack = {"schema": "embry.listener_turn_arm_ack.v1", "armed": True, "session_id": command["session_id"], "turn_id": command["turn_id"]}
            connection.sendall(json.dumps(ack).encode() + b"\n")
            return command

    def close(self) -> None:
        self.socket.close()
        self.path.unlink(missing_ok=True)


def send_arm_command(path: Path, command: dict[str, Any]) -> dict[str, Any]:
    validate_arm_command(command)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(json.dumps(command, sort_keys=True).encode() + b"\n")
        return json.loads(client.makefile("rb").readline())
