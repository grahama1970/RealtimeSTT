"""Unix framed-PCM ingress for the Embry RealtimeSTT runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import threading
import zlib
from typing import Any, BinaryIO, Callable


HEADER_LENGTH = struct.Struct("!I")
MAX_HEADER_BYTES = 16 * 1024
PCM_BYTES = 1024
FRAME_SAMPLES = 512


def canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_exact(stream: BinaryIO, size: int) -> bytes:
    if not isinstance(size, int) or size < 0:
        raise ValueError("pcm_payload_length_invalid")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("pcm_stream_ended")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    header_size = HEADER_LENGTH.unpack(read_exact(stream, HEADER_LENGTH.size))[0]
    if header_size < 2 or header_size > MAX_HEADER_BYTES:
        raise ValueError("pcm_header_length_invalid")
    try:
        header = json.loads(read_exact(stream, header_size))
    except json.JSONDecodeError as exc:
        raise ValueError("pcm_header_json_invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("pcm_header_invalid")
    expected = {
        "schema": "embry.pcm_frame.v1",
        "format": "s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
        "samples": FRAME_SAMPLES,
        "payload_bytes": PCM_BYTES,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"pcm_{key}_invalid")
    for key in ("source_node", "stream_id"):
        if not isinstance(header.get(key), str) or not header[key]:
            raise ValueError(f"pcm_{key}_invalid")
    sequence = header.get("frame_sequence")
    if not isinstance(sequence, int) or sequence < 1:
        raise ValueError("pcm_frame_sequence_invalid")
    if header.get("sample_offset") != (sequence - 1) * FRAME_SAMPLES:
        raise ValueError("pcm_sample_offset_invalid")
    pcm = read_exact(stream, PCM_BYTES)
    if header.get("crc32") != f"{zlib.crc32(pcm) & 0xFFFFFFFF:08x}":
        raise ValueError("pcm_crc32_mismatch")
    return header, pcm


class PcmIngress:
    """Accept one external PCM stream and validate every frame continuously."""

    def __init__(
        self,
        socket_path: Path,
        on_pcm: Callable[[bytes], None],
        ack_interval: int = 32,
        on_first_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.on_pcm = on_pcm
        self.ack_interval = max(1, int(ack_interval))
        self.on_first_frame = on_first_frame
        self.frame_count = 0
        self.last_sequence = 0
        self.gap_count = 0
        self.sample_gap_count = 0
        self.last_header: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.bound = False
        self.connected = False
        self._bound_event = threading.Event()
        self._stop_event = threading.Event()
        self._server: socket.socket | None = None
        self._connection: socket.socket | None = None

    def wait_until_bound(self, timeout: float) -> bool:
        return self._bound_event.wait(timeout)

    def stop(self) -> None:
        self._stop_event.set()
        for value in (self._connection, self._server):
            if value is not None:
                try:
                    value.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    value.close()
                except OSError:
                    pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "socket_path": str(self.socket_path),
            "bound": self.bound,
            "connected": self.connected,
            "frame_count": self.frame_count,
            "last_sequence": self.last_sequence,
            "gap_count": self.gap_count,
            "sample_gap_count": self.sample_gap_count,
            "last_error": self.last_error,
            "stream_id": (self.last_header or {}).get("stream_id"),
            "source_node": (self.last_header or {}).get("source_node"),
        }

    def serve_one(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                self._server = server
                server.bind(str(self.socket_path))
                socket_mode = int(os.environ.get("EMBRY_PCM_SOCKET_MODE", "0660"), 8)
                self.socket_path.chmod(socket_mode)
                server.listen(1)
                self.bound = True
                self._bound_event.set()
                connection, _ = server.accept()
                self._connection = connection
                self.connected = True
                with connection, connection.makefile("rb") as stream:
                    while not self._stop_event.is_set():
                        try:
                            header, pcm = read_frame(stream)
                        except EOFError:
                            return
                        sequence = int(header["frame_sequence"])
                        if self.frame_count == 0 and self.on_first_frame is not None:
                            self.on_first_frame(header)
                        if sequence != self.last_sequence + 1:
                            self.gap_count += 1
                            raise RuntimeError("pcm_frame_sequence_gap")
                        expected_offset = self.frame_count * FRAME_SAMPLES
                        if header["sample_offset"] != expected_offset:
                            self.sample_gap_count += 1
                            raise RuntimeError("pcm_sample_offset_gap")
                        self.on_pcm(pcm)
                        self.last_header = header
                        self.last_sequence = sequence
                        self.frame_count += 1
                        if sequence % self.ack_interval == 0:
                            ack = canonical_json(
                                {
                                    "schema": "embry.pcm_ack.v1",
                                    "accepted_through_sequence": sequence,
                                    "frame_sequence": sequence,
                                    "gap_count": self.gap_count,
                                    "sample_gap_count": self.sample_gap_count,
                                }
                            )
                            connection.sendall(HEADER_LENGTH.pack(len(ack)) + ack)
        except Exception as exc:
            if self._stop_event.is_set():
                return
            self.last_error = f"{type(exc).__name__}:{exc}"
            raise
        finally:
            self.connected = False
            self.bound = False
            self._connection = None
            self._server = None
            self.socket_path.unlink(missing_ok=True)
