from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import socket
import struct
import threading
import time
import zlib

import pytest

from RealtimeSTT_server.embry_pcm import (
    HEADER_LENGTH,
    PCM_BYTES,
    PcmIngress,
    read_frame,
)


def encoded_frame(sequence: int, *, pcm: bytes | None = None) -> bytes:
    payload = pcm if pcm is not None else bytes([sequence % 251]) * PCM_BYTES
    header = {
        "schema": "embry.pcm_frame.v1",
        "format": "s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
        "samples": 512,
        "payload_bytes": PCM_BYTES,
        "source_node": "alsa_input.jabra",
        "stream_id": "stream-a",
        "frame_sequence": sequence,
        "sample_offset": (sequence - 1) * 512,
        "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
    }
    raw_header = json.dumps(header, sort_keys=True).encode()
    return HEADER_LENGTH.pack(len(raw_header)) + raw_header + payload


def test_read_frame_accepts_exact_16k_mono_frame() -> None:
    header, pcm = read_frame(BytesIO(encoded_frame(1)))
    assert header["frame_sequence"] == 1
    assert header["sample_offset"] == 0
    assert len(pcm) == PCM_BYTES


def test_read_frame_rejects_crc_mismatch() -> None:
    raw = bytearray(encoded_frame(1))
    raw[-1] ^= 0xFF
    with pytest.raises(ValueError, match="pcm_crc32_mismatch"):
        read_frame(BytesIO(bytes(raw)))


def test_ingress_rejects_frame_gap_and_records_counter(tmp_path: Path) -> None:
    socket_path = tmp_path / "pcm.sock"
    received: list[bytes] = []
    ingress = PcmIngress(socket_path, received.append, ack_interval=100)
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            ingress.serve_one()
        except BaseException as exc:  # test captures the exact boundary failure
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ingress.wait_until_bound(2.0)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(encoded_frame(1))
        client.sendall(encoded_frame(3))

    thread.join(timeout=2.0)
    assert errors
    assert "pcm_frame_sequence_gap" in str(errors[0])
    assert ingress.frame_count == 1
    assert ingress.gap_count == 1
    assert ingress.sample_gap_count == 0
    assert len(received) == 1


def test_ingress_exposes_bound_readiness_before_client_connects(tmp_path: Path) -> None:
    ingress = PcmIngress(tmp_path / "ready.sock", lambda _pcm: None)
    thread = threading.Thread(target=ingress.serve_one, daemon=True)
    thread.start()
    assert ingress.wait_until_bound(2.0)
    assert ingress.snapshot()["bound"] is True
    ingress.stop()
    thread.join(timeout=2.0)
