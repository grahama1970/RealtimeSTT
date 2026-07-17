from __future__ import annotations

from io import BytesIO
import json
import socket
import threading

import pytest

from RealtimeSTT_server.embry_pcm import HEADER_LENGTH, PcmIngress, read_exact
from RealtimeSTT_server.embry_pcm_sender import (
    encode_frame,
    ffmpeg_command,
    pipewire_command,
    read_ack,
    send_stream,
)


def test_encoded_frame_round_trips_through_ingress_parser() -> None:
    pcm = bytes(range(256)) * 4
    from RealtimeSTT_server.embry_pcm import read_frame

    header, decoded = read_frame(BytesIO(encode_frame(
        pcm,
        source_node="alsa_input.c920",
        stream_id="stream-a",
        frame_sequence=3,
    )))
    assert decoded == pcm
    assert header["frame_sequence"] == 3
    assert header["sample_offset"] == 1024


def test_sender_consumes_ack_and_delivers_contiguous_frames(tmp_path) -> None:
    socket_path = tmp_path / "pcm.sock"
    received: list[bytes] = []
    ingress = PcmIngress(socket_path, received.append, ack_interval=2)
    thread = threading.Thread(target=ingress.serve_one, daemon=True)
    thread.start()
    assert ingress.wait_until_bound(2)

    pcm = b"".join(bytes([index]) * 1024 for index in range(1, 5))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        assert send_stream(
            connection,
            BytesIO(pcm),
            source_node="alsa_input.c920",
            stream_id="stream-a",
            ack_interval=2,
        ) == 4

    ingress.stop()
    thread.join(timeout=2)
    assert received == [bytes([index]) * 1024 for index in range(1, 5)]
    assert ingress.frame_count == 4
    assert ingress.gap_count == 0
    assert ingress.sample_gap_count == 0


def test_sender_rejects_ack_sequence_mismatch() -> None:
    payload = json.dumps({
        "schema": "embry.pcm_ack.v1",
        "accepted_through_sequence": 31,
        "frame_sequence": 31,
        "gap_count": 0,
        "sample_gap_count": 0,
    }).encode()
    stream = BytesIO(HEADER_LENGTH.pack(len(payload)) + payload)
    with pytest.raises(ValueError, match="pcm_ack_sequence_mismatch"):
        read_ack(stream, 32)


def test_capture_pipeline_preserves_native_stereo_before_downmix() -> None:
    capture = pipewire_command("alsa_input.c920")
    convert = ffmpeg_command()
    assert capture[capture.index("--rate") + 1] == "48000"
    assert capture[capture.index("--channels") + 1] == "2"
    assert convert[-5:] == ["s16le", "-ar", "16000", "-ac", "1", "pipe:1"][-5:]
