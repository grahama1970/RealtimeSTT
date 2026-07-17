"""PipeWire capture sender for the Embry framed-PCM ingress."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import time
from typing import Any, BinaryIO, Sequence
from uuid import uuid4
import zlib

from RealtimeSTT_server.embry_pcm import (
    FRAME_SAMPLES,
    HEADER_LENGTH,
    MAX_HEADER_BYTES,
    PCM_BYTES,
    canonical_json,
    read_exact,
)


LOGGER = logging.getLogger("embry_pcm_sender")
ACK_INTERVAL = 32


def encode_frame(
    pcm: bytes,
    *,
    source_node: str,
    stream_id: str,
    frame_sequence: int,
) -> bytes:
    """Encode one exact 16 kHz mono s16le PCM frame."""
    if len(pcm) != PCM_BYTES:
        raise ValueError("pcm_payload_bytes_invalid")
    if frame_sequence < 1:
        raise ValueError("pcm_frame_sequence_invalid")
    header = canonical_json(
        {
            "schema": "embry.pcm_frame.v1",
            "format": "s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "samples": FRAME_SAMPLES,
            "payload_bytes": PCM_BYTES,
            "source_node": source_node,
            "stream_id": stream_id,
            "frame_sequence": frame_sequence,
            "sample_offset": (frame_sequence - 1) * FRAME_SAMPLES,
            "crc32": f"{zlib.crc32(pcm) & 0xFFFFFFFF:08x}",
        }
    )
    return HEADER_LENGTH.pack(len(header)) + header + pcm


def read_ack(stream: BinaryIO, expected_sequence: int) -> dict[str, Any]:
    """Read and validate one receiver acknowledgement."""
    header_size = HEADER_LENGTH.unpack(read_exact(stream, HEADER_LENGTH.size))[0]
    if header_size < 2 or header_size > MAX_HEADER_BYTES:
        raise ValueError("pcm_ack_length_invalid")
    try:
        ack = json.loads(read_exact(stream, header_size))
    except json.JSONDecodeError as exc:
        raise ValueError("pcm_ack_json_invalid") from exc
    if not isinstance(ack, dict) or ack.get("schema") != "embry.pcm_ack.v1":
        raise ValueError("pcm_ack_invalid")
    if ack.get("accepted_through_sequence") != expected_sequence:
        raise ValueError("pcm_ack_sequence_mismatch")
    if ack.get("gap_count") != 0 or ack.get("sample_gap_count") != 0:
        raise RuntimeError("pcm_ack_reports_gap")
    return ack


def send_stream(
    connection: socket.socket,
    pcm_stream: BinaryIO,
    *,
    source_node: str,
    stream_id: str,
    ack_interval: int = ACK_INTERVAL,
) -> int:
    """Send a continuous PCM stream and return the complete frame count."""
    frame_sequence = 0
    with connection.makefile("rb") as ack_stream:
        while True:
            pcm = pcm_stream.read(PCM_BYTES)
            if not pcm:
                return frame_sequence
            if len(pcm) != PCM_BYTES:
                raise EOFError("pcm_capture_partial_frame")
            frame_sequence += 1
            connection.sendall(
                encode_frame(
                    pcm,
                    source_node=source_node,
                    stream_id=stream_id,
                    frame_sequence=frame_sequence,
                )
            )
            if frame_sequence % ack_interval == 0:
                read_ack(ack_stream, frame_sequence)


def pipewire_command(source_node: str) -> list[str]:
    """Capture the C920-compatible native stereo stream from PipeWire."""
    return [
        "pw-record",
        "--target",
        source_node,
        "--rate",
        "48000",
        "--channels",
        "2",
        "--format",
        "s16",
        "-",
    ]


def ffmpeg_command() -> list[str]:
    """Downmix and resample native capture to the ingress PCM contract."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "pipe:1",
    ]


def connect_with_retry(
    socket_path: Path,
    *,
    timeout_seconds: float,
    retry_seconds: float = 0.25,
) -> socket.socket:
    """Connect to the runtime socket within a bounded startup window."""
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(socket_path))
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
            time.sleep(retry_seconds)
    raise TimeoutError(f"pcm_socket_connect_timeout:{socket_path}:{last_error}")


def run_sender(
    *,
    socket_path: Path,
    source_node: str,
    stream_id: str,
    log_path: Path,
    connect_timeout_seconds: float,
) -> int:
    """Run physical capture, conversion, and framed socket delivery."""
    for executable in ("pw-record", "ffmpeg"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required_executable_missing:{executable}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_with_retry(
        socket_path,
        timeout_seconds=connect_timeout_seconds,
    )
    with ExitStack() as stack:
        stack.callback(connection.close)
        log = stack.enter_context(log_path.open("ab"))
        capture = subprocess.Popen(
            pipewire_command(source_node),
            stdout=subprocess.PIPE,
            stderr=log,
        )
        stack.callback(_stop_process, capture)
        assert capture.stdout is not None
        converter = subprocess.Popen(
            ffmpeg_command(),
            stdin=capture.stdout,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        capture.stdout.close()
        stack.callback(_stop_process, converter)
        assert converter.stdout is not None
        frame_count = send_stream(
            connection,
            converter.stdout,
            source_node=source_node,
            stream_id=stream_id,
        )
        converter_returncode = converter.wait(timeout=5)
        capture_returncode = capture.wait(timeout=5)
        if converter_returncode != 0:
            raise RuntimeError(f"ffmpeg_capture_failed:{converter_returncode}")
        if capture_returncode != 0:
            raise RuntimeError(f"pipewire_capture_failed:{capture_returncode}")
        return frame_count


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--source-node", required=True)
    parser.add_argument("--stream-id", default=f"pipewire-{uuid4().hex}")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info(
        "connecting source_node=%s socket_path=%s stream_id=%s log_path=%s",
        args.source_node,
        args.socket_path,
        args.stream_id,
        args.log_path,
    )
    try:
        frame_count = run_sender(
            socket_path=args.socket_path,
            source_node=args.source_node,
            stream_id=args.stream_id,
            log_path=args.log_path,
            connect_timeout_seconds=args.connect_timeout_seconds,
        )
    except KeyboardInterrupt:
        LOGGER.info("capture interrupted")
        return 0
    LOGGER.info("capture ended frame_count=%d", frame_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
