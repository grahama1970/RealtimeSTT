#!/usr/bin/env python3
"""Run sanity checks against every Embry voice-control proof endpoint."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[2]
SERVICE = "proofs.embry_pipewire_ingress.service:app"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            last_error = response.text
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"health did not become ready: {last_error}")


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def main() -> int:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    output_root = Path("/tmp/embry-voice-control-endpoint-sanity") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root.mkdir(parents=True, exist_ok=False)
    result_path = output_root / "endpoint_sanity_receipt.json"

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            SERVICE,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    checks: list[dict] = []
    try:
        health = wait_for_health(base_url)
        assert_true(health["ok"], "health ok")
        checks.append({"endpoint": "GET /health", "status": "pass", "response": health})

        same = httpx.post(f"{base_url}/sanity/same-speaker", timeout=90).json()
        assert_true(same["status"] == "pass", "same-speaker status")
        assert_true(same["speaker_count"] == 1, "same-speaker expected one label")
        checks.append({"endpoint": "POST /sanity/same-speaker", "status": "pass", "receipt_path": same["receipt_path"]})

        different = httpx.post(f"{base_url}/sanity/different-speaker", timeout=90).json()
        assert_true(different["status"] == "pass", "different-speaker status")
        assert_true(different["speaker_count"] >= 2, "different-speaker expected two labels")
        checks.append({"endpoint": "POST /sanity/different-speaker", "status": "pass", "receipt_path": different["receipt_path"]})

        gate = httpx.post(f"{base_url}/speaker/gate", timeout=90).json()
        assert_true(gate["decision"] == "speaker_gate_rejected_not_enrolled", "speaker gate fail closed")
        assert_true(gate["accepted_for_tau"] is False, "speaker gate blocks tau")
        checks.append({"endpoint": "POST /speaker/gate", "status": "pass", "receipt_path": gate["receipt_path"]})

        ingress_payload = {
            "source_wav": "tests/unit/audio/asr-reference-short.wav",
            "expected_phrase": (
                "Hey guys! Welcome to the new demo of my real-time transcription library, "
                "designed to showcase its lightning-fast capabilities. As you'll see, "
                "speech is transcribed almost instantly into text"
            ),
            "capture_seconds": 15,
            "max_wer": 0.35,
            "speaker_gate_device": "cpu",
        }
        ingress = httpx.post(f"{base_url}/sanity/pipewire-realtimestt-ingress", json=ingress_payload, timeout=240).json()
        assert_true(ingress["status"] == "pass", "ingress status")
        assert_true(ingress["acceptance"]["pass"], "ingress acceptance")
        checks.append({"endpoint": "POST /sanity/pipewire-realtimestt-ingress", "status": "pass", "receipt_path": ingress["receipt_path"]})

        run_id = Path(same["receipt_path"]).parent.name
        fetched = httpx.get(f"{base_url}/receipts/speaker-separation/{run_id}", timeout=10).json()
        assert_true(fetched["run_id"] == run_id, "receipt fetch")
        checks.append({"endpoint": "GET /receipts/{run_id}", "status": "pass", "run_id": f"speaker-separation/{run_id}"})

        receipt = {
            "schema": "embry.voice_control_endpoint_sanity.v1",
            "status": "pass",
            "base_url": base_url,
            "check_count": len(checks),
            "checks": checks,
            "used_ui": False,
            "used_mock_transcript": False,
            "used_typed_prompt": False,
        }
        result_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(str(result_path))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        stdout, stderr = process.communicate(timeout=2) if process.poll() is not None else ("", "")
        receipt = {
            "schema": "embry.voice_control_endpoint_sanity.v1",
            "status": "fail",
            "base_url": base_url,
            "check_count": len(checks),
            "checks": checks,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "server_stdout": stdout,
            "server_stderr": stderr,
            "used_ui": False,
            "used_mock_transcript": False,
            "used_typed_prompt": False,
        }
        result_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(str(result_path))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
