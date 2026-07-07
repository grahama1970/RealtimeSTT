# Embry PipeWire Ingress Proof

This proof is intentionally non-UI. It proves only:

```text
real local audio path -> captured PCM -> RealtimeSTT -> realtime/final transcript event
```
It does not prove browser mic capture, speaker identity, Tau/memory,
Chatterbox, Chat UX sync, orb sync, replay, or interruption.

## Run

Use the local RealtimeSTT virtualenv:

```bash
./.venv-fastapi/bin/python proofs/embry_pipewire_ingress/run_pipewire_realtimestt_ingress.py
```

The runner writes a run directory under:

```text
/tmp/embry-realtimestt-ingress/<run_id>/
```

Expected artifacts:

- `receipt.json`
- `source.wav`
- `captured.wav`
- `captured.raw`
- `realtime_stt_callbacks.jsonl`
- `live_session_events.jsonl`
- `speaker_gate_events.jsonl`
- `commands.txt`
- `environment.txt`

Default local targets are based on this workstation's detected PipeWire graph:

- playback sink target: `64` (`Jabra SPEAK 510 Analog Stereo`)
- capture source target: `67` (`USB Audio Front Microphone`)

Override targets when needed:

```bash
./.venv-fastapi/bin/python proofs/embry_pipewire_ingress/run_pipewire_realtimestt_ingress.py \
  --playback-target 64 \
  --capture-target 67
```

The runner also executes a fail-closed pyannote speaker gate when
`pyannote.audio` and `HF_TOKEN` are available:

```text
captured.wav -> pyannote diarization -> speaker_gate.rejected.not_enrolled
```

This does not identify Horus. It only proves diarization can run and that Tau,
Chatterbox, and UI routing remain blocked when no enrolled primary-speaker
profile exists.

## Endpoint Service

Start the proof API:

```bash
HF_TOKEN=... ./.venv-fastapi/bin/python -m uvicorn \
  proofs.embry_pipewire_ingress.service:app \
  --host 127.0.0.1 \
  --port 8769
```

Endpoints:

- `GET /health`
- `POST /sanity/same-speaker`
- `POST /sanity/different-speaker`
- `POST /speaker/gate`
- `POST /speaker/enroll`
- `POST /sanity/primary-acceptance`
- `POST /sanity/non-primary-rejection`
- `POST /sanity/overlap-rejection`
- `POST /sanity/noise-probe`
- `POST /sanity/self-audio-rejection`
- `POST /sanity/pipewire-realtimestt-ingress`
- `GET /receipts/{run_id}`

The enrollment and primary-acceptance checks use synthetic local speech
fixtures. They prove the endpoint contract and fail-closed policy wiring, not
real Horus identity.

Run endpoint sanity checks:

```bash
HF_TOKEN=... ./.venv-fastapi/bin/python \
  proofs/embry_pipewire_ingress/sanity_endpoints.py
```

The sanity runner starts a temporary local API server, calls every endpoint
above, validates each returned receipt, and writes an endpoint sanity receipt
under `/tmp/embry-voice-control-endpoint-sanity/`.
