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
