# Project Knowledge: RealtimeSTT

**Last updated:** 2026-07-10 by agent
**Status:** Active development

## Current Understanding

- Project initialized, knowledge tracking started
- 2026-07-03: RealtimeSTT is the listener/VAD-ASR companion for the Chatterbox voice stack, not the renderer or memory owner. Chatterbox Rung 8 receipts prove RealtimeSTT external-audio frame ingestion, automatic VAD endpointing for a Horus+factory+Embry stress WAV, physical microphone capture routed through listener and memory, local browser getUserMedia transport into Python, and downstream Chatterbox/Tau rendering. RealtimeSTT should expose clean audio-frame intake and transcript/recording-boundary events that a coordinator can route into speaker verification, pyannote diarization, memory speaker resolution, and Tau voice-render requests.
- 2026-07-10: The PipeWire proof runner can publish live `embry.voice_event.v1` callbacks to the localhost `embry-voice-control` journal using explicit `session_id`, `turn_id`, and monotonic sequence numbers. High-quality Horus proof session `embry-horus-hq-proof-20260710` produced three passing acoustic turns and 194 contiguous journal events. After the journal service restarted against the same SQLite WAL file, all 194 events and three final transcripts remained; replaying event 1 returned `inserted=false` and did not increase the count. Journal artifact: `/tmp/embry-horus-hq-event-spine-journal-after-restart.json`. Source: `/tmp/horus-high-quality-listener-source.wav`, derived from the Horus clone candidate rather than espeak-ng.

## Recent Decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-07-03 | Initialize project knowledge | Enable shared human/agent context |
| 2026-07-03 | Keep RealtimeSTT as listener companion | The proven architecture keeps RealtimeSTT responsible for VAD, recording boundaries, live/external audio intake, and ASR transcript events while Chatterbox remains the TTS renderer and memory/Tau remain responsible for identity, recall, and response policy. |
| 2026-07-10 | Publish callbacks to a persistent local journal | Per-run files and `latest` pointers cannot prove same-session lineage. The proof runner now optionally posts callbacks to the `embry-voice-control` localhost service while retaining local JSONL and WAV artifacts. |

## Open Questions

- [ ] What are the key architectural decisions?
- [ ] What are the known issues?
- [ ] Should RealtimeSTT add a first-class browser/WebRTC PCM bridge receipt that mirrors the Chatterbox browser getUserMedia smoke, or should the browser transport stay in Chatterbox/Tau integration harnesses only?

## Key Files

| File | Purpose |
|------|---------|
| PROJECT_KNOWLEDGE.md | Shared project knowledge |
| proofs/embry_pipewire_ingress/run_pipewire_realtimestt_ingress.py | PipeWire capture, RealtimeSTT callback, and optional live event-service publisher proof runner |

## Infrastructure State

<!-- Auto-populated from /project-state --quick -->
