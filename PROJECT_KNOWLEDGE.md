# Project Knowledge: RealtimeSTT

**Last updated:** 2026-07-03 08:44 by agent
**Status:** Active development

## Current Understanding

- Project initialized, knowledge tracking started
- 2026-07-03: RealtimeSTT is the listener/VAD-ASR companion for the Chatterbox voice stack, not the renderer or memory owner. Chatterbox Rung 8 receipts prove RealtimeSTT external-audio frame ingestion, automatic VAD endpointing for a Horus+factory+Embry stress WAV, physical microphone capture routed through listener and memory, local browser getUserMedia transport into Python, and downstream Chatterbox/Tau rendering. RealtimeSTT should expose clean audio-frame intake and transcript/recording-boundary events that a coordinator can route into speaker verification, pyannote diarization, memory speaker resolution, and Tau voice-render requests.

## Recent Decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-07-03 | Initialize project knowledge | Enable shared human/agent context |
| 2026-07-03 | Keep RealtimeSTT as listener companion | The proven architecture keeps RealtimeSTT responsible for VAD, recording boundaries, live/external audio intake, and ASR transcript events while Chatterbox remains the TTS renderer and memory/Tau remain responsible for identity, recall, and response policy. |

## Open Questions

- [ ] What are the key architectural decisions?
- [ ] What are the known issues?
- [ ] Should RealtimeSTT add a first-class browser/WebRTC PCM bridge receipt that mirrors the Chatterbox browser getUserMedia smoke, or should the browser transport stay in Chatterbox/Tau integration harnesses only?

## Key Files

| File | Purpose |
|------|---------|
| PROJECT_KNOWLEDGE.md | Shared project knowledge |

## Infrastructure State

<!-- Auto-populated from /project-state --quick -->
