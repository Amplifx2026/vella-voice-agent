# vella-voice-agent

LiveKit Cloud voice agent for Vella. Minimal pipeline: Deepgram STT →
Claude Opus 4.7 → ElevenLabs TTS, with Silero VAD for endpointing.

This file currently does ONE thing: connect to a LiveKit room and have
a conversation. The backend `/api/voice/action` bridge that lets the
LLM execute real marketing actions (post generation, analytics, ads,
etc.) is intentionally NOT wired in yet — it lands in a follow-up once
we've confirmed the agent boots and speaks reliably on Railway.

## Required environment

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Opus 4.7 LLM |
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `ELEVENLABS_API_KEY` | Text-to-speech |
| `LIVEKIT_URL` | LiveKit Cloud URL |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit auth |

## Pinned to livekit-agents 1.4

`livekit-agents` 1.5 does not exist on PyPI yet (latest is `1.4.0rc2`).
`requirements.txt` is pinned to `~=1.4` so the build is deterministic.
When 1.5 ships, bump the pin and re-verify every constructor against
the new source.
