# vella-voice-agent

LiveKit Cloud voice agent for Vella. Pipeline: Deepgram STT → Claude
Opus 4.7 → ElevenLabs TTS, with Silero VAD. The LLM has one function
tool — `execute_action` — that delegates real marketing actions to the
Vella Node backend's full MCP tool suite via `POST /api/voice/action`.

## Required environment

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Opus 4.7 LLM |
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `ELEVENLABS_API_KEY` | Text-to-speech |
| `LIVEKIT_URL` | LiveKit Cloud URL |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit auth |
| `VOICE_AGENT_SHARED_SECRET` | Server-to-server secret used by `execute_action` — must match the Vella backend value |
| `ELEVENLABS_VOICE_ID` | Optional override; defaults to `T720RsqorTx4ZZWohrNN` (matches the backend's voice) |
| `VELLA_BACKEND_URL` | Optional override; defaults to `https://vella-backend-production.up.railway.app` |
| `VELLA_ACTION_TIMEOUT_S` | Optional per-action HTTP timeout (default `180`) |

## Action bridge

When the LLM decides to take a real action it calls the `execute_action`
function tool, which `POST`s to `/api/voice/action` on the Vella backend.
The backend runs the request through the same MCP tool suite the web
chat uses (74+ tools — Higgsfield, Meta, TikTok, analytics, …) and
returns a short, speakable reply.

User identity is read from the LiveKit room/participant metadata
(`account_id`, baked in by the backend when minting the LiveKit token).

## Pinned to livekit-agents 1.4

`livekit-agents` 1.5 does not exist on PyPI yet (latest is `1.4.0rc2`).
`requirements.txt` is pinned to `~=1.4` so the build is deterministic.
When 1.5 ships, bump the pin and re-verify every constructor against
the new source.
