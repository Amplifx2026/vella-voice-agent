# vella-voice-agent

LiveKit Cloud voice agent for Vella. Stays lightweight — STT (Deepgram) →
Claude Opus 4.7 → TTS (ElevenLabs) — and delegates real marketing actions
to the Vella backend brain via the `/api/voice/action` bridge.

## Required environment

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Opus 4.7 LLM |
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `ELEVENLABS_API_KEY` | Text-to-speech |
| `LIVEKIT_URL` | LiveKit Cloud URL |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit auth |
| `VOICE_AGENT_SHARED_SECRET` | Server-to-server secret used by the action bridge — must match the Vella backend value |
| `VELLA_BACKEND_URL` | Optional override; defaults to `https://vella-backend-production.up.railway.app` |
| `VELLA_ACTION_TIMEOUT_S` | Optional per-action HTTP timeout (default `180`) |

## Action bridge

When the LLM decides to take a real action (generate posts, pull
analytics, post to social, run ads, etc.) it calls the `execute_action`
function tool, which `POST`s to `/api/voice/action` on the Vella backend.
The backend runs the request through the same MCP tool suite the web
chat uses (74+ tools — Higgsfield, Meta, TikTok, analytics, …) and
returns a short, speakable reply.

User identity is read from the LiveKit room/participant metadata
(`account_id`, baked in by the backend when minting the LiveKit token).
