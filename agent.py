"""Vella voice agent — verified against livekit-agents 1.4.0rc2.

Pipeline: Deepgram STT → Claude Opus 4.7 → ElevenLabs TTS, with
Silero VAD and a backend bridge tool (`execute_action`) that
delegates real marketing actions to the Vella Node backend's full
MCP tool suite.

GREETING FLOW (when a SIP caller joins):
  1. Extract phone from the SIP participant's identity.
  2. Kick off a background lookup (`_lookup_caller`) against the backend.
  3. Speak a short literal "Hey, this is Vella" via session.say() — no
     LLM round-trip, plays immediately.
  4. While that's playing, the lookup completes in parallel.
  5. Generate a personalized follow-up via the LLM that references the
     caller's name + business, OR a generic "what can I help with?"
     when the caller isn't recognized.

Every API call here was checked against the installed package source
on disk — not GitHub main, not docs. See README.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import anthropic, deepgram, elevenlabs, silero

logger = logging.getLogger("agent-Vella")
load_dotenv(".env.local")

# ─── Backend bridge config ────────────────────────────────────────────
VELLA_BACKEND_URL = os.environ.get(
    "VELLA_BACKEND_URL",
    "https://vella-backend-production.up.railway.app",
).rstrip("/")
VELLA_AGENT_SECRET = os.environ.get("VOICE_AGENT_SHARED_SECRET", "")
VELLA_ACTION_TIMEOUT_S = int(os.environ.get("VELLA_ACTION_TIMEOUT_S", "180"))

# Caller lookup is a fast DB query — keep its timeout aggressive so a
# slow backend doesn't make the second greeting feel laggy.
VELLA_LOOKUP_TIMEOUT_S = float(os.environ.get("VELLA_LOOKUP_TIMEOUT_S", "3.0"))

# ─── Voice config ─────────────────────────────────────────────────────
# Defaults to ElevenLabs' "Rachel" voice — a well-known, high-quality
# stock voice that's been validated by the community for speech-agent
# use. The previous default (T720RsqorTx4ZZWohrNN) was a custom clone
# and may have been the source of the garbled-audio reports. Override
# via env when you've validated a custom voice end-to-end.
VELLA_VOICE_ID = os.environ.get(
    "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
)


VELLA_INSTRUCTIONS = """
You are Vella. You're on a phone call with a real person. Speak like a confident marketing strategist, not a chatbot.

HOW YOU SOUND:
- Be direct and authoritative. Lead with the answer, then explain. State what you're doing and why.
- Professional and warm — you're the expert they called, not a help desk. Keep the warmth, drop the casualness.
- Use contractions every time: I'm, don't, what's, here's, you're, that's, I'll.
- Keep replies SHORT — one or two sentences max per turn. People interrupt — that's fine.
- No filler words. Skip "um", "well", "let me think", "so what I'm seeing is…". Don't open with "Great question!", "I'd be happy to", "Certainly!", "Absolutely!". Lead with the finding.
- If you're cut off, stop talking immediately. Listen.
- Numbers conversational: "twenty bucks", "around three thousand", "half a percent" — never "$20.00" or "3,000.00".
- No markdown, no lists, no bullet points, no URLs. You're speaking, not writing.
- Don't say "as an AI" or "I'm an assistant". Don't reveal you're AI unless someone directly asks.

WHAT YOU CAN DO:
You can actually do things — generate posts, ads, captions, images; send texts and emails; check analytics; launch or pause campaigns; look stuff up on the web; pull up client info. When the caller asks for something, call `execute_action` with a plain-English description of what they want, including the specifics they mentioned.

Good action: "Generate three Instagram captions for our spring sale, friendly tone, twenty percent off, ends Sunday"
Bad action: "make captions"

The tool gives you a short result. Just say it. You can add a quick framing sentence if it helps, but don't pad. If something breaks, say so plainly and offer one quick fix.
"""


# ─── Identity helpers ─────────────────────────────────────────────────

def _extract_user_id(metadata: Optional[str]) -> Optional[str]:
    """The backend mints LiveKit tokens with metadata = {account_id, ...}.
    The user's participant.metadata carries that JSON when they join."""
    if not metadata:
        return None
    try:
        data = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return None
    return data.get("account_id") or data.get("userId")


def _phone_from_identity(identity: Optional[str]) -> Optional[str]:
    """Parse an E.164 phone from a LiveKit participant identity or room
    name. LiveKit's SIP integration registers inbound callers with
    identity like `sip_+18286722118`. Returns the phone in E.164 form
    (always with leading `+`), or None when nothing phone-shaped is
    found."""
    if not identity:
        return None
    s = identity.strip()
    if s.startswith("sip_"):
        s = s[4:]
    # Allow optional leading +, then 8–15 digits (E.164 max is 15)
    m = re.search(r"\+?(\d{8,15})", s)
    if not m:
        return None
    return "+" + m.group(1)


def _extract_phone(room: rtc.Room) -> Optional[str]:
    """Return the inbound caller's phone if a SIP participant is present.
    Falls back to the room name (some flows encode the number there)."""
    for p in room.remote_participants.values():
        phone = _phone_from_identity(p.identity)
        if phone:
            return phone
    return _phone_from_identity(room.name)


# ─── Agent ────────────────────────────────────────────────────────────

class VellaAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=VELLA_INSTRUCTIONS)
        self._user_id: Optional[str] = None
        self._room_id: Optional[str] = None
        self._phone: Optional[str] = None
        self._http: Optional[aiohttp.ClientSession] = None

    def bind_session(
        self,
        room_id: str,
        user_id: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> None:
        self._room_id = room_id
        self._user_id = user_id
        self._phone = phone
        logger.info(
            "[Vella] session bound — room=%s user=%s phone=%s",
            room_id, user_id, phone,
        )

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def aclose(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()

    async def _post_action(
        self,
        action: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> dict:
        """Internal: POST /api/voice/action. Returns the parsed JSON body
        regardless of HTTP status so callers can route on `ok` / `found`.
        Raises only on network/JSON failure."""
        if not VELLA_AGENT_SECRET:
            raise RuntimeError("VOICE_AGENT_SHARED_SECRET is not configured")
        url = f"{VELLA_BACKEND_URL}/api/voice/action"
        payload = {
            "action": action,
            "userId": self._user_id,
            "roomId": self._room_id,
        }
        headers = {
            "Content-Type": "application/json",
            "X-Vella-Agent-Secret": VELLA_AGENT_SECRET,
        }
        session = await self._ensure_http()
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s or VELLA_ACTION_TIMEOUT_S),
        ) as resp:
            return await resp.json(content_type=None)

    async def lookup_caller(self) -> Optional[dict]:
        """Pre-greeting: ask the backend who's calling. Returns the
        `caller` dict from the response when found, else None. All
        failures swallowed — caller-recognition is best-effort."""
        if not self._phone:
            return None
        if not VELLA_AGENT_SECRET:
            logger.warning("[Vella] caller lookup skipped — no agent secret")
            return None
        try:
            body = await self._post_action(
                f"lookup_caller: {self._phone}",
                timeout_s=VELLA_LOOKUP_TIMEOUT_S,
            )
        except Exception:
            logger.exception("[Vella] caller lookup HTTP error")
            return None
        if not body.get("ok") or not body.get("found"):
            logger.info("[Vella] caller not recognized: %s", self._phone)
            return None
        caller = body.get("caller") or {}
        logger.info(
            "[Vella] caller recognized — name=%s business=%s",
            caller.get("name"), caller.get("businessName"),
        )
        return caller

    # ─── LLM-callable tool ──────────────────────────────────────────────
    # SCHEMA SAFETY: single `str` arg only. RunContext is auto-skipped from
    # the generated schema (utils.py:317) and the FunctionTool descriptor
    # strips `self` (tool_context.py:153). The 1.4 anthropic provider uses
    # build_legacy_openai_schema, which produces `additionalProperties: true`
    # for any free `dict` field — Anthropic rejects that. Do NOT add a dict
    # arg here.
    @function_tool
    async def execute_action(self, context: RunContext, action: str) -> str:
        """Execute a real marketing action through the Vella backend brain.

        Use this whenever the user asks you to DO something — generate posts,
        pull analytics, post to social media, manage campaigns, check the
        inbox, build landing pages, run ads, research competitors. The
        backend brain has access to all 74 MCP tools and runs the action
        end-to-end. The returned string is short and ready to speak aloud.

        Args:
            action: Natural-language description of what to do, including
                any specific values the user named (platform, budget, tone,
                dates). e.g. "Generate three Instagram captions for our
                spring sale, friendly tone, twenty percent off offer".
        """
        if not VELLA_AGENT_SECRET:
            logger.error("[Vella] VOICE_AGENT_SHARED_SECRET not configured")
            return "I can't run that action — the backend bridge isn't configured. Tell the team to set the voice agent secret."

        if not self._user_id:
            logger.error("[Vella] execute_action invoked before user bound")
            return "I don't know which account you're signed in as yet. Try again in a moment."

        logger.info("[Vella] execute_action → %s", action[:120])
        try:
            body = await self._post_action(action)
        except asyncio.TimeoutError:
            logger.warning("[Vella] action timed out after %ss", VELLA_ACTION_TIMEOUT_S)
            return "That action took too long and timed out. Should I try a smaller version of it?"
        except Exception as exc:  # noqa: BLE001 — surface anything to the user
            logger.exception("[Vella] action errored")
            return f"Something broke on my end: {exc}. Want me to retry?"

        if not body.get("ok"):
            err = body.get("error") or "unknown error"
            logger.warning("[Vella] action failed: %s", err)
            return f"That didn't work — {err}. Want me to try again or do something else?"
        reply = body.get("reply") or "Done."
        logger.info("[Vella] action ok — %s", reply[:120])
        return reply


# ─── Greeting prompt builders ─────────────────────────────────────────

def _follow_up_instructions(caller: Optional[dict]) -> Optional[str]:
    """Build the instructions for the post-greeting personalized turn.
    The first greeting was 'Hey, this is Vella. What's going on?' via TTS,
    which already asks the caller what they need. So we only follow up
    when we recognize the caller and want to drop their name in. For
    unknown callers, return None — let them speak first."""
    if not caller or not caller.get("name"):
        return None
    name = caller["name"]
    biz = caller.get("businessName")
    if biz:
        return (
            f"You just realized the caller is {name} from {biz}. "
            f"Drop their first name in casually so they know you recognize them — "
            f"something like 'Oh hey {name} — how's {biz}?' or '{name}! What's new with {biz}?'. "
            "One short sentence, very casual. Don't repeat 'Hey, this is Vella' or 'What's going on'."
        )
    return (
        f"You just realized the caller is {name}. "
        f"Drop their first name in casually — something like 'Oh hey {name}!' or "
        f"'{name} — what's up?'. "
        "One short sentence, very casual. Don't repeat the greeting."
    )


# ─── Entrypoint ───────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    logger.info("Connecting to room: %s", ctx.room.name)
    await ctx.connect()

    agent = VellaAgent()

    # Identify caller. Web sessions carry account_id in participant
    # metadata; SIP calls carry the phone in the SIP participant's
    # identity. Either or both may be present; either or both may be
    # absent.
    user_id = _extract_user_id(ctx.room.metadata)
    for p in ctx.room.remote_participants.values():
        if not user_id:
            user_id = _extract_user_id(p.metadata)
    phone = _extract_phone(ctx.room)
    agent.bind_session(ctx.room.name, user_id=user_id, phone=phone)

    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        if not agent._user_id:
            uid = _extract_user_id(participant.metadata)
            if uid:
                agent.bind_session(ctx.room.name, user_id=uid, phone=agent._phone)
        if not agent._phone:
            ph = _phone_from_identity(participant.identity)
            if ph:
                agent.bind_session(ctx.room.name, user_id=agent._user_id, phone=ph)

    @ctx.room.on("participant_metadata_changed")
    def _on_metadata_changed(participant: rtc.RemoteParticipant, _prev: str) -> None:
        if agent._user_id:
            return
        uid = _extract_user_id(participant.metadata)
        if uid:
            agent.bind_session(ctx.room.name, user_id=uid, phone=agent._phone)

    session = AgentSession(
        llm=anthropic.LLM(
            # claude-opus-4-8 deprecated `temperature` — leave it unset.
            model="claude-opus-4-8",
        ),
        # Model: eleven_turbo_v2_5 (the elevenlabs plugin's own default —
        # verified in livekit/plugins/elevenlabs/tts.py:101). Better
        # naturalness than eleven_flash_v2_5 with only marginally higher
        # latency. flash_v2_5 prioritizes speed and was producing garbled
        # output paired with the aggressive voice-settings tuning below.
        #
        # Voice settings: Rachel, tuned to sound like a confident marketing
        # executive giving a briefing — not a smooth late-night radio host.
        # High stability (0.71) strips out the breathy variation for a clean,
        # direct read; similarity_boost (0.82) tracks the target voice; style
        # kept minimal (0.08) so it stays professional, not performative.
        tts=elevenlabs.TTS(
            model="eleven_turbo_v2_5",
            voice_id=VELLA_VOICE_ID,
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.71,
                similarity_boost=0.82,
                style=0.08,
                use_speaker_boost=True,
            ),
        ),
        stt=deepgram.STT(model="nova-3", language="en-US"),
        vad=silero.VAD.load(),
    )

    # session.start signature: (agent: Agent, *, room: Room | NOT_GIVEN, ...)
    await session.start(agent, room=ctx.room)

    # ── Greeting flow ──
    # Kick off the caller lookup BEFORE the first greeting so it runs
    # in parallel with TTS playback. By the time "Hey, this is Vella"
    # finishes (~1.5s), the lookup is usually done.
    lookup_task: Optional[asyncio.Task] = None
    if phone:
        lookup_task = asyncio.create_task(agent.lookup_caller())

    # Speak the immediate, casual greeting via TTS — no LLM round-trip.
    # session.say returns a SpeechHandle; awaiting it waits for playout.
    greeting_handle = session.say("Hey, this is Vella. What's going on?")

    # Resolve the lookup (capped at VELLA_LOOKUP_TIMEOUT_S so we don't
    # hold the call hostage if the backend hangs). All errors → caller=None.
    caller: Optional[dict] = None
    if lookup_task is not None:
        try:
            caller = await asyncio.wait_for(
                lookup_task, timeout=VELLA_LOOKUP_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning("[Vella] caller lookup wait timed out")
            lookup_task.cancel()
        except Exception:
            logger.exception("[Vella] caller lookup awaited error")

    # Wait for the literal greeting to finish before we start speaking
    # the personalized follow-up — otherwise both speeches overlap.
    await greeting_handle

    # Personalized follow-up only when we recognized the caller — the
    # greeting already asked "What's going on?", so for unknown callers
    # we just wait for them to speak.
    follow_up = _follow_up_instructions(caller)
    if follow_up is not None:
        await session.generate_reply(instructions=follow_up)


def prewarm(proc: JobProcess) -> None:
    """Pre-load VAD so the first job doesn't pay the ~200ms init cost."""
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="Vella",
        )
    )
