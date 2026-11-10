"""Turn-based Twilio phone interface for SerenAI (development prototype)."""
import os
import re
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

from agent.pipeline import process_message

load_dotenv()
app = FastAPI()

# Development-only, in-memory sessions. Restarting the server clears them.
# Use a durable store before running multiple server workers or deploying.
sessions = {}
sessions_lock = Lock()


def xml(response):
    return Response(content=str(response), media_type="application/xml")


def validate_twilio(request: Request, form: dict):
    token = os.getenv("TWILIO_AUTH_TOKEN")
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not token or not base:
        raise HTTPException(500, "Missing Twilio configuration")
    # Twilio signs the exact external URL, including query string if present.
    url = base + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    signature = request.headers.get("X-Twilio-Signature", "")
    if not RequestValidator(token).validate(url, form, signature):
        raise HTTPException(403, "Invalid Twilio signature")


async def verified_form(request: Request):
    form = dict(await request.form())
    validate_twilio(request, form)
    return form


def ask(response, prompt, action="/respond"):
    gather = response.gather(
        input="speech",
        action=action,
        method="POST",
        speech_timeout="auto",
        timeout=7,
        action_on_empty_result=True,
    )
    gather.say(prompt, voice="alice")
    return response


def preferred_name(text):
    text = text.strip()
    for pattern in (
        r"\bmy name is\s*,?\s*(.+)",
        r"\byou can call me\s*,?\s*(.+)",
        r"\bcall me\s*,?\s*(.+)",
        r"\bi'm\s+(.+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = match.group(1)
            break
    text = re.sub(r"^(?:uh|um|erm|hmm)[,\s]+", "", text, flags=re.I)
    text = text.strip(" .,!?")
    # Don't treat a long explanation as a person's name.
    if not text or len(text.split()) > 3 or len(text) > 50:
        return None
    return text.title()


@app.get("/")
def home():
    return {"message": "SerenAI Twilio server is running"}


@app.post("/voice")
async def incoming_call(request: Request):
    form = await verified_form(request)
    sid = form.get("CallSid")
    if not sid:
        raise HTTPException(400, "Missing CallSid")
    with sessions_lock:
        sessions[sid] = {"name": None, "history": [], "stage": "name"}
    response = VoiceResponse()
    ask(
        response,
        "Hi, I'm SerenAI, your AI stress support assistant. "
        "I'm here to listen and help with everyday stress. "
        "Before we begin, what should I call you?",
    )
    return xml(response)


@app.post("/respond")
async def respond(request: Request):
    form = await verified_form(request)
    sid = form.get("CallSid")
    if not sid:
        raise HTTPException(400, "Missing CallSid")
    speech = (form.get("SpeechResult") or "").strip()
    with sessions_lock:
        session = sessions.get(sid)
    if session is None:
        response = VoiceResponse()
        response.say("This call session has expired. Please call again.")
        response.hangup()
        return xml(response)

    response = VoiceResponse()
    if not speech:
        ask(response, "I didn't catch that. Could you please say it again?")
        return xml(response)

    if session["stage"] == "name":
        name = preferred_name(speech)
        if not name:
            ask(response, "Sorry, I didn't catch your name. Please say just the name you'd like me to use.")
            return xml(response)
        session["name"] = name
        session["stage"] = "conversation"
        ask(response, f"Nice to meet you, {name}. How are you feeling today?")
        return xml(response)

    if speech.lower().strip(" .!?") in {"quit", "goodbye", "bye"}:
        response.say("Take care. Goodbye!", voice="alice")
        response.hangup()
        with sessions_lock:
            sessions.pop(sid, None)
        return xml(response)

    try:
        # Run blocking OpenAI and Hugging Face work outside the event loop.
        result = await run_in_threadpool(
            process_message,
            speech,
            user_name=session["name"],
            history=session["history"],
        )
    except Exception:
        # Avoid leaking internal error details to callers.
        ask(response, "I'm having trouble responding right now. Please try again.")
        return xml(response)

    reply = result["response"]
    if result["should_close"]:
        response.say(reply, voice="alice")
        response.hangup()
        with sessions_lock:
            sessions.pop(sid, None)
    else:
        ask(response, reply)
    return xml(response)
