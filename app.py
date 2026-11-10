"""
WE web UI.

Run:   python app.py                  -> http://127.0.0.1:5000
Demo:  SERENAI_DEMO=1 python app.py   (no models / API keys; canned scenarios to preview the UI)
"""
import io
import os
import re
import struct
import time
import uuid
import wave
from collections import OrderedDict

from name_utils import (clean_typed_name, is_no, is_yes, name_candidates, spell_out,
                        spelled_name)
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEMO = os.getenv("SERENAI_DEMO") == "1"
if DEMO:
    os.environ.setdefault("RAG_FAKE_EMBED", "1")   # offline keyword embeddings so RAG works without an API key
from rag import info as rag_info, retrieve_knowledge  # noqa: E402

TOPICS = ["academic", "work or career", "relationships", "family",
          "financial", "health or wellness", "general"]

if DEMO:
    INTRO = (
        "Hi, I'm SerenAI, your stress support assistant. "
        "I'm here to listen and help you work through everyday stress. "
        "Before we get started, what should I call you?"
    )

    def _emo(**kw):
        d = {"anger": .02, "disgust": .02, "fear": .02, "joy": .02, "neutral": .02, "sadness": .02, "surprise": .02}
        d.update(kw)
        return [{"label": k, "score": v} for k, v in d.items()]

    def _topic(top, score):
        rest = [t for t in TOPICS if t != top]
        left = (1 - score) / len(rest)
        return [{"label": top, "score": score}] + [{"label": t, "score": left} for t in rest]

    def process_message(text, user_name=None, history=None):
        time.sleep(0.7)
        low = text.lower()
        r = dict(safety_state="normal", conversation_state="continue", should_close=False)
        if "give up" in low or "can't take this" in low:
            r.update(safety_state="concern", conversation_state=None, emotion=None, emotion_score=None,
                     stress_label=None, stress_score=None, topic=None, topic_score=None,
                     response="That sounds incredibly heavy, and I'm really glad you told me. "
                              "Are you having thoughts of hurting yourself?")
        elif low.strip(" .!") in ("bye", "thanks bye", "i'm good now"):
            r.update(should_close=True, conversation_state="close", emotion=None, emotion_score=None,
                     stress_label=None, stress_score=None, topic=None, topic_score=None,
                     response="I'm glad we could talk. Take care of yourself, and come back whenever you'd like.")
        else:
            if "afraid" in low or "talk" in low:
                e, st, tp = _emo(fear=.68, sadness=.12, anger=.06, surprise=.05), .72, ("academic", .71)
                resp = "Nerves before a talk are so common. Let's try a slow breath together, then pick the one point you most want to land."
            elif "burnt" in low or "deadline" in low:
                e, st, tp = _emo(sadness=.42, anger=.27, fear=.17), .88, ("work or career", .83)
                resp = "That's a lot stacked on you at once. What's the single deadline that's weighing on you most?"
            elif "walk" in low or "clearer" in low:
                e, st, tp = _emo(joy=.41, neutral=.33, surprise=.1), .24, ("health or wellness", .58)
                resp = "I'm glad the walk helped. Moving your body can really clear the fog. Want to keep that going, or is this a good place to pause?"
                r["conversation_state"] = "offer_to_close"
            else:
                e, st, tp = _emo(neutral=.5, sadness=.2, fear=.1), .45, ("general", .6)
                resp = "Thanks for sharing that. Tell me a bit more about what's been on your mind."
            r.update(response=resp, emotion=max(e, key=lambda x: x["score"])["label"],
                     emotion_score=max(e, key=lambda x: x["score"])["score"],
                     stress_label="stress" if st > .5 else "not stress", stress_score=max(st, 1 - st),
                     topic=tp[0], topic_score=tp[1],
                     emotion_all=e, stress_all=[{"label": "stress", "score": st}, {"label": "not stress", "score": 1 - st}],
                     topic_all=_topic(*tp))
        history = history if history is not None else []
        if r.get("emotion") is not None:
            r["rag_sources"] = retrieve_knowledge(text, history, r["topic"])
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": r["response"]}]
        return r
else:
    from main import INTRO, process_message


def stress_intensity(results):
    """Probability that the text is 'stressed', regardless of how the model names its labels."""
    if not results:
        return None
    by = {str(r["label"]).lower(): r["score"] for r in results}
    for k, v in by.items():
        if k in ("stress", "stressed", "label_1", "1", "yes"):
            return v
    for k, v in by.items():
        if k in ("label_0", "0", "no") or any(t in k for t in ("not stress", "no stress", "non-stress", "non stress", "not_stress")):
            return 1 - v
    top = max(results, key=lambda r: r["score"])
    return top["score"] if "stress" in str(top["label"]).lower() else None


# ---------- natural text-to-speech (OpenAI) ----------
# (id, label, how it sounds) - the first ones are the most natural-sounding on gpt-4o-mini-tts
TTS_VOICE_INFO = [
    ("marin", "Marin", "Warm and natural, very lifelike"),
    ("cedar", "Cedar", "Calm, grounded and low"),
    ("coral", "Coral", "Warm and friendly"),
    ("sage", "Sage", "Soft, steady and reassuring"),
    ("nova", "Nova", "Bright and gentle"),
    ("shimmer", "Shimmer", "Light and soothing"),
    ("ballad", "Ballad", "Smooth and expressive"),
    ("verse", "Verse", "Calm storyteller"),
    ("ash", "Ash", "Soft and relaxed"),
    ("alloy", "Alloy", "Neutral and clear"),
    ("echo", "Echo", "Calm and even"),
    ("fable", "Fable", "Warm and lyrical"),
    ("onyx", "Onyx", "Deep and steady"),
]
TTS_VOICES = [v[0] for v in TTS_VOICE_INFO]
TTS_LEGACY = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}   # voices tts-1 supports
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_DEFAULT_VOICE = "marin"
# How the voice should sound. The user picks one in the voice settings.
TTS_TONES = {
    "calm": ("Calm and gentle", "Speak in a warm, calm, gentle and unhurried voice, like a supportive friend. "
             "Natural pacing with soft pauses between sentences. Never robotic, flat or overly cheerful."),
    "warm": ("Warm and encouraging", "Speak in a warm, kind and encouraging voice with a light smile in it, "
             "like a caring friend who is genuinely glad to talk. Natural, relaxed pacing; never salesy."),
    "soothing": ("Slow and soothing", "Speak slowly and softly in a soothing, meditative voice. Long, easy pauses "
                 "between sentences. When giving breathing or grounding instructions, pace them gently so the "
                 "listener can follow along."),
}
TTS_DEFAULT_TONE = "calm"
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")           # falls back to whisper-1
DEMO_TTS = DEMO and os.getenv("SERENAI_DEMO_TTS") == "1"      # silent test tone, for UI testing only
TTS_AVAILABLE = DEMO_TTS or (not DEMO and bool(os.getenv("OPENAI_API_KEY")))
STT_AVAILABLE = not DEMO and bool(os.getenv("OPENAI_API_KEY"))
_tts_client = None
_tts_cache = OrderedDict()           # (model, voice, tone, text, name hint) -> (bytes, mimetype); replays are free
_TTS_CACHE_MAX = 48

app = Flask(__name__, static_folder="web", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024     # microphone clips are small; refuse anything huge
SESSIONS = {}


def _openai_client():
    global _tts_client
    if _tts_client is None:
        from openai import OpenAI
        _tts_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _tts_client


def prepare_speech(text):
    """Make text sound right when read aloud: no markdown symbols, links or emoji."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)          # list bullets
    text = re.sub(r"[*_`#>~]+", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", text)   # emoji / pictographs
    text = text.replace("&", " and ")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", ". ", text)).strip()


def _pronunciation_note(text, name, say_as):
    """Tell the TTS model how to say the user's name, if it occurs in `text`."""
    if not name or name.lower() not in text.lower():
        return ""
    if say_as:
        return f' The listener\'s name is "{name}", pronounced "{say_as}". Say it exactly that way.'
    return f' The listener\'s name is "{name}". Pronounce it carefully and clearly, exactly as spelled.'


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.post("/api/start")
def start():
    sid = uuid.uuid4().hex
    # stage: ask_name -> confirm_name -> chat
    SESSIONS[sid] = {"name": None, "say_as": None, "history": [], "named": False,
                     "stage": "ask_name", "pending": None, "tries": 0}
    return jsonify(session_id=sid, intro=INTRO)


def _name_reply(s, name, candidates=None, **extra):
    """Response for a turn that belongs to the name conversation."""
    done = s["stage"] == "chat"
    s["named"] = done
    return jsonify(name_step=True, name_done=done, name=name if done else None, name_pending=None if done else name,
                   candidates=[c for c in (candidates or []) if c != name][:3], **extra)


def _read_back(name):
    spell = spell_out(name)
    return f"I heard {name}, spelled {spell}. Is that right?" if len(spell) > 3 else f"I heard {name}. Is that right?"


def handle_name_step(s, text, alts):
    """
    The name conversation: ask -> read back the spelling -> confirm. Speech recognition often garbles names,
    so the name is never used until the user has confirmed it (or typed it themselves).
    Returns a Flask response, or None when the user has moved on and the message should be handled normally.
    """
    if s["stage"] == "ask_name":
        cands = name_candidates([text] + alts)
        if not cands:
            s["tries"] += 1
            if s["tries"] >= 2:
                s["stage"] = "chat"
                return _name_reply(s, None, response="That's okay, we can skip names. How are you feeling today?")
            return _name_reply(s, None, response="Sorry, I didn't catch your name. Could you say it again, "
                                                 "or spell it for me letter by letter?")
        s.update(stage="confirm_name", pending=cands[0], tries=0)
        return _name_reply(s, cands[0], cands, response=f"Nice to meet you. {_read_back(cands[0])} "
                                                         "If not, just spell it for me.")

    # confirm_name
    pending = s["pending"]
    spelled = spelled_name(text)
    if is_yes(text) and not spelled:
        new = pending
    elif spelled:
        new = spelled
    else:
        found = name_candidates([text] + alts)
        new = found[0] if found else None
        if new is None and not is_no(text):
            s.update(stage="chat", name=pending, named=True, pending=None)     # user moved on: keep the name
            return None
    if new is None:                                                              # a plain "no"
        s["tries"] += 1
        if s["tries"] >= 2:
            s.update(stage="chat", pending=None)
            return _name_reply(s, None, response="No problem, I won't use a name. How are you feeling today?")
        return _name_reply(s, pending, response="Sorry about that. Could you spell your name for me, "
                                                "one letter at a time?")
    if new.lower() == pending.lower() or s["tries"] >= 2:                         # confirmed (or tried enough)
        s.update(stage="chat", name=new, pending=None)
        return _name_reply(s, new, response=f"Great, {new}. How are you feeling today?")
    s.update(pending=new, tries=s["tries"] + 1)
    return _name_reply(s, new, response=f"Got it. {_read_back(new)}")


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    s = SESSIONS.get(data.get("session_id"))
    if s is None:
        return jsonify(error="session_expired"), 410

    # Optional: skip the name question (used by the quick-simulation buttons)
    if data.get("skip_name") and s["stage"] != "chat":
        s["stage"] = "chat"
        return _name_reply(s, None, response="That's okay. How are you feeling today?")

    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(response="I didn't catch that. Please try again.", name_step=False, safety_state=None)

    # The first messages are the name conversation (ask, read back the spelling, confirm)
    if s["stage"] != "chat":
        alts = [a for a in (data.get("alts") or []) if isinstance(a, str)][:6]
        out = handle_name_step(s, text, alts)
        if out is not None:
            return out

    t0 = time.perf_counter()
    try:
        r = process_message(text=text, user_name=s["name"], history=s["history"])
    except Exception as e:  # keep the UI alive if a model/API call fails
        app.logger.exception("pipeline error")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    analysis = None
    if r.get("emotion") is not None:
        analysis = {
            "stress": {"intensity": stress_intensity(r.get("stress_all")),
                       "label": r["stress_label"], "score": r["stress_score"]},
            "emotions": sorted(r.get("emotion_all") or [{"label": r["emotion"], "score": r["emotion_score"]}],
                               key=lambda x: -x["score"]),
            "topics": sorted(r.get("topic_all") or [{"label": r["topic"], "score": r["topic_score"]}],
                             key=lambda x: -x["score"]),
        }
    signals = {
        "user_message": text, "user_name": s["name"],
        "safety_state": r["safety_state"], "conversation_state": r["conversation_state"],
        "emotion": r["emotion"], "emotion_score": r["emotion_score"],
        "stress_label": r["stress_label"], "stress_score": r["stress_score"],
        "topic": r["topic"], "topic_score": r["topic_score"],
        "history_turns": len(s["history"]) // 2,
        "rag_passages": [f"{k['title']} / {k['section']}" for k in (r.get("rag_sources") or [])],
    }
    rag = None
    if "rag_sources" in r:   # absent on safety / closing turns, where retrieval is not used
        rag = [{"title": k["title"], "section": k["section"], "score": k["score"],
                "snippet": k["text"].split(". ", 1)[-1][:190]} for k in (r["rag_sources"] or [])]
    return jsonify(response=r["response"], analysis=analysis, signals=signals, rag=rag, name_step=False,
                   safety_state=r["safety_state"], conversation_state=r["conversation_state"],
                   should_close=r["should_close"], elapsed_ms=elapsed_ms)


@app.post("/api/name")
def set_name():
    """Let the user correct the name SerenAI uses (e.g. after a speech-recognition mistake)."""
    d = request.get_json(force=True)
    s = SESSIONS.get(d.get("session_id"))
    if s is None:
        return jsonify(error="session_expired"), 410
    name = clean_typed_name(d.get("name"))
    say_as = " ".join((d.get("say_as") or "").split())[:40] or None
    if say_as:
        say_as = "".join(c for c in say_as if c.isalpha() or c in " '’-.") or None
    s.update(name=name, say_as=say_as if name else None, named=True, stage="chat", pending=None)
    reply = (f"Thanks, {name}. How are you feeling today?" if name else "Okay, I won't use a name.") \
        if d.get("announce") else None
    return jsonify(name=name, say_as=s["say_as"], response=reply)


@app.get("/api/config")
def config():
    return jsonify(tts=TTS_AVAILABLE, stt=STT_AVAILABLE, rag=rag_info(),
                   voices=[{"id": i, "label": l, "desc": desc} for i, l, desc in TTS_VOICE_INFO],
                   tones=[{"id": k, "label": v[0]} for k, v in TTS_TONES.items()],
                   defaults={"voice": TTS_DEFAULT_VOICE, "tone": TTS_DEFAULT_TONE})


@app.post("/api/tts")
def tts():
    if not TTS_AVAILABLE:
        return jsonify(error="tts_unavailable"), 503
    d = request.get_json(force=True)
    text = prepare_speech((d.get("text") or ""))[:3500]
    voice = d.get("voice") if d.get("voice") in TTS_VOICES else TTS_DEFAULT_VOICE
    tone = d.get("tone") if d.get("tone") in TTS_TONES else TTS_DEFAULT_TONE
    if not text:
        return jsonify(error="empty"), 400
    # Speed is applied in the browser (playbackRate): gpt-4o-mini-tts ignores the API's speed setting.

    s = SESSIONS.get(d.get("session_id")) or {}
    name, say_as = s.get("name"), s.get("say_as")
    note = _pronunciation_note(text, name, say_as)

    if DEMO_TTS:  # 0.4 s of silence
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(struct.pack("<h", 0) * 3200)
        return Response(buf.getvalue(), mimetype="audio/wav")

    key = (TTS_MODEL, voice, tone, text, note)
    if key in _tts_cache:
        _tts_cache.move_to_end(key)
        audio, mime = _tts_cache[key]
        return Response(audio, mimetype=mime)

    client = _openai_client()
    try:
        r = client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text, response_format="mp3",
                                       instructions=TTS_TONES[tone][1] + note)
    except Exception:
        app.logger.warning("TTS model %s failed, falling back to tts-1", TTS_MODEL, exc_info=True)
        try:   # tts-1 has no style instructions, so respell the name phonetically in the text itself
            fallback = text
            if name and say_as:
                fallback = re.sub(re.escape(name), say_as, text, flags=re.I)
            r = client.audio.speech.create(model="tts-1-hd", voice=voice if voice in TTS_LEGACY else "nova",
                                           input=fallback)
        except Exception as e:
            app.logger.exception("TTS failed")
            return jsonify(error=f"{type(e).__name__}: {e}"), 502
    audio = r.content
    _tts_cache[key] = (audio, "audio/mpeg")
    while len(_tts_cache) > _TTS_CACHE_MAX:
        _tts_cache.popitem(last=False)
    return Response(audio, mimetype="audio/mpeg")


@app.post("/api/stt")
def stt():
    """
    Accurate speech recognition for the browser's recorded clip (multipart: audio, session_id, context, lang).
    Much better than the browser's built-in recogniser on names, because the model is primed with what to expect.
    """
    if not STT_AVAILABLE:
        return jsonify(error="stt_unavailable"), 503
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="no_audio"), 400
    data = f.read()
    if len(data) < 1200:                                     # essentially silence / a click
        return jsonify(text="")
    s = SESSIONS.get(request.form.get("session_id")) or {}
    in_name_step = request.form.get("context") == "name" or s.get("stage") in ("ask_name", "confirm_name")
    if in_name_step:
        prompt = ("The speaker is telling an assistant their first name, for example \"My name is Saja.\" "
                  "or spelling it letter by letter. Names can be from any culture, such as Saja, Aisha, Mohammed, "
                  "Yusuf, Priya, Siobhan, Nguyen, Oluwaseun, Dmitri, Zoë. Transcribe the name with its most likely spelling.")
    else:
        prompt = "A calm conversation with a stress-support assistant called SerenAI."
        if s.get("name"):
            prompt += f" The user's name is {s['name']}."
    lang = (request.form.get("lang") or "").lower()[:2]
    kwargs = {"language": lang} if re.fullmatch(r"[a-z]{2}", lang) else {}
    name = f.filename or "speech.webm"
    clip = (name, data, f.mimetype or "audio/webm")
    client = _openai_client()
    for model in dict.fromkeys([STT_MODEL, "whisper-1"]):
        try:
            r = client.audio.transcriptions.create(model=model, file=clip, prompt=prompt, **kwargs)
            return jsonify(text=(r.text or "").strip())
        except Exception as e:
            app.logger.warning("STT model %s failed: %s", model, e)
    return jsonify(error="stt_failed"), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
