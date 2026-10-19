"""
WE web UI.

Run:   python app.py                  -> http://127.0.0.1:5000
Demo:  WE_DEMO=1 python app.py   (no models / API keys; canned scenarios to preview the UI)
"""
import os
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory

DEMO = os.getenv("SerenAI_DEMO") == "1"

TOPICS = ["academic", "work or career", "relationships", "family",
          "financial", "health or wellness", "general"]

if DEMO:
    INTRO = (
        "Hi, I'm SerenAI, your stress support assistant. "
        "I'm here to listen and help you work through everyday stress. "
        "Before we get started, what should I call you?"
    )

    def extract_name(text):
        return (text or "").strip().strip(".,!?").split()[-1].title() or None

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
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": r["response"]}]
        return r
else:
    from main import INTRO, extract_name, process_message


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


app = Flask(__name__, static_folder="web", static_url_path="/static")
SESSIONS = {}


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.post("/api/start")
def start():
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"name": None, "history": [], "named": False}
    return jsonify(session_id=sid, intro=INTRO)


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    s = SESSIONS.get(data.get("session_id"))
    if s is None:
        return jsonify(error="session_expired"), 410

    # Optional: skip the name question (used by the quick-simulation buttons)
    if data.get("skip_name") and not s["named"]:
        s["named"] = True
        return jsonify(response="That's okay. How are you feeling today?", name_step=True)

    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(response="I didn't catch that. Please try again.", name_step=False, safety_state=None)

    # The first message is the user's name
    if not s["named"]:
        s["named"] = True
        s["name"] = extract_name(text)
        reply = (f"Nice to meet you, {s['name']}. How are you feeling today?" if s["name"]
                 else "That's okay. How are you feeling today?")
        return jsonify(response=reply, name=s["name"], name_step=True)

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
    }
    return jsonify(response=r["response"], analysis=analysis, signals=signals, name_step=False,
                   safety_state=r["safety_state"], conversation_state=r["conversation_state"],
                   should_close=r["should_close"], elapsed_ms=elapsed_ms)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
