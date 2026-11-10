import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

load_dotenv()

app = FastAPI()


@app.get("/")
def home():
    return {"message": "SerenAI Twilio server is running"}


@app.post("/voice")
async def incoming_call(request: Request):
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

    if not auth_token or not public_base_url:
        raise HTTPException(
            status_code=500,
            detail="Twilio configuration is missing"
        )

    # Validate the incoming request using the public URL
    signature = request.headers.get("X-Twilio-Signature", "")
    form_data = dict(await request.form())

    webhook_url = f"{public_base_url}{request.url.path}"

    validator = RequestValidator(auth_token)

    if not validator.validate(webhook_url, form_data, signature):
        raise HTTPException(
            status_code=403,
            detail="Invalid Twilio signature"
        )

    response = VoiceResponse()

    response.say(
        "Hi, I'm SerenAI, your AI stress support assistant. "
        "Thanks for calling. "
        "We're currently testing our phone assistant.",
        voice="alice"
    )

    response.hangup()

    return Response(
        content=str(response),
        media_type="application/xml"
    )