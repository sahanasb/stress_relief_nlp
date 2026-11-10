"""Terminal-mode text to speech (the web app uses /api/tts in app.py instead)."""
import os
import platform
import subprocess
import tempfile

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "marin")   # marin, cedar, coral, sage, nova, shimmer, ballad, verse, ash, alloy, ...
TTS_STYLE = ("Speak in a warm, calm, gentle and unhurried voice, like a supportive friend. "
             "Natural pacing with soft pauses. Never robotic, flat or overly cheerful.")


def _play(path):
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["afplay", path], check=False)
    elif system == "Windows":
        # Windows Media Player COM object plays mp3 without extra packages
        ps = ("Add-Type -AssemblyName presentationCore; $p = New-Object System.Windows.Media.MediaPlayer; "
              f"$p.Open([uri]'{path}'); $p.Play(); Start-Sleep -Milliseconds 400; "
              "while ($p.NaturalDuration.HasTimeSpan -eq $false) { Start-Sleep -Milliseconds 100 }; "
              "Start-Sleep -Seconds ([math]::Ceiling($p.NaturalDuration.TimeSpan.TotalSeconds)); $p.Close()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
    else:
        for player in (["mpg123", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"], ["play", "-q"]):
            try:
                subprocess.run(player + [path], check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        print("No audio player found (install mpg123 or ffmpeg).")


def speak(text, voice=None, name=None, say_as=None):
    """
    Convert text to speech and play it.

    `name` (and optionally `say_as`, a phonetic spelling) tells the voice how to pronounce the user's name.
    """
    note = ""
    if name and name.lower() in text.lower():
        note = (f' The listener\'s name is "{name}", pronounced "{say_as}". Say it exactly that way.' if say_as
                else f' The listener\'s name is "{name}". Pronounce it carefully, exactly as spelled.')

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio_path = temp_file.name
    temp_file.close()

    try:
        try:
            response = client.audio.speech.create(
                model=TTS_MODEL, voice=voice or TTS_VOICE, input=text, instructions=TTS_STYLE + note)
        except Exception:      # model/voice not available on this account: fall back to the classic model
            response = client.audio.speech.create(model="tts-1-hd", voice="nova", input=text)
        response.write_to_file(audio_path)
        _play(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# Temporary test
if __name__ == "__main__":
    speak("Hi. I'm your stress relief companion. How are you feeling today?")
