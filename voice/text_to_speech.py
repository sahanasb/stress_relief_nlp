import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def speak(text):
    """
    Convert text to speech and play it on the Mac.
    """

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    )

    audio_path = temp_file.name
    temp_file.close()

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        response.stream_to_file(audio_path)

        # afplay is built into macOS
        os.system(f'afplay "{audio_path}"')

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# Temporary test
if __name__ == "__main__":
    speak(
        "Hi. I'm your stress relief companion. "
        "How are you feeling today?"
    )