"""Terminal-mode speech to text (the web app uses /api/stt in app.py instead)."""
import os
import tempfile
import queue

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")      # falls back to whisper-1

# Priming the recogniser with what to expect makes it far better at names.
NAME_PROMPT = ("The speaker is telling an assistant their first name, for example \"My name is Saja.\" "
               "or spelling it letter by letter. Names can be from any culture, such as Saja, Aisha, Mohammed, "
               "Yusuf, Priya, Siobhan, Nguyen, Oluwaseun, Dmitri, Zoë. Transcribe the name with its most likely spelling.")
CHAT_PROMPT = "A calm conversation with a stress-support assistant called SerenAI."


def record_audio(
    sample_rate=16000,
    silence_duration=1.5,
    silence_threshold=None,
    max_duration=30
):
    """
    Record from the microphone.

    Recording stops when the user has started speaking and then been silent for `silence_duration` seconds.
    The speech threshold adapts to the room: the first half second measures the background noise.
    `max_duration` prevents recording forever.
    """

    audio_queue = queue.Queue()
    audio_chunks = []

    has_started_speaking = False
    silent_chunks = 0

    block_duration = 0.1
    block_size = int(sample_rate * block_duration)
    silence_chunks_needed = int(silence_duration / block_duration)
    max_chunks = int(max_duration / block_duration)
    calibration_chunks = 5
    noise_levels = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    print("\nListening... Speak now.")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=block_size,
        callback=callback
    ):
        for i in range(max_chunks):
            chunk = audio_queue.get()
            audio_chunks.append(chunk)

            volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if silence_threshold is None and i < calibration_chunks:
                noise_levels.append(volume)
                continue
            threshold = silence_threshold if silence_threshold is not None else max(
                400.0, float(np.mean(noise_levels)) * 3)

            if volume > threshold:
                has_started_speaking = True
                silent_chunks = 0
            elif has_started_speaking:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    break

    print("Recording finished.")

    if not audio_chunks or not has_started_speaking:
        return None

    recording = np.concatenate(audio_chunks, axis=0)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    write(temp_file.name, sample_rate, recording)
    return temp_file.name


def transcribe_audio(audio_file, prompt=None):
    """Convert recorded speech into text. `prompt` primes the model with what it is about to hear."""
    last_error = None
    for model in dict.fromkeys([STT_MODEL, "whisper-1"]):
        try:
            with open(audio_file, "rb") as file:
                kwargs = {"prompt": prompt} if prompt else {}
                return client.audio.transcriptions.create(model=model, file=file, **kwargs).text.strip()
        except Exception as e:
            last_error = e
    raise last_error


def listen(name_mode=False):
    """
    Listen until the user stops speaking, then return the transcription.
    Pass name_mode=True when the user is about to say their name.
    """

    audio_file = record_audio(silence_duration=1.8 if name_mode else 1.5)

    if audio_file is None:
        return ""

    try:
        return transcribe_audio(audio_file, NAME_PROMPT if name_mode else CHAT_PROMPT)
    finally:
        os.remove(audio_file)


if __name__ == "__main__":

    text = listen()

    print("\nYou said:")
    print(text)
