import os
import tempfile
import queue

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def record_audio(
    sample_rate=16000,
    silence_duration=1.5,
    silence_threshold=500,
    max_duration=30
):
    """
    Record from the microphone.

    Recording stops when:
    - the user has started speaking, and
    - silence is detected for silence_duration seconds.

    max_duration prevents recording forever.
    """

    audio_queue = queue.Queue()
    audio_chunks = []

    has_started_speaking = False
    silent_chunks = 0

    block_duration = 0.1
    block_size = int(sample_rate * block_duration)

    silence_chunks_needed = int(
        silence_duration / block_duration
    )

    max_chunks = int(
        max_duration / block_duration
    )

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

        for _ in range(max_chunks):

            chunk = audio_queue.get()
            audio_chunks.append(chunk)

            # Measure volume
            volume = np.sqrt(
                np.mean(
                    chunk.astype(np.float32) ** 2
                )
            )

            # Detect speech
            if volume > silence_threshold:

                has_started_speaking = True
                silent_chunks = 0

            elif has_started_speaking:

                silent_chunks += 1

                if silent_chunks >= silence_chunks_needed:
                    break

    print("Recording finished.")

    if not audio_chunks:
        return None

    recording = np.concatenate(
        audio_chunks,
        axis=0
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    write(
        temp_file.name,
        sample_rate,
        recording
    )

    return temp_file.name


def transcribe_audio(audio_file):
    """
    Convert recorded speech into text.
    """

    with open(audio_file, "rb") as file:

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )

    return transcription.text


def listen():
    """
    Listen until the user stops speaking,
    then return the transcription.
    """

    audio_file = record_audio()

    if audio_file is None:
        return ""

    try:
        text = transcribe_audio(audio_file)

    finally:
        os.remove(audio_file)

    return text


if __name__ == "__main__":

    text = listen()

    print("\nYou said:")
    print(text)