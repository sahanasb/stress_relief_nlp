"""Local microphone/text-to-speech interface for SerenAI."""
import re

from agent.pipeline import process_message
from voice.speech_to_text import listen
from voice.text_to_speech import speak


# -------------------------
# Agent introduction
# -------------------------
INTRO = (
    "Hi, I'm SerenAI, your stress support assistant. "
    "I'm here to listen and help you work through everyday stress. "
    "Before we get started, what should I call you?"
)


# -------------------------
# Name extraction
# -------------------------
def extract_name(text):
    """Extract a user's preferred name from natural spoken introductions."""

    if not text:
        return None

    text = text.strip()

    patterns = [
        r"\bmy name is\s*,?\s*(.+)",
        r"\byou can call me\s*,?\s*(.+)",
        r"\bcall me\s*,?\s*(.+)",
        r"\bi am\s*,?\s*(.+)",
        r"\bi'm\s*,?\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            name = match.group(1)

            # Remove common speech fillers from the beginning
            name = re.sub(
                r"^(uh|um|umm|erm|hmm)[,\s]+",
                "",
                name,
                flags=re.IGNORECASE,
            )

            # Clean punctuation
            name = name.strip().strip(".,!?")

            if name:
                return name.title()

    # User may have simply said their name
    name = text.strip().strip(".,!?")

    # Remove greetings/fillers from the beginning
    name = re.sub(
        r"^(hi|hello|hey|uh|um|umm)[,\s]+",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name.strip().title() if name.strip() else None


# -------------------------
# Current voice / terminal app
# -------------------------
def run_voice_app():
    """Run SerenAI using the microphone and text-to-speech flow."""

    # Independent short-term memory for this session
    conversation_history = []

    # -------------------------
    # Introduction
    # -------------------------
    print(f"\nSerenAI: {INTRO}")
    speak(INTRO)

    # -------------------------
    # Ask for user's name
    # -------------------------
    name_input = listen()

    print(f"\nYou: {name_input}")

    user_name = extract_name(name_input)

    if user_name:
        greeting = (
            f"Nice to meet you, {user_name}. "
            "How are you feeling today?"
        )
    else:
        greeting = (
            "That's okay. "
            "How are you feeling today?"
        )

    print(f"\nSerenAI: {greeting}")
    speak(greeting)

    # -------------------------
    # Main conversation loop
    # -------------------------
    while True:
        text = listen()

        # For keyboard testing, replace the line above with:
        # text = input("\nYou: ")

        print(f"\nYou: {text}")

        if not text or not text.strip():
            message = "I didn't catch that. Please try again."
            print(f"\nSerenAI: {message}")
            speak(message)
            continue

        # Manual quit option
        if text.strip().lower() == "quit":
            if user_name:
                goodbye = f"Take care, {user_name}. Goodbye!"
            else:
                goodbye = "Take care. Goodbye!"

            print(f"\nSerenAI: {goodbye}")
            speak(goodbye)
            break

        # Process user message
        result = process_message(
            text=text,
            user_name=user_name,
            history=conversation_history,
        )

        print(f"\nSafety State: {result['safety_state']}")

        if result["conversation_state"] is not None:
            print(
                f"Conversation State: "
                f"{result['conversation_state']}"
            )

        # Show NLP analysis only when the normal pipeline ran
        if result["emotion"] is not None:
            print("\n--- NLP Analysis ---")
            print(
                f"Emotion: {result['emotion']} "
                f"({result['emotion_score']:.2%})"
            )
            print(
                f"Stress: {result['stress_label']} "
                f"({result['stress_score']:.2%})"
            )
            print(
                f"Topic: {result['topic']} "
                f"({result['topic_score']:.2%})"
            )

        if result["safety_state"] in ["concern", "immediate_risk"]:
            print("\nSafety Support Agent:")
        else:
            print("\nSerenAI:")

        print(result["response"])
        speak(result["response"])

        if result["should_close"]:
            break


# -------------------------
# Start application
# -------------------------
if __name__ == "__main__":
    run_voice_app()
