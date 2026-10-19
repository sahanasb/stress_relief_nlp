import re

from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic

from agent.safety import check_safety
from agent.stress_agent import generate_response
from agent.conversation_state import detect_conversation_state
from agent.safety_agent import generate_safety_response

from voice.speech_to_text import listen
from voice.text_to_speech import speak


# -------------------------
# Short-term memory
# -------------------------
conversation_history = []


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
                flags=re.IGNORECASE
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
        flags=re.IGNORECASE
    )

    return name.strip().title() if name.strip() else None


# -------------------------
# Core NLP / agent pipeline
# -------------------------
def process_message(
    text,
    user_name=None,
    history=None
):
    """
    Process one user message through SerenAI's pipeline.

    This reusable application logic can later be called
    from the voice application or another interface,
    such as Gradio.
    """

    if history is None:
        history = conversation_history

    text = text.strip()

    if not text:
        return {
            "response": "I didn't catch that. Please try again.",
            "safety_state": None,
            "conversation_state": None,
            "emotion": None,
            "emotion_score": None,
            "stress_label": None,
            "stress_score": None,
            "topic": None,
            "topic_score": None,
            "should_close": False,
        }

    # -------------------------
    # 1. Safety check
    # -------------------------
    safety_state = check_safety(
        user_message=text,
        conversation_history=history,
    )

    # -------------------------
    # 2. Handle safety concerns
    # -------------------------
    if safety_state in [
        "concern",
        "immediate_risk"
    ]:

        response = generate_safety_response(
            user_message=text,
            safety_state=safety_state,
            user_name=user_name,
            conversation_history=history,
        )

        # Save safety interaction
        history.append(
            {
                "role": "user",
                "content": text
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        return {
            "response": response,
            "safety_state": safety_state,
            "conversation_state": None,
            "emotion": None,
            "emotion_score": None,
            "stress_label": None,
            "stress_score": None,
            "topic": None,
            "topic_score": None,
            "should_close": False,
        }

    # -------------------------
    # 3. Detect conversation state
    # -------------------------
    conversation_state = detect_conversation_state(
        user_message=text,
        conversation_history=history,
    )

    # -------------------------
    # 4. Close conversation
    # -------------------------
    if conversation_state == "close":

        if user_name:
            closing_message = (
                f"I'm glad we could talk, {user_name}. "
                "Take care of yourself, and you're welcome "
                "to come back whenever you'd like."
            )

        else:
            closing_message = (
                "I'm glad we could talk. "
                "Take care of yourself, and you're welcome "
                "to come back whenever you'd like."
            )

        # Save final user message
        history.append(
            {
                "role": "user",
                "content": text
            }
        )

        # Save SerenAI's closing response
        history.append(
            {
                "role": "assistant",
                "content": closing_message
            }
        )

        return {
            "response": closing_message,
            "safety_state": safety_state,
            "conversation_state": conversation_state,
            "emotion": None,
            "emotion_score": None,
            "stress_label": None,
            "stress_score": None,
            "topic": None,
            "topic_score": None,
            "should_close": True,
        }

    # -------------------------
    # 5. Emotion detection
    # -------------------------
    emotion_results = detect_emotion(text)

    top_emotion = max(
        emotion_results,
        key=lambda x: x["score"]
    )

    emotion = top_emotion["label"]
    emotion_score = top_emotion["score"]

    # -------------------------
    # 6. Stress detection
    # -------------------------
    stress_results = detect_stress(text)

    top_stress = max(
        stress_results,
        key=lambda x: x["score"]
    )

    stress_label = top_stress["label"]
    stress_score = top_stress["score"]

    # -------------------------
    # 7. Topic detection
    # -------------------------
    topic_result = detect_topic(text)

    topic = topic_result["labels"][0]
    topic_score = topic_result["scores"][0]

    # -------------------------
    # 8. Generate AI response
    # -------------------------
    response = generate_response(
        user_message=text,
        emotion=emotion,
        emotion_score=emotion_score,
        stress_label=stress_label,
        stress_score=stress_score,
        topic=topic,
        topic_score=topic_score,
        user_name=user_name,
        conversation_history=history,
        conversation_state=conversation_state,
    )

    # -------------------------
    # 9. Save to short-term memory
    # -------------------------
    history.append(
        {
            "role": "user",
            "content": text
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return {
        "response": response,
        "safety_state": safety_state,
        "conversation_state": conversation_state,
        "emotion": emotion,
        "emotion_score": emotion_score,
        "stress_label": stress_label,
        "stress_score": stress_score,
        "topic": topic,
        "topic_score": topic_score,
        "should_close": False,
    }


# -------------------------
# Current voice / terminal app
# -------------------------
def run_voice_app():
    """
    Run SerenAI using the microphone
    and text-to-speech flow.
    """

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

    # Extract only the actual name
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

        # For keyboard testing, replace:
        #
        # text = listen()
        #
        # with:
        #
        # text = input("\nYou: ")

        print(f"\nYou: {text}")

        # -------------------------
        # Empty input
        # -------------------------
        if not text or not text.strip():

            message = (
                "I didn't catch that. "
                "Please try again."
            )

            print(f"\nSerenAI: {message}")
            speak(message)

            continue

        # -------------------------
        # Manual quit option
        # -------------------------
        if text.strip().lower() == "quit":

            if user_name:
                goodbye = (
                    f"Take care, {user_name}. Goodbye!"
                )

            else:
                goodbye = "Take care. Goodbye!"

            print(f"\nSerenAI: {goodbye}")
            speak(goodbye)

            break

        # -------------------------
        # Process user message
        # -------------------------
        result = process_message(
            text=text,
            user_name=user_name,
        )

        # -------------------------
        # Display safety state
        # -------------------------
        print(
            f"\nSafety State: "
            f"{result['safety_state']}"
        )

        # -------------------------
        # Display conversation state
        # -------------------------
        if result["conversation_state"] is not None:

            print(
                f"Conversation State: "
                f"{result['conversation_state']}"
            )

        # -------------------------
        # Display NLP analysis
        # -------------------------
        if result["emotion"] is not None:

            print("\n--- NLP Analysis ---")

            print(
                f"Emotion: "
                f"{result['emotion']} "
                f"({result['emotion_score']:.2%})"
            )

            print(
                f"Stress: "
                f"{result['stress_label']} "
                f"({result['stress_score']:.2%})"
            )

            print(
                f"Topic: "
                f"{result['topic']} "
                f"({result['topic_score']:.2%})"
            )

        # -------------------------
        # Display response
        # -------------------------
        if result["safety_state"] in [
            "concern",
            "immediate_risk"
        ]:

            print("\nSafety Support Agent:")

        else:

            print("\nSerenAI:")

        print(result["response"])

        # -------------------------
        # Speak response
        # -------------------------
        speak(result["response"])

        # -------------------------
        # End conversation
        # -------------------------
        if result["should_close"]:
            break


# -------------------------
# Start application
# -------------------------
if __name__ == "__main__":
    run_voice_app()