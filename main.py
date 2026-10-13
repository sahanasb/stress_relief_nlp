from agent.safety import check_safety
from agent.stress_agent import generate_response
from agent.conversation_state import detect_conversation_state
from agent.safety import check_safety
from agent.safety_agent import generate_safety_response
from voice.speech_to_text import listen
from voice.text_to_speech import speak
from nlp.openai_analyzer import analyze_message

# from voice.speech_to_text import listen
from voice.text_to_speech import speak

conversation_history = []
VOICE_OUTPUT = False


def output_response(message):
    print(f"\nSerenAI: {message}")

    if VOICE_OUTPUT:
        try:
            speak(message)
        except Exception as exc:
            print(f"The voice playback failed, but the text reply is still readable: {type(exc).__name__}")

# -------------------------
# Short-term memory
# -------------------------
conversation_history = []
# Agent introduction
intro = (
    "Hi, I'm SerenAI, your stress support assistant. "
    "I'm here to listen and help you work through everyday stress. "
    "How are you feeling today?"
)

print(f"\nSerenAI: {intro}")
speak(intro)



while True:

    text = listen().strip()
    #text = input("\nYou: ").strip()
    print(f"\nYou: {text}")

    if not text:
        print("I didn't catch that. Please try again.")
        continue

    # -------------------------
    # Exit manually
    # -------------------------
    if text.lower() == "quit":
        print("\nSerenAI: Take care. Goodbye!")
        break

    conversation_state = detect_conversation_state(
            text, conversation_history
    )
    print(f"Conversation State: {conversation_state}")

    if conversation_state == "close":
        output_response("I'm glad we could talk. Take care of yourself, "
                    "and you're welcome to come back whenever you'd like.")
        break

    # -------------------------
    # Safety check
    # -------------------------
    safety_state = check_safety(
        user_message=text,
        conversation_history=conversation_history
    )
    print(f"\nSafety State: {safety_state}")

    # -------------------------
    # Handle safety concerns
    # -------------------------
    if safety_state in ["concern", "immediate_risk"]:

        response = generate_safety_response(
            user_message=text,
            safety_state=safety_state,
            conversation_history=conversation_history
        )
        print("\nSerenAI:")
        print(response)

    else:
        analysis = analyze_message(
            user_message=text,
            conversation_history=conversation_history,
        )
        print(f"NLP Analysis: {analysis}")

        response = generate_response(
            user_message=text,
            analysis=analysis,
            conversation_history=conversation_history,
            conversation_state=conversation_state,
        )

        # Save this interaction to conversation memory
    conversation_history.extend([
        {
            "role": "user",
            "content": text
        },
        {
            "role": "assistant",
            "content": response
        }
    ])

    output_response(response)