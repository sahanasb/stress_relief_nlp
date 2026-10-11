from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic
from agent.safety import check_safety
from agent.stress_agent import generate_response
from agent.conversation_state import detect_conversation_state
from agent.safety import check_safety
from agent.safety_agent import generate_safety_response
from voice.speech_to_text import listen
from voice.text_to_speech import speak


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

    text = listen()
    # text = input("\nYou: ")
    print(f"\nYou: {text}")

    if not text.strip():
        print("I didn't catch that. Please try again.")
        continue

    # -------------------------
    # Exit manually
    # -------------------------
    if text.lower() == "quit":
        print("\nStress Relief Agent: Take care. Goodbye!")
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

        print("\nSafety Support Agent:")
        print(response)
        speak(response)

        # Save this interaction to conversation memory
        conversation_history.append(
            {
                "role": "user",
                "content": text
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Skip the normal stress-relief pipeline
        continue

    # -------------------------
    # 1. Detect conversation state
    # -------------------------
    conversation_state = detect_conversation_state(
        user_message=text,
        conversation_history=conversation_history
    )

    print(f"\nConversation State: {conversation_state}")


    # -------------------------
    # 2. Close conversation
    # -------------------------
    if conversation_state == "close":

        closing_message = (
            "I'm glad we could talk. Take care of yourself, "
            "and you're welcome to come back whenever you'd like."
        )

        print(f"\nStress Relief Agent: {closing_message}")

        speak(closing_message)

        break


    # -------------------------
    # 3. Emotion detection
    # -------------------------
    emotion_results = detect_emotion(text)

    top_emotion = max(
        emotion_results,
        key=lambda x: x["score"]
    )

    emotion = top_emotion["label"]
    emotion_score = top_emotion["score"]


    # -------------------------
    # 4. Stress detection
    # -------------------------
    stress_results = detect_stress(text)

    top_stress = max(
        stress_results,
        key=lambda x: x["score"]
    )

    stress_label = top_stress["label"]
    stress_score = top_stress["score"]


    # -------------------------
    # 5. Topic detection
    # -------------------------
    topic_result = detect_topic(text)

    topic = topic_result["labels"][0]
    topic_score = topic_result["scores"][0]


    # -------------------------
    # 6. Show NLP analysis
    # -------------------------
    print("\n--- NLP Analysis ---")
    print(f"Emotion: {emotion} ({emotion_score:.2%})")
    print(f"Stress: {stress_label} ({stress_score:.2%})")
    print(f"Topic: {topic} ({topic_score:.2%})")


    # -------------------------
    # 7. Generate AI response
    # -------------------------
    response = generate_response(
        user_message=text,
        emotion=emotion,
        emotion_score=emotion_score,
        stress_label=stress_label,
        stress_score=stress_score,
        topic=topic,
        topic_score=topic_score,
        conversation_history=conversation_history,
        conversation_state=conversation_state
    )

    print("\nStress Relief Agent:")
    print(response)
    speak(response)

    # -------------------------
    # 8. Save this turn to memory
    # -------------------------
    conversation_history.append(
        {
            "role": "user",
            "content": text
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )