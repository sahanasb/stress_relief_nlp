"""Reusable SerenAI text-processing pipeline for voice and future Twilio clients."""
import time
from concurrent.futures import ThreadPoolExecutor

from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic
from agent.safety import check_safety
from agent.stress_agent import generate_response
from agent.conversation_state import detect_conversation_state
from agent.safety_agent import generate_safety_response


# -------------------------
# Core NLP / agent pipeline
# -------------------------
def process_message(
    text,
    user_name=None,
    history=None,
):
    """
    Process one user message through SerenAI's pipeline.

    Safety is checked first. For normal messages, conversation-state
    classification runs concurrently with sequential NLP analyses.
    """

    total_start = time.perf_counter()

    if history is None:
        history = []

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
    safety_start = time.perf_counter()

    safety_state = check_safety(
        user_message=text,
        conversation_history=history,
    )

    print(
        f"Safety latency: "
        f"{time.perf_counter() - safety_start:.2f}s"
    )

    # -------------------------
    # 2. Handle safety concerns
    # -------------------------
    if safety_state in ["concern", "immediate_risk"]:
        safety_response_start = time.perf_counter()

        response = generate_safety_response(
            user_message=text,
            safety_state=safety_state,
            user_name=user_name,
            conversation_history=history,
        )

        print(
            f"Safety response latency: "
            f"{time.perf_counter() - safety_response_start:.2f}s"
        )

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})

        print(
            f"Total processing latency: "
            f"{time.perf_counter() - total_start:.2f}s"
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
    # 3. Run conversation state in parallel with sequential NLP analysis
    # -------------------------
    parallel_start = time.perf_counter()

    # Only the OpenAI conversation-state call runs in a worker thread.
    # The local Hugging Face models remain sequential to avoid resource
    # contention/hanging on shared CPU/MPS resources.
    with ThreadPoolExecutor(max_workers=1) as executor:
        state_future = executor.submit(
            detect_conversation_state,
            user_message=text,
            conversation_history=history,
        )

        emotion_results = detect_emotion(text)
        stress_results = detect_stress(text)
        topic_result = detect_topic(text)

        conversation_state = state_future.result()

    print(
        f"Parallel State + Sequential NLP latency: "
        f"{time.perf_counter() - parallel_start:.2f}s"
    )

    # -------------------------
    # 4. Close conversation when appropriate
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

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": closing_message})

        print(
            f"Total processing latency: "
            f"{time.perf_counter() - total_start:.2f}s"
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
    # 5. Process emotion result
    # -------------------------
    top_emotion = max(
        emotion_results,
        key=lambda x: x["score"],
    )

    emotion = top_emotion["label"]
    emotion_score = top_emotion["score"]

    # -------------------------
    # 6. Process stress result
    # -------------------------
    top_stress = max(
        stress_results,
        key=lambda x: x["score"],
    )

    stress_label = top_stress["label"]
    stress_score = top_stress["score"]

    # -------------------------
    # 7. Process topic result
    # -------------------------
    topic = topic_result["labels"][0]
    topic_score = topic_result["scores"][0]

    # -------------------------
    # 8. Generate AI response
    # -------------------------
    response_start = time.perf_counter()

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

    print(
        f"Response generation latency: "
        f"{time.perf_counter() - response_start:.2f}s"
    )

    # -------------------------
    # 9. Save to short-term memory
    # -------------------------
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})

    print(
        f"Total processing latency: "
        f"{time.perf_counter() - total_start:.2f}s"
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


