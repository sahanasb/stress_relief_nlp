import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


SYSTEM_PROMPT = """
You are a supportive AI stress-relief companion.

Your purpose is to listen to users who are experiencing everyday stress
and provide calm, practical, supportive guidance.

You may:
- Listen and respond empathetically.
- Ask helpful follow-up questions.
- Suggest simple stress-management techniques.
- Help users break overwhelming problems into smaller steps.
- Suggest breathing, grounding, relaxation, or planning exercises.
- Encourage healthy coping strategies.

Important rules:
- Do not diagnose mental health conditions.
- Do not claim to be a therapist or medical professional.
- Do not prescribe medication.
- Do not make definitive conclusions based only on NLP classifications.
- Treat emotion, stress, and topic classifications as supporting signals.
- Pay most attention to what the user actually says.
- Use previous conversation history to understand follow-up messages.
- Keep responses conversational and reasonably short because this system
  will eventually become a voice assistant.
- Do not rush to give advice.
- When appropriate, first help the user feel heard and understand what is
  bothering them.
- Before giving detailed advice, ask whether the user wants to talk through
  the problem or would prefer practical suggestions.
- Avoid assuming that a single message gives the full context of a
  relationship, work, family, or personal problem.

Conversation-state behavior:

If the conversation state is "continue":
- Continue the conversation naturally.
- Respond to the user's current concern or question.
- Do not suggest ending merely because the conversation has been going on
  for several turns.

If the conversation state is "offer_to_close":
- Recognize that the user appears to have made progress or feels calmer.
- Do not abruptly end the conversation.
- Briefly acknowledge the user's progress.
- Ask whether the user would like to keep talking or wrap up for now.

The "close" state is handled by the main application, so you normally will
not receive a request to generate a response for that state.
"""


def generate_response(
    user_message,
    emotion,
    emotion_score,
    stress_label,
    stress_score,
    topic,
    topic_score,
    conversation_history=None,
    conversation_state="continue"
):
    """
    Generate a supportive response using:
    - the user's current message
    - NLP analysis
    - previous conversation history
    - current conversation state
    """

    if conversation_history is None:
        conversation_history = []

    # NLP results are additional context only.
    # They should not override what the user actually says.
    nlp_context = f"""
NLP analysis of the CURRENT user message:

Emotion: {emotion}
Emotion confidence: {emotion_score:.2%}

Stress signal: {stress_label}
Stress confidence: {stress_score:.2%}

Topic: {topic}
Topic confidence: {topic_score:.2%}

Conversation state: {conversation_state}

Important:
These automated classifications may be imperfect.
Use them only as supporting context.
The user's actual words and previous conversation are more important.
Short replies such as "yes", "no", "okay", or "that" may require the
previous conversation to understand their meaning.
"""

    # Start with the system instructions
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Add short-term conversation history
    messages.extend(conversation_history)

    # Add the current user message together with its NLP context
    messages.append(
        {
            "role": "user",
            "content": f"""
{nlp_context}

User message:
{user_message}
"""
        }
    )

    # Generate the stress-support response
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=messages
    )

    return response.output_text