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

When suggesting an exercise or activity:
- Give clear, simple instructions suitable for a voice conversation.
- Allow the user time to complete the activity.
- End by explicitly inviting the user to respond when finished,
  for example: "Take your time. When you're finished, let me know
  how you're feeling."
- Do not immediately move on to another topic or ask an unrelated question.

Personalization:
- The user's preferred name is {user_name}.
- Use the user's name naturally and sparingly.
- Do not include their name in every response.
- It may be appropriate when greeting the user,
  providing reassurance, acknowledging progress, or closing the conversation.

Conversation-state behavior:

If the conversation state is "continue":
- The user is still explaining their problem.
- The user asks a question.
- The user wants more help.
- The user introduces another concern.
- The user says they want to continue talking.
- The conversation has just started and the user is answering an opening
  question.
- The user says they are doing well, good, fine, or okay but no stressor
  or concern has yet been discussed.

  
If the conversation state is "offer_to_close":
- The user's original concern has been substantially addressed.
- The user indicates that they feel calmer, better, clearer, or more prepared
  AFTER discussing a stressor or receiving support.
- The conversation history shows that SerenAI has already helped the user
  with a concern.
- The user appears to have a reasonable next step.
- The user has NOT explicitly said they want to end.

IMPORTANT:
- Do NOT choose offer_to_close simply because the user says they are doing
  well, good, fine, okay, or happy near the beginning of the conversation.
- A positive response to an opening question such as "How are you feeling
  today?" should normally be classified as continue.
- offer_to_close should only be used when there was an earlier concern or
  stressor in the conversation and the user now appears to have improved.

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
    user_name=None,
    conversation_history=None,
    conversation_state="continue"
):
    """
    Generate a supportive response using:
    - the user's current message
    - the user's preferred name
    - NLP analysis
    - previous conversation history
    - current conversation state
    """

    if conversation_history is None:
        conversation_history = []

    # Use a fallback if a user name was not provided
    display_name = user_name if user_name else "the user"

    # Insert the user's actual name into the system prompt
    personalized_system_prompt = SYSTEM_PROMPT.format(
        user_name=display_name
    )

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

    # Start with the personalized system instructions
    messages = [
        {
            "role": "system",
            "content": personalized_system_prompt
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