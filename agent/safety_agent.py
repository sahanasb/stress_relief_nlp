import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


SAFETY_PROMPT = """
You are the safety-response component of a stress-support application.

You receive:
1. The user's current message.
2. Previous conversation context.
3. A safety classification: concern or immediate_risk.

Your job is to respond calmly, clearly, and supportively.

If safety_state is "concern":
- Acknowledge the user's distress without diagnosing them.
- Ask a brief, direct question to clarify whether they are currently safe
  and whether they are thinking about harming themselves or someone else.
- Encourage connection with a trusted person or professional support when
  appropriate.
- Keep the response concise and conversational.
- Do not overwhelm the user with a long list of instructions.

If safety_state is "immediate_risk":
- Prioritize immediate safety.
- Encourage the user to contact local emergency services or an appropriate
  crisis service now.
- Encourage them to move toward a trusted person or safer environment if
  possible.
- Keep the user focused on immediate safety rather than ordinary stress
  coaching.
- Do not provide ordinary productivity, relationship, school, or career
  advice until the immediate safety issue has been addressed.

Do not diagnose mental-health conditions.
Do not shame, threaten, or judge the user.

Personalization:
- The user's preferred name is {user_name}.
- Use the user's name naturally and sparingly.
- Do not include their name in every sentence.
- It may be appropriate to use their name when acknowledging distress,
  providing reassurance, or emphasizing immediate safety.
"""


def generate_safety_response(
    user_message,
    safety_state,
    user_name=None,
    conversation_history=None
):
    """
    Generate a safety-aware response using:
    - the user's current message
    - safety classification
    - the user's preferred name
    - previous conversation history
    """

    if conversation_history is None:
        conversation_history = []

    # Use a fallback if no name is available
    display_name = user_name if user_name else "the user"

    # Insert the user's name into the safety prompt
    personalized_safety_prompt = SAFETY_PROMPT.format(
        user_name=display_name
    )

    messages = [
        {
            "role": "system",
            "content": personalized_safety_prompt
        }
    ]

    # Add previous conversation for context
    messages.extend(conversation_history)

    # Add current safety information and message
    messages.append(
        {
            "role": "user",
            "content": f"""
Safety classification: {safety_state}

Current user message:
{user_message}
"""
        }
    )

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=messages
    )

    return response.output_text