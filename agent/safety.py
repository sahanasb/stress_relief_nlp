import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def check_safety(user_message, conversation_history=None):
    """
    Classify the user's current safety state.

    Possible states:
    - normal
    - concern
    - immediate_risk
    """

    if conversation_history is None:
        conversation_history = []

    history_text = ""

    for message in conversation_history:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a safety classifier for a stress-support application.

Review the previous conversation and the user's current message.

Previous conversation:
{history_text}

Current user message:
{user_message}

Classify the situation into exactly ONE of these categories:

normal
- Everyday stress, sadness, anger, anxiety, frustration, arguments,
  school stress, work stress, relationship problems, or similar concerns.
- There is no indication that the user may harm themselves or someone else.

concern
- The user expresses hopelessness, not wanting to live, wishing they
  could disappear, or other statements that could indicate a safety concern.
- The situation deserves a supportive safety-oriented check-in, but the
  message does not clearly indicate immediate intent, plan, or action.

immediate_risk
- The user clearly indicates an immediate intention, plan, attempt,
  or inability to stay safe.
- The user indicates an immediate intention or plan to seriously harm
  another person.

Important:
Do not classify ordinary sadness, anger, arguments, anxiety, or stress
as immediate_risk merely because the emotion is intense.

Return ONLY one exact value:

normal
concern
immediate_risk
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    safety_state = response.output_text.strip().lower()

    valid_states = {
        "normal",
        "concern",
        "immediate_risk"
    }

    # Fail conservatively if an unexpected value is returned.
    if safety_state not in valid_states:
        return "concern"

    return safety_state

