from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def detect_conversation_state(user_message, conversation_history):
    """
    Determine the current state of the conversation.

    Possible states:
    - continue
    - offer_to_close
    - close
    """

    history_text = ""

    for message in conversation_history:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are analyzing the state of a stress-support conversation.

Previous conversation:
{history_text}

Current user message:
{user_message}

Classify the conversation into exactly ONE of these states:

continue
- The user is still explaining their problem.
- The user asks a question.
- The user wants more help.
- The user introduces another concern.
- The user says they want to continue talking.

offer_to_close
- The user's original concern has been substantially addressed.
- The user indicates they feel calmer, better, clearer, or more prepared.
- The user appears to have a reasonable next step.
- However, the user has NOT explicitly said they want to end.

close
- The user clearly indicates they are finished.
- Examples include:
  "I'm good now."
  "That's all."
  "Thanks, that's enough."
  "I have to go."
  "Bye."
  "Thank you, I feel better now."

Important:
Do not choose close merely because the conversation is long.
If the user is still asking questions, choose continue.

Return ONLY one of these exact values:

continue
offer_to_close
close
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    state = response.output_text.strip().lower()

    # Fallback in case the model returns unexpected text
    valid_states = {
        "continue",
        "offer_to_close",
        "close"
    }

    if state not in valid_states:
        return "continue"

    return state
