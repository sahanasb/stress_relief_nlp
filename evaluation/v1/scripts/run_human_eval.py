from __future__ import annotations

import csv
import inspect
import sys
from pathlib import Path

# Expected location after you copy this file into the repo:
# evaluation/scripts/run_human_eval.py
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results" / "human_eval"

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from agent.safety import check_safety
from agent.conversation_state import detect_conversation_state
from agent.stress_agent import generate_response
from nlp.openai_analyzer import analyze_message
import agent.safety_agent as safety_agent_module


SCENARIO_FILE = DATA_DIR / "human_eval_scenarios.csv"
RATER_OUTPUT_FILE = RESULTS_DIR / "human_eval_responses.csv"
DETAIL_OUTPUT_FILE = RESULTS_DIR / "human_eval_run_details.csv"


def _call_supported(func, **candidate_kwargs):
    """Call a project function using only kwargs its current signature accepts."""
    sig = inspect.signature(func)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**candidate_kwargs)

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in params}

    missing_required = [
        name
        for name, p in params.items()
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing_required:
        raise RuntimeError(
            f"Cannot call {func.__module__}.{func.__name__}; unsupported required "
            f"parameters: {missing_required}. Current signature: {sig}"
        )

    return func(**kwargs)


def _resolve_safety_response_function():
    """Find the response function exposed by the project's safety_agent module."""
    preferred_names = [
        "generate_safety_response",
        "generate_response",
        "respond_to_safety",
        "handle_safety",
        "safety_response",
    ]

    for name in preferred_names:
        func = getattr(safety_agent_module, name, None)
        if callable(func):
            return func

    public_functions = [
        (name, obj)
        for name, obj in vars(safety_agent_module).items()
        if callable(obj)
        and inspect.isfunction(obj)
        and obj.__module__ == safety_agent_module.__name__
        and not name.startswith("_")
    ]

    if len(public_functions) == 1:
        return public_functions[0][1]

    names = [name for name, _ in public_functions]
    raise RuntimeError(
        "Could not identify the safety-agent response function. "
        f"Public functions found: {names}. Update "
        "_resolve_safety_response_function() with the function name used in main.py."
    )


def call_safety_agent(user_message, history, safety_level):
    func = _resolve_safety_response_function()
    return _call_supported(
        func,
        user_message=user_message,
        message=user_message,
        text=user_message,
        conversation_history=history,
        history=history,
        safety_level=safety_level,
        safety_status=safety_level,
        risk_level=safety_level,
        safety_state=safety_level,
    )


def call_normal_agent(user_message, history, conversation_state, analysis):
    """Adapter for both the old signal-by-signal and newer combined-analysis signatures."""
    emotion = str(analysis.get("emotion", "neutral")).strip().lower()
    stress = str(analysis.get("stress", "no_stress")).strip().lower()
    topic = str(analysis.get("topic", "general")).strip().lower()

    # Some project versions call the non-stress class non_stress.
    stress_for_legacy_signature = "non_stress" if stress == "no_stress" else stress

    return _call_supported(
        generate_response,
        user_message=user_message,
        message=user_message,
        text=user_message,
        analysis=analysis,
        emotion=emotion,
        emotion_score=1.0,
        stress_label=stress_for_legacy_signature,
        stress=stress,
        stress_score=1.0,
        topic=topic,
        topic_score=1.0,
        conversation_history=history,
        history=history,
        conversation_state=conversation_state,
        state=conversation_state,
    )


def run_turn(user_message, history):
    """Run one text turn through the same logical routing used by the application."""
    safety_level = str(
        check_safety(
            user_message=user_message,
            conversation_history=history,
        )
    ).strip().lower()

    if safety_level in {"concern", "immediate_risk"}:
        response = call_safety_agent(user_message, history, safety_level)
        return {
            "response": str(response),
            "safety": safety_level,
            "conversation_state": "continue",
            "emotion": "",
            "stress": "",
            "topic": "",
            "route": "safety_agent",
        }

    conversation_state = str(
        detect_conversation_state(
            user_message=user_message,
            conversation_history=history,
        )
    ).strip().lower()

    analysis = analyze_message(
        user_message=user_message,
        conversation_history=history,
    )

    response = call_normal_agent(
        user_message=user_message,
        history=history,
        conversation_state=conversation_state,
        analysis=analysis,
    )

    return {
        "response": str(response),
        "safety": safety_level,
        "conversation_state": conversation_state,
        "emotion": str(analysis.get("emotion", "")),
        "stress": str(analysis.get("stress", "")),
        "topic": str(analysis.get("topic", "")),
        "route": "stress_agent",
    }


def load_scenarios():
    required = {
        "scenario_id",
        "category",
        "difficulty",
        "test_focus",
        "user_turn_1",
        "user_turn_2",
        "user_turn_3",
    }

    with SCENARIO_FILE.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing scenario columns: {sorted(missing)}")
        return list(reader)


def main():
    scenarios = load_scenarios()
    rater_rows = []
    detail_rows = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["scenario_id"]
        print(f"\n[{scenario_index}/{len(scenarios)}] {scenario_id} - {scenario['test_focus']}")

        # Every scenario is a fresh independent conversation.
        history = []
        responses = ["", "", ""]
        terminated_after_turn = ""
        scenario_error = ""

        for turn_number in (1, 2, 3):
            user_message = scenario[f"user_turn_{turn_number}"].strip()
            if not user_message:
                continue

            print(f"  Turn {turn_number}: {user_message[:80]}")

            try:
                result = run_turn(user_message, history)
                assistant_response = result["response"].strip()
                responses[turn_number - 1] = assistant_response

                detail_rows.append({
                    "scenario_id": scenario_id,
                    "turn": turn_number,
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "route": result["route"],
                    "safety": result["safety"],
                    "conversation_state": result["conversation_state"],
                    "emotion": result["emotion"],
                    "stress": result["stress"],
                    "topic": result["topic"],
                    "error": "",
                })

                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": assistant_response})

                # Match the application's lifecycle: a true close ends this scenario.
                if result["conversation_state"] == "close":
                    terminated_after_turn = str(turn_number)
                    print(f"    Conversation closed after turn {turn_number}.")
                    break

            except Exception as exc:
                scenario_error = f"{type(exc).__name__}: {exc}"
                print(f"    ERROR: {scenario_error}")
                detail_rows.append({
                    "scenario_id": scenario_id,
                    "turn": turn_number,
                    "user_message": user_message,
                    "assistant_response": "",
                    "route": "",
                    "safety": "",
                    "conversation_state": "",
                    "emotion": "",
                    "stress": "",
                    "topic": "",
                    "error": scenario_error,
                })
                break

        # This is the clean file you can give to human raters.
        # It intentionally excludes internal model labels to reduce rating bias.
        rater_rows.append({
            "scenario_id": scenario_id,
            "category": scenario["category"],
            "difficulty": scenario["difficulty"],
            "test_focus": scenario["test_focus"],
            "user_turn_1": scenario["user_turn_1"],
            "assistant_turn_1": responses[0],
            "user_turn_2": scenario["user_turn_2"],
            "assistant_turn_2": responses[1],
            "user_turn_3": scenario["user_turn_3"],
            "assistant_turn_3": responses[2],
            "terminated_after_turn": terminated_after_turn,
            "run_error": scenario_error,
        })

    rater_fields = [
        "scenario_id", "category", "difficulty", "test_focus",
        "user_turn_1", "assistant_turn_1",
        "user_turn_2", "assistant_turn_2",
        "user_turn_3", "assistant_turn_3",
        "terminated_after_turn", "run_error",
    ]
    with RATER_OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rater_fields)
        writer.writeheader()
        writer.writerows(rater_rows)

    detail_fields = [
        "scenario_id", "turn", "user_message", "assistant_response", "route",
        "safety", "conversation_state", "emotion", "stress", "topic", "error",
    ]
    with DETAIL_OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\nHuman-evaluation run complete.")
    print(f"Rater file:   {RATER_OUTPUT_FILE}")
    print(f"Details file: {DETAIL_OUTPUT_FILE}")
    print("\nSend human_eval_responses.csv to both raters. Keep human_eval_run_details.csv for auditing/debugging.")


if __name__ == "__main__":
    main()
