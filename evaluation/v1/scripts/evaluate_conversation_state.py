from pathlib import Path
import sys
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results" / "conversation_state"

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the ACTUAL project function.
from agent.conversation_state import detect_conversation_state


STATE_LABELS = [
    "continue",
    "offer_to_close",
    "close",
]


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm)

    ax.set_title("Conversation State Confusion Matrix")
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("True state")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def accuracy_breakdown(df, group_col):
    rows = []

    for group_name, group in df.groupby(group_col, sort=False):
        rows.append(
            {
                group_col: group_name,
                "n_examples": len(group),
                "accuracy": accuracy_score(
                    group["state_true"],
                    group["state_pred"],
                ),
            }
        )

    return pd.DataFrame(rows)


def parse_history(raw_history, row_id):
    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in history for {row_id}: {exc}"
        ) from exc

    if not isinstance(history, list):
        raise ValueError(f"History for {row_id} must be a JSON list.")

    for i, message in enumerate(history):
        if not isinstance(message, dict):
            raise ValueError(
                f"History message {i} for {row_id} must be an object."
            )
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"History message {i} for {row_id} must contain role and content."
            )

    return history


def main():
    csv_path = DATA_DIR / "conversation_state_eval.csv"

    print("=" * 70)
    print("Evaluating CONVERSATION STATE CLASSIFIER")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    required_columns = {
        "id",
        "history",
        "current_message",
        "state_true",
        "difficulty",
        "subcategory",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    unexpected_true_labels = sorted(
        set(df["state_true"]) - set(STATE_LABELS)
    )
    if unexpected_true_labels:
        raise ValueError(
            "Unexpected ground-truth conversation-state labels found: "
            f"{unexpected_true_labels}"
        )

    predictions = []
    latencies = []
    errors = []

    total = len(df)

    for index, row in df.iterrows():
        start = time.perf_counter()

        try:
            history = parse_history(row["history"], row["id"])

            # This is the same function and argument structure used by the project.
            predicted_state = detect_conversation_state(
                user_message=row["current_message"],
                conversation_history=history,
            )

            elapsed = time.perf_counter() - start

            predictions.append(str(predicted_state).strip().lower())
            latencies.append(elapsed)
            errors.append("")

        except Exception as exc:
            elapsed = time.perf_counter() - start
            predictions.append("ERROR")
            latencies.append(elapsed)
            errors.append(str(exc))

        current = index + 1
        if current == 1 or current % 10 == 0 or current == total:
            print(f"Conversation state: {current}/{total}")

    df["state_pred"] = predictions
    df["latency_seconds"] = latencies
    df["error"] = errors
    df["correct"] = df["state_true"] == df["state_pred"]

    valid_df = df[df["state_pred"].isin(STATE_LABELS)].copy()
    failed_df = df[~df["state_pred"].isin(STATE_LABELS)].copy()

    if valid_df.empty:
        raise RuntimeError(
            "No valid conversation-state predictions were produced. "
            "Check your API key and project configuration."
        )

    y_true = valid_df["state_true"]
    y_pred = valid_df["state_pred"]

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=STATE_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=STATE_LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=STATE_LABELS,
        output_dict=True,
        zero_division=0,
    )

    recalls = recall_score(
        y_true,
        y_pred,
        labels=STATE_LABELS,
        average=None,
        zero_division=0,
    )
    recall_by_state = dict(zip(STATE_LABELS, recalls))

    # ---------------------------------------------------------
    # Conversation-specific error types
    # ---------------------------------------------------------
    # Premature termination: user should continue, but classifier says close.
    continue_as_close = valid_df[
        (valid_df["state_true"] == "continue")
        & (valid_df["state_pred"] == "close")
    ]

    # Premature hard close: system should only offer closure, but classifier closes.
    offer_as_close = valid_df[
        (valid_df["state_true"] == "offer_to_close")
        & (valid_df["state_pred"] == "close")
    ]

    # Missed termination: user clearly wants to close, but classifier continues.
    close_as_continue = valid_df[
        (valid_df["state_true"] == "close")
        & (valid_df["state_pred"] == "continue")
    ]

    # Soft miss: user wants close, but system only offers to close.
    close_as_offer = valid_df[
        (valid_df["state_true"] == "close")
        & (valid_df["state_pred"] == "offer_to_close")
    ]

    n_continue = int((valid_df["state_true"] == "continue").sum())
    n_offer = int((valid_df["state_true"] == "offer_to_close").sum())
    n_close = int((valid_df["state_true"] == "close").sum())

    metrics = {
        "n_total_examples": int(len(df)),
        "n_valid_predictions": int(len(valid_df)),
        "n_failed_predictions": int(len(failed_df)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "continue_recall": float(recall_by_state["continue"]),
        "offer_to_close_recall": float(recall_by_state["offer_to_close"]),
        "close_recall": float(recall_by_state["close"]),
        "continue_to_close_count": int(len(continue_as_close)),
        "continue_to_close_rate": float(len(continue_as_close) / n_continue if n_continue else 0.0),
        "offer_to_close_to_close_count": int(len(offer_as_close)),
        "offer_to_close_to_close_rate": float(len(offer_as_close) / n_offer if n_offer else 0.0),
        "close_to_continue_count": int(len(close_as_continue)),
        "close_to_continue_rate": float(len(close_as_continue) / n_close if n_close else 0.0),
        "close_to_offer_count": int(len(close_as_offer)),
        "close_to_offer_rate": float(len(close_as_offer) / n_close if n_close else 0.0),
        "mean_latency_seconds": float(valid_df["latency_seconds"].mean()),
        "median_latency_seconds": float(valid_df["latency_seconds"].median()),
        "p95_latency_seconds": float(valid_df["latency_seconds"].quantile(0.95)),
        "classification_report": report,
    }

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    df.to_csv(RESULTS_DIR / "conversation_state_predictions.csv", index=False)

    with open(RESULTS_DIR / "conversation_state_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    accuracy_breakdown(valid_df, "difficulty").to_csv(
        RESULTS_DIR / "conversation_state_by_difficulty.csv",
        index=False,
    )

    accuracy_breakdown(valid_df, "subcategory").to_csv(
        RESULTS_DIR / "conversation_state_by_subcategory.csv",
        index=False,
    )

    accuracy_breakdown(valid_df, "state_true").to_csv(
        RESULTS_DIR / "conversation_state_by_class.csv",
        index=False,
    )

    valid_df[~valid_df["correct"]].to_csv(
        RESULTS_DIR / "conversation_state_misclassified.csv",
        index=False,
    )

    continue_as_close.to_csv(
        RESULTS_DIR / "critical_continue_to_close.csv",
        index=False,
    )

    close_as_continue.to_csv(
        RESULTS_DIR / "critical_close_to_continue.csv",
        index=False,
    )

    if not failed_df.empty:
        failed_df.to_csv(
            RESULTS_DIR / "failed_predictions.csv",
            index=False,
        )

    save_confusion_matrix(
        y_true,
        y_pred,
        STATE_LABELS,
        RESULTS_DIR / "conversation_state_confusion_matrix.png",
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------
    print("\nConversation-state evaluation results")
    print("-" * 70)
    print(f"Examples:                    {len(df)}")
    print(f"Valid predictions:           {len(valid_df)}")
    print(f"Failed predictions:          {len(failed_df)}")
    print(f"Accuracy:                    {accuracy:.4f}")
    print(f"Macro Precision:             {macro_precision:.4f}")
    print(f"Macro Recall:                {macro_recall:.4f}")
    print(f"Macro F1:                    {macro_f1:.4f}")
    print(f"Continue Recall:             {recall_by_state['continue']:.4f}")
    print(f"Offer-to-close Recall:       {recall_by_state['offer_to_close']:.4f}")
    print(f"Close Recall:                {recall_by_state['close']:.4f}")
    print(
        "Continue -> Close:           "
        f"{len(continue_as_close)}/{n_continue} "
        f"({len(continue_as_close)/n_continue:.2%})" if n_continue else "N/A"
    )
    print(
        "Offer-to-close -> Close:     "
        f"{len(offer_as_close)}/{n_offer} "
        f"({len(offer_as_close)/n_offer:.2%})" if n_offer else "N/A"
    )
    print(
        "Close -> Continue:           "
        f"{len(close_as_continue)}/{n_close} "
        f"({len(close_as_continue)/n_close:.2%})" if n_close else "N/A"
    )
    print(
        "Close -> Offer-to-close:     "
        f"{len(close_as_offer)}/{n_close} "
        f"({len(close_as_offer)/n_close:.2%})" if n_close else "N/A"
    )
    print(f"Mean latency:                {metrics['mean_latency_seconds']:.4f} s")
    print(f"Median latency:              {metrics['median_latency_seconds']:.4f} s")
    print(f"P95 latency:                 {metrics['p95_latency_seconds']:.4f} s")

    print(f"\nResults saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
