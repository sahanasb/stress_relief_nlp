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
# Expected location:
# evaluation/scripts/evaluate_safety.py
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results" / "safety"

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the ACTUAL safety function used by the project.
from agent.safety import check_safety


SAFETY_LABELS = [
    "normal",
    "concern",
    "immediate_risk",
]


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm)

    ax.set_title("Safety Classification Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

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
                    group["safety_true"],
                    group["safety_pred"],
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    csv_path = DATA_DIR / "safety_eval.csv"

    print("=" * 70)
    print("Evaluating SAFETY CLASSIFIER")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    required_columns = {
        "id",
        "text",
        "safety_true",
        "difficulty",
        "subcategory",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    # Validate ground-truth labels before running expensive API calls.
    unexpected_true_labels = sorted(
        set(df["safety_true"]) - set(SAFETY_LABELS)
    )

    if unexpected_true_labels:
        raise ValueError(
            "Unexpected ground-truth safety labels found: "
            f"{unexpected_true_labels}"
        )

    predictions = []
    latencies = []
    errors = []

    total = len(df)

    for index, row in df.iterrows():
        text = row["text"]

        try:
            start = time.perf_counter()

            # IMPORTANT:
            # This calls the same check_safety() function used by main.py.
            #
            # conversation_history=[] intentionally isolates each CSV row.
            # That means this benchmark measures single-message safety
            # classification without context leaking between test cases.
            predicted_label = check_safety(
                user_message=text,
                conversation_history=[],
            )

            elapsed = time.perf_counter() - start

            predictions.append(str(predicted_label).strip().lower())
            latencies.append(elapsed)
            errors.append("")

        except Exception as exc:
            elapsed = time.perf_counter() - start

            predictions.append("ERROR")
            latencies.append(elapsed)
            errors.append(str(exc))

        current = index + 1

        if current == 1 or current % 10 == 0 or current == total:
            print(f"Safety: {current}/{total}")

    df["safety_pred"] = predictions
    df["latency_seconds"] = latencies
    df["error"] = errors
    df["correct"] = df["safety_true"] == df["safety_pred"]

    # ---------------------------------------------------------
    # Separate successful API/model calls from failures.
    # ---------------------------------------------------------
    valid_df = df[df["safety_pred"].isin(SAFETY_LABELS)].copy()
    failed_df = df[~df["safety_pred"].isin(SAFETY_LABELS)].copy()

    if valid_df.empty:
        raise RuntimeError(
            "No valid safety predictions were produced. "
            "Check your API key and project configuration."
        )

    y_true = valid_df["safety_true"]
    y_pred = valid_df["safety_pred"]

    # ---------------------------------------------------------
    # Standard classification metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=SAFETY_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=SAFETY_LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=SAFETY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # Safety-specific metrics
    # ---------------------------------------------------------
    immediate_risk_recall = recall_score(
        y_true,
        y_pred,
        labels=SAFETY_LABELS,
        average=None,
        zero_division=0,
    )[SAFETY_LABELS.index("immediate_risk")]

    concern_recall = recall_score(
        y_true,
        y_pred,
        labels=SAFETY_LABELS,
        average=None,
        zero_division=0,
    )[SAFETY_LABELS.index("concern")]

    # Most critical error:
    # true immediate risk incorrectly classified as normal.
    immediate_as_normal = valid_df[
        (valid_df["safety_true"] == "immediate_risk")
        & (valid_df["safety_pred"] == "normal")
    ]

    n_immediate = int(
        (valid_df["safety_true"] == "immediate_risk").sum()
    )

    immediate_to_normal_rate = (
        len(immediate_as_normal) / n_immediate
        if n_immediate > 0
        else 0.0
    )

    # Broader safety false negative:
    # any safety-sensitive example (concern OR immediate_risk)
    # incorrectly routed to normal.
    safety_sensitive = valid_df[
        valid_df["safety_true"].isin(
            ["concern", "immediate_risk"]
        )
    ]

    safety_sensitive_as_normal = safety_sensitive[
        safety_sensitive["safety_pred"] == "normal"
    ]

    safety_false_negative_rate = (
        len(safety_sensitive_as_normal) / len(safety_sensitive)
        if len(safety_sensitive) > 0
        else 0.0
    )

    # False alarm rate:
    # a truly normal example routed into concern/immediate_risk.
    normal_cases = valid_df[
        valid_df["safety_true"] == "normal"
    ]

    normal_false_alarms = normal_cases[
        normal_cases["safety_pred"].isin(
            ["concern", "immediate_risk"]
        )
    ]

    normal_false_alarm_rate = (
        len(normal_false_alarms) / len(normal_cases)
        if len(normal_cases) > 0
        else 0.0
    )

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
        "concern_recall": float(concern_recall),
        "immediate_risk_recall": float(immediate_risk_recall),
        "immediate_risk_to_normal_count": int(
            len(immediate_as_normal)
        ),
        "immediate_risk_to_normal_rate": float(
            immediate_to_normal_rate
        ),
        "safety_false_negative_count": int(
            len(safety_sensitive_as_normal)
        ),
        "safety_false_negative_rate": float(
            safety_false_negative_rate
        ),
        "normal_false_alarm_count": int(
            len(normal_false_alarms)
        ),
        "normal_false_alarm_rate": float(
            normal_false_alarm_rate
        ),
        "mean_latency_seconds": float(
            valid_df["latency_seconds"].mean()
        ),
        "median_latency_seconds": float(
            valid_df["latency_seconds"].median()
        ),
        "p95_latency_seconds": float(
            valid_df["latency_seconds"].quantile(0.95)
        ),
        "classification_report": report,
    }

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    predictions_path = RESULTS_DIR / "safety_predictions.csv"
    df.to_csv(predictions_path, index=False)

    metrics_path = RESULTS_DIR / "safety_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    difficulty_df = accuracy_breakdown(
        valid_df,
        "difficulty",
    )
    difficulty_df.to_csv(
        RESULTS_DIR / "safety_by_difficulty.csv",
        index=False,
    )

    subcategory_df = accuracy_breakdown(
        valid_df,
        "subcategory",
    )
    subcategory_df.to_csv(
        RESULTS_DIR / "safety_by_subcategory.csv",
        index=False,
    )

    class_df = accuracy_breakdown(
        valid_df,
        "safety_true",
    )
    class_df.to_csv(
        RESULTS_DIR / "safety_by_class.csv",
        index=False,
    )

    immediate_as_normal.to_csv(
        RESULTS_DIR / "critical_immediate_risk_false_negatives.csv",
        index=False,
    )

    safety_sensitive_as_normal.to_csv(
        RESULTS_DIR / "all_safety_false_negatives.csv",
        index=False,
    )

    normal_false_alarms.to_csv(
        RESULTS_DIR / "normal_false_alarms.csv",
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
        SAFETY_LABELS,
        RESULTS_DIR / "safety_confusion_matrix.png",
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------
    print("\nSafety evaluation results")
    print("-" * 70)
    print(f"Examples:                       {len(df)}")
    print(f"Valid predictions:              {len(valid_df)}")
    print(f"Failed predictions:             {len(failed_df)}")
    print(f"Accuracy:                       {accuracy:.4f}")
    print(f"Macro Precision:                {macro_precision:.4f}")
    print(f"Macro Recall:                   {macro_recall:.4f}")
    print(f"Macro F1:                       {macro_f1:.4f}")
    print(f"Concern Recall:                 {concern_recall:.4f}")
    print(f"Immediate-risk Recall:          {immediate_risk_recall:.4f}")
    print(
        "Immediate risk -> normal:       "
        f"{len(immediate_as_normal)}/{n_immediate} "
        f"({immediate_to_normal_rate:.2%})"
    )
    print(
        "Safety false-negative rate:     "
        f"{safety_false_negative_rate:.2%}"
    )
    print(
        "Normal false-alarm rate:        "
        f"{normal_false_alarm_rate:.2%}"
    )
    print(
        "Mean latency:                   "
        f"{metrics['mean_latency_seconds']:.4f} s"
    )
    print(
        "Median latency:                 "
        f"{metrics['median_latency_seconds']:.4f} s"
    )
    print(
        "P95 latency:                    "
        f"{metrics['p95_latency_seconds']:.4f} s"
    )

    print(f"\nResults saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
