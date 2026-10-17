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
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
# This file should live at:
# evaluation/scripts/evaluate_hf.py
#
# PROJECT_ROOT becomes the repository root so imports such as
# "from nlp.emotion_detector import detect_emotion" work.
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the exact Hugging Face functions used by main.py.
# The evaluator does not recreate or reimplement the models.
from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic


# ---------------------------------------------------------
# Fixed label orders used by the project/evaluation datasets
# ---------------------------------------------------------
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

STRESS_LABELS = [
    "non_stress",
    "stress",
]

TOPIC_LABELS = [
    "academic",
    "work or career",
    "relationships",
    "family",
    "financial",
    "health or wellness",
    "general",
]


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------
def predict_emotion(text):
    """
    detect_emotion() returns a list such as:
    [
        {"label": "fear", "score": 0.82},
        {"label": "sadness", "score": 0.10},
        ...
    ]

    We select the label with the highest score.
    """
    results = detect_emotion(text)
    top = max(results, key=lambda item: item["score"])
    return str(top["label"]).lower().strip(), float(top["score"])


def predict_stress(text):
    """
    Uses the project's existing detect_stress() function exactly as the
    application does, then selects the highest-scoring model label.
    """
    results = detect_stress(text)
    top = max(results, key=lambda item: item["score"])
    return str(top["label"]).lower().strip(), float(top["score"])


def predict_topic(text):
    """
    detect_topic() returns a zero-shot-classification dictionary:
    {
        "sequence": ...,
        "labels": [...],
        "scores": [...]
    }

    Hugging Face sorts labels from highest to lowest score.
    """
    result = detect_topic(text)
    return str(result["labels"][0]).lower().strip(), float(result["scores"][0])


# ---------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def calculate_metrics(df, true_col, pred_col, labels):
    y_true = df[true_col]
    y_pred = df[pred_col]

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "n_examples": int(len(df)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mean_latency_seconds": float(df["latency_seconds"].mean()),
        "median_latency_seconds": float(df["latency_seconds"].median()),
        "p95_latency_seconds": float(df["latency_seconds"].quantile(0.95)),
        "classification_report": report,
    }


def difficulty_breakdown(df, true_col, pred_col):
    rows = []

    for difficulty, group in df.groupby("difficulty", sort=False):
        rows.append(
            {
                "difficulty": difficulty,
                "n_examples": len(group),
                "accuracy": accuracy_score(group[true_col], group[pred_col]),
            }
        )

    return pd.DataFrame(rows)


def subcategory_breakdown(df, true_col, pred_col):
    rows = []

    for subcategory, group in df.groupby("subcategory", sort=False):
        rows.append(
            {
                "subcategory": subcategory,
                "n_examples": len(group),
                "accuracy": accuracy_score(group[true_col], group[pred_col]),
            }
        )

    return pd.DataFrame(rows)


def evaluate_dataset(
    task_name,
    csv_path,
    true_col,
    pred_col,
    score_col,
    labels,
    predictor,
):
    print(f"\n{'=' * 70}")
    print(f"Evaluating {task_name.upper()}")
    print(f"{'=' * 70}")

    df = pd.read_csv(csv_path)

    required_columns = {"id", "text", true_col, "difficulty", "subcategory"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    predictions = []
    scores = []
    latencies = []

    total = len(df)

    for index, row in df.iterrows():
        text = row["text"]

        start = time.perf_counter()
        predicted_label, confidence = predictor(text)
        elapsed = time.perf_counter() - start

        predictions.append(predicted_label)
        scores.append(confidence)
        latencies.append(elapsed)

        # Lightweight progress display.
        current = index + 1
        if current == 1 or current % 10 == 0 or current == total:
            print(f"{task_name}: {current}/{total}")

    df[pred_col] = predictions
    df[score_col] = scores
    df["latency_seconds"] = latencies
    df["correct"] = df[true_col] == df[pred_col]

    # Save raw predictions. This is important because later analysis
    # can be repeated without rerunning the models.
    predictions_path = RESULTS_DIR / f"{task_name}_hf_predictions.csv"
    df.to_csv(predictions_path, index=False)

    # Overall metrics.
    metrics = calculate_metrics(df, true_col, pred_col, labels)

    metrics_path = RESULTS_DIR / f"{task_name}_hf_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Per-difficulty results.
    difficulty_df = difficulty_breakdown(df, true_col, pred_col)
    difficulty_path = RESULTS_DIR / f"{task_name}_hf_by_difficulty.csv"
    difficulty_df.to_csv(difficulty_path, index=False)

    # Per-case-type results.
    subcategory_df = subcategory_breakdown(df, true_col, pred_col)
    subcategory_path = RESULTS_DIR / f"{task_name}_hf_by_subcategory.csv"
    subcategory_df.to_csv(subcategory_path, index=False)

    # Confusion matrix.
    cm_path = RESULTS_DIR / f"{task_name}_hf_confusion_matrix.png"
    save_confusion_matrix(
        df[true_col],
        df[pred_col],
        labels,
        f"Hugging Face {task_name.title()} Confusion Matrix",
        cm_path,
    )

    print(f"\n{task_name.title()} results")
    print(f"Examples:       {metrics['n_examples']}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro Precision:{metrics['macro_precision']:.4f}")
    print(f"Macro Recall:   {metrics['macro_recall']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Mean latency:   {metrics['mean_latency_seconds']:.4f} s")
    print(f"Median latency: {metrics['median_latency_seconds']:.4f} s")
    print(f"P95 latency:    {metrics['p95_latency_seconds']:.4f} s")

    return {
        "task": task_name,
        "n_examples": metrics["n_examples"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "mean_latency_seconds": metrics["mean_latency_seconds"],
        "median_latency_seconds": metrics["median_latency_seconds"],
        "p95_latency_seconds": metrics["p95_latency_seconds"],
    }


# ---------------------------------------------------------
# Main evaluation run
# ---------------------------------------------------------
def main():
    summaries = []

    summaries.append(
        evaluate_dataset(
            task_name="emotion",
            csv_path=DATA_DIR / "emotion_eval.csv",
            true_col="emotion_true",
            pred_col="emotion_pred",
            score_col="emotion_confidence",
            labels=EMOTION_LABELS,
            predictor=predict_emotion,
        )
    )

    summaries.append(
        evaluate_dataset(
            task_name="stress",
            csv_path=DATA_DIR / "stress_eval.csv",
            true_col="stress_true",
            pred_col="stress_pred",
            score_col="stress_confidence",
            labels=STRESS_LABELS,
            predictor=predict_stress,
        )
    )

    summaries.append(
        evaluate_dataset(
            task_name="topic",
            csv_path=DATA_DIR / "topic_eval.csv",
            true_col="topic_true",
            pred_col="topic_pred",
            score_col="topic_confidence",
            labels=TOPIC_LABELS,
            predictor=predict_topic,
        )
    )

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "hf_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 70}")
    print("HUGGING FACE BASELINE COMPLETE")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"\nAll results saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
