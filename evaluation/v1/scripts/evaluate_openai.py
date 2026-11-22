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

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results" / "openai"

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the exact OpenAI analyzer used by the current application.
from nlp.openai_analyzer import analyze_message

EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]

# Canonical evaluation labels match the HF baseline dataset.
STRESS_LABELS = ["non_stress", "stress"]

TOPIC_LABELS = [
    "academic",
    "work or career",
    "relationships",
    "family",
    "financial",
    "health or wellness",
    "general",
]


def normalize_openai_stress(label):
    """
    The OpenAI analyzer currently emits 'no_stress', while the HF baseline
    and evaluation CSV use 'non_stress'. These mean the same class.
    We preserve the raw OpenAI output separately and normalize only for
    metric comparison.
    """
    label = str(label).lower().strip()
    if label == "no_stress":
        return "non_stress"
    return label


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
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
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


def grouped_accuracy(df, group_col, true_col, pred_col):
    rows = []
    for value, group in df.groupby(group_col, sort=False):
        rows.append({
            group_col: value,
            "n_examples": int(len(group)),
            "accuracy": float(accuracy_score(group[true_col], group[pred_col])),
        })
    return pd.DataFrame(rows)


def evaluate_dataset(task_name, csv_path, true_col, labels):
    print(f"\n{'=' * 70}")
    print(f"Evaluating OpenAI {task_name.upper()}")
    print(f"{'=' * 70}")

    df = pd.read_csv(csv_path)
    required = {"id", "text", true_col, "difficulty", "subcategory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    raw_predictions = []
    metric_predictions = []
    all_emotions = []
    all_stresses = []
    all_topics = []
    latencies = []
    errors = []

    total = len(df)

    for index, row in df.iterrows():
        text = row["text"]

        try:
            start = time.perf_counter()
            analysis = analyze_message(user_message=text, conversation_history=[])
            elapsed = time.perf_counter() - start

            emotion = str(analysis["emotion"]).lower().strip()
            stress = str(analysis["stress"]).lower().strip()
            topic = str(analysis["topic"]).lower().strip()

            if task_name == "emotion":
                raw_pred = emotion
                metric_pred = emotion
            elif task_name == "stress":
                raw_pred = stress
                metric_pred = normalize_openai_stress(stress)
            elif task_name == "topic":
                raw_pred = topic
                metric_pred = topic
            else:
                raise ValueError(f"Unknown task: {task_name}")

            error = ""

        except Exception as exc:
            elapsed = float("nan")
            emotion = ""
            stress = ""
            topic = ""
            raw_pred = "__error__"
            metric_pred = "__error__"
            error = f"{type(exc).__name__}: {exc}"

        raw_predictions.append(raw_pred)
        metric_predictions.append(metric_pred)
        all_emotions.append(emotion)
        all_stresses.append(stress)
        all_topics.append(topic)
        latencies.append(elapsed)
        errors.append(error)

        current = index + 1
        if current == 1 or current % 10 == 0 or current == total:
            print(f"{task_name}: {current}/{total}")

    pred_col = f"{task_name}_pred"
    raw_col = f"{task_name}_pred_raw"

    df[raw_col] = raw_predictions
    df[pred_col] = metric_predictions
    df["openai_emotion"] = all_emotions
    df["openai_stress"] = all_stresses
    df["openai_topic"] = all_topics
    df["latency_seconds"] = latencies
    df["error"] = errors
    df["correct"] = df[true_col] == df[pred_col]

    predictions_path = RESULTS_DIR / f"{task_name}_openai_predictions.csv"
    df.to_csv(predictions_path, index=False)

    metrics = calculate_metrics(df, true_col, pred_col, labels)

    # Additional diagnostics for OpenAI-only labels such as mixed/uncertain.
    out_of_schema_mask = ~df[pred_col].isin(labels)
    metrics["out_of_schema_predictions"] = int(out_of_schema_mask.sum())
    metrics["out_of_schema_rate"] = float(out_of_schema_mask.mean())
    metrics["api_errors"] = int((df[raw_col] == "__error__").sum())

    with open(RESULTS_DIR / f"{task_name}_openai_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    grouped_accuracy(df, "difficulty", true_col, pred_col).to_csv(
        RESULTS_DIR / f"{task_name}_openai_by_difficulty.csv", index=False
    )
    grouped_accuracy(df, "subcategory", true_col, pred_col).to_csv(
        RESULTS_DIR / f"{task_name}_openai_by_subcategory.csv", index=False
    )

    save_confusion_matrix(
        df[true_col],
        df[pred_col],
        labels,
        f"OpenAI {task_name.title()} Confusion Matrix",
        RESULTS_DIR / f"{task_name}_openai_confusion_matrix.png",
    )

    print(f"\n{task_name.title()} results")
    print(f"Examples:             {metrics['n_examples']}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Mean latency:         {metrics['mean_latency_seconds']:.4f} s")
    print(f"Median latency:       {metrics['median_latency_seconds']:.4f} s")
    print(f"P95 latency:          {metrics['p95_latency_seconds']:.4f} s")
    print(f"Out-of-schema rate:   {metrics['out_of_schema_rate']:.4f}")
    print(f"API errors:           {metrics['api_errors']}")

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
        "out_of_schema_rate": metrics["out_of_schema_rate"],
        "api_errors": metrics["api_errors"],
    }


def main():
    summaries = []

    summaries.append(evaluate_dataset(
        task_name="emotion",
        csv_path=DATA_DIR / "emotion_eval.csv",
        true_col="emotion_true",
        labels=EMOTION_LABELS,
    ))

    summaries.append(evaluate_dataset(
        task_name="stress",
        csv_path=DATA_DIR / "stress_eval.csv",
        true_col="stress_true",
        labels=STRESS_LABELS,
    ))

    summaries.append(evaluate_dataset(
        task_name="topic",
        csv_path=DATA_DIR / "topic_eval.csv",
        true_col="topic_true",
        labels=TOPIC_LABELS,
    ))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "openai_summary.csv", index=False)

    print(f"\n{'=' * 70}")
    print("OPENAI NLP EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"\nAll results saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
