import json
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from _common import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, ensure_dir, save_json

HF_DIR = ensure_dir(RESULTS_DIR / "hf")

from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
STRESS_LABELS = ["non_stress", "stress"]
TOPIC_LABELS = ["academic", "work or career", "relationships", "family", "financial", "health or wellness", "general"]


def predict_emotion(text):
    top = max(detect_emotion(text), key=lambda x: x["score"])
    return str(top["label"]).lower().strip(), float(top["score"])


def predict_stress(text):
    top = max(detect_stress(text), key=lambda x: x["score"])
    label = str(top["label"]).lower().strip()
    if label in {"not_stress", "not-stress", "no_stress"}:
        label = "non_stress"
    return label, float(top["score"])


def predict_topic(text):
    result = detect_topic(text)
    return str(result["labels"][0]).lower().strip(), float(result["scores"][0])


def save_confusion(y_true, y_pred, labels, title, path):
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
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def breakdown(df, group_col, true_col, pred_col):
    return pd.DataFrame([
        {group_col: name, "n_examples": int(len(group)), "accuracy": float(accuracy_score(group[true_col], group[pred_col]))}
        for name, group in df.groupby(group_col, sort=False)
    ])


def evaluate(task, filename, true_col, pred_col, score_col, labels, predictor):
    df = pd.read_csv(DATA_DIR / filename)
    required = {"id", "text", true_col, "difficulty", "subcategory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {sorted(missing)}")

    preds, scores, latencies, errors = [], [], [], []
    total = len(df)
    for i, row in df.iterrows():
        start = time.perf_counter()
        try:
            pred, score = predictor(row["text"])
            error = ""
        except Exception as exc:
            pred, score = "ERROR", float("nan")
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        preds.append(pred); scores.append(score); latencies.append(elapsed); errors.append(error)
        n = i + 1
        if n == 1 or n % 10 == 0 or n == total:
            print(f"{task}: {n}/{total}")

    df[pred_col] = preds
    df[score_col] = scores
    df["latency_seconds"] = latencies
    df["error"] = errors
    df["correct"] = df[true_col] == df[pred_col]
    valid = df[df[pred_col].isin(labels)].copy()
    failed = df[~df[pred_col].isin(labels)].copy()
    if valid.empty:
        raise RuntimeError(f"No valid {task} predictions were produced")

    y_true, y_pred = valid[true_col], valid[pred_col]
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    metrics = {
        "n_total_examples": int(len(df)),
        "n_valid_predictions": int(len(valid)),
        "n_failed_predictions": int(len(failed)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1),
        "mean_latency_seconds": float(valid["latency_seconds"].mean()),
        "median_latency_seconds": float(valid["latency_seconds"].median()),
        "p95_latency_seconds": float(valid["latency_seconds"].quantile(0.95)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }

    df.to_csv(HF_DIR / f"{task}_hf_predictions.csv", index=False)
    save_json(HF_DIR / f"{task}_hf_metrics.json", metrics)
    breakdown(valid, "difficulty", true_col, pred_col).to_csv(HF_DIR / f"{task}_hf_by_difficulty.csv", index=False)
    breakdown(valid, "subcategory", true_col, pred_col).to_csv(HF_DIR / f"{task}_hf_by_subcategory.csv", index=False)
    valid[~valid["correct"]].to_csv(HF_DIR / f"{task}_hf_misclassified.csv", index=False)
    if not failed.empty:
        failed.to_csv(HF_DIR / f"{task}_hf_failed.csv", index=False)
    save_confusion(y_true, y_pred, labels, f"V2 Hugging Face {task.title()} Confusion Matrix", HF_DIR / f"{task}_hf_confusion_matrix.png")

    print(f"{task}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return {"task": task, **{k: metrics[k] for k in ["n_total_examples", "accuracy", "macro_precision", "macro_recall", "macro_f1", "mean_latency_seconds", "median_latency_seconds", "p95_latency_seconds"]}}


def main():
    summaries = [
        evaluate("emotion", "emotion_eval.csv", "emotion_true", "emotion_pred", "emotion_confidence", EMOTION_LABELS, predict_emotion),
        evaluate("stress", "stress_eval.csv", "stress_true", "stress_pred", "stress_confidence", STRESS_LABELS, predict_stress),
        evaluate("topic", "topic_eval.csv", "topic_true", "topic_pred", "topic_confidence", TOPIC_LABELS, predict_topic),
    ]
    pd.DataFrame(summaries).to_csv(HF_DIR / "hf_summary.csv", index=False)
    print(f"\nSaved V2 HF results to {HF_DIR}")


if __name__ == "__main__":
    main()
