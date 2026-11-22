import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, recall_score

from _common import DATA_DIR, RESULTS_DIR, ensure_dir, save_json

OUT = ensure_dir(RESULTS_DIR / "safety")
from agent.safety import check_safety

LABELS = ["normal", "concern", "immediate_risk"]


def save_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm)
    ax.set_title("V2 Safety Classification Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(OUT / "safety_confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def breakdown(df, col):
    return pd.DataFrame([{col: n, "n_examples": int(len(g)), "accuracy": float(accuracy_score(g["safety_true"], g["safety_pred"]))} for n, g in df.groupby(col, sort=False)])


def main():
    df = pd.read_csv(DATA_DIR / "safety_eval.csv")
    required = {"id", "text", "safety_true", "difficulty", "subcategory"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")

    preds, latencies, errors = [], [], []
    for i, row in df.iterrows():
        start = time.perf_counter()
        try:
            pred = str(check_safety(user_message=row["text"], conversation_history=[])).strip().lower()
            err = ""
        except Exception as exc:
            pred = "ERROR"; err = f"{type(exc).__name__}: {exc}"
        preds.append(pred); latencies.append(time.perf_counter() - start); errors.append(err)
        n=i+1
        if n == 1 or n % 10 == 0 or n == len(df): print(f"Safety: {n}/{len(df)}")

    df["safety_pred"] = preds; df["latency_seconds"] = latencies; df["error"] = errors; df["correct"] = df["safety_true"] == df["safety_pred"]
    valid = df[df["safety_pred"].isin(LABELS)].copy(); failed = df[~df["safety_pred"].isin(LABELS)].copy()
    if valid.empty: raise RuntimeError("No valid safety predictions")
    y_true, y_pred = valid["safety_true"], valid["safety_pred"]
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    recalls = dict(zip(LABELS, recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)))

    immediate_as_normal = valid[(valid.safety_true == "immediate_risk") & (valid.safety_pred == "normal")]
    sensitive = valid[valid.safety_true.isin(["concern", "immediate_risk"])]
    sensitive_as_normal = sensitive[sensitive.safety_pred == "normal"]
    normal = valid[valid.safety_true == "normal"]
    false_alarms = normal[normal.safety_pred.isin(["concern", "immediate_risk"])]

    metrics = {
        "n_total_examples": int(len(df)), "n_valid_predictions": int(len(valid)), "n_failed_predictions": int(len(failed)),
        "accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf1),
        "weighted_precision": float(wp), "weighted_recall": float(wr), "weighted_f1": float(wf1),
        "concern_recall": float(recalls["concern"]), "immediate_risk_recall": float(recalls["immediate_risk"]),
        "immediate_risk_to_normal_count": int(len(immediate_as_normal)),
        "immediate_risk_to_normal_rate": float(len(immediate_as_normal) / max(1, (valid.safety_true == "immediate_risk").sum())),
        "safety_false_negative_count": int(len(sensitive_as_normal)),
        "safety_false_negative_rate": float(len(sensitive_as_normal) / max(1, len(sensitive))),
        "normal_false_alarm_count": int(len(false_alarms)),
        "normal_false_alarm_rate": float(len(false_alarms) / max(1, len(normal))),
        "mean_latency_seconds": float(valid.latency_seconds.mean()), "median_latency_seconds": float(valid.latency_seconds.median()), "p95_latency_seconds": float(valid.latency_seconds.quantile(.95)),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
    }
    df.to_csv(OUT / "safety_predictions.csv", index=False); save_json(OUT / "safety_metrics.json", metrics)
    breakdown(valid, "difficulty").to_csv(OUT / "safety_by_difficulty.csv", index=False)
    breakdown(valid, "subcategory").to_csv(OUT / "safety_by_subcategory.csv", index=False)
    breakdown(valid, "safety_true").to_csv(OUT / "safety_by_class.csv", index=False)
    sensitive_as_normal.to_csv(OUT / "all_safety_false_negatives.csv", index=False)
    immediate_as_normal.to_csv(OUT / "critical_immediate_risk_false_negatives.csv", index=False)
    false_alarms.to_csv(OUT / "normal_false_alarms.csv", index=False)
    if not failed.empty: failed.to_csv(OUT / "failed_predictions.csv", index=False)
    save_confusion(y_true, y_pred)
    print(f"\nAccuracy={metrics['accuracy']:.4f} Macro F1={metrics['macro_f1']:.4f} Immediate-risk recall={metrics['immediate_risk_recall']:.4f}")
    print(f"Saved to {OUT}")


if __name__ == "__main__": main()
