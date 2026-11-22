from pathlib import Path
import sys
import time

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / "data"
RESULTS_DIR = EVALUATION_DIR / "results"
LATENCY_DIR = RESULTS_DIR / "latency"

sys.path.insert(0, str(PROJECT_ROOT))
LATENCY_DIR.mkdir(parents=True, exist_ok=True)

from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic
from nlp.openai_analyzer import analyze_message


def load_all_unique_texts():
    frames = []
    for filename, task in [
        ("emotion_eval.csv", "emotion"),
        ("stress_eval.csv", "stress"),
        ("topic_eval.csv", "topic"),
    ]:
        df = pd.read_csv(DATA_DIR / filename)[["id", "text"]].copy()
        df["source_task"] = task
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Keep exact duplicate texts only once so repeated wording does not overweight latency.
    return combined.drop_duplicates(subset=["text"]).reset_index(drop=True)


def summarize(series):
    return {
        "n": int(series.notna().sum()),
        "mean_seconds": float(series.mean()),
        "median_seconds": float(series.median()),
        "p95_seconds": float(series.quantile(0.95)),
    }


def main():
    df = load_all_unique_texts()
    hf_latencies = []
    openai_latencies = []
    hf_errors = []
    openai_errors = []

    total = len(df)

    for index, row in df.iterrows():
        text = row["text"]

        # Real HF application analysis path: three models, sequentially.
        try:
            start = time.perf_counter()
            detect_emotion(text)
            detect_stress(text)
            detect_topic(text)
            hf_elapsed = time.perf_counter() - start
            hf_error = ""
        except Exception as exc:
            hf_elapsed = float("nan")
            hf_error = f"{type(exc).__name__}: {exc}"

        # Real current OpenAI application analysis path: one combined request.
        try:
            start = time.perf_counter()
            analyze_message(user_message=text, conversation_history=[])
            openai_elapsed = time.perf_counter() - start
            openai_error = ""
        except Exception as exc:
            openai_elapsed = float("nan")
            openai_error = f"{type(exc).__name__}: {exc}"

        hf_latencies.append(hf_elapsed)
        openai_latencies.append(openai_elapsed)
        hf_errors.append(hf_error)
        openai_errors.append(openai_error)

        current = index + 1
        if current == 1 or current % 10 == 0 or current == total:
            print(f"latency: {current}/{total}")

    df["hf_pipeline_latency_seconds"] = hf_latencies
    df["openai_pipeline_latency_seconds"] = openai_latencies
    df["hf_error"] = hf_errors
    df["openai_error"] = openai_errors
    df.to_csv(LATENCY_DIR / "pipeline_latency_raw.csv", index=False)

    hf_summary = summarize(df["hf_pipeline_latency_seconds"])
    openai_summary = summarize(df["openai_pipeline_latency_seconds"])

    summary = pd.DataFrame([
        {"method": "hugging_face_3_model_pipeline", **hf_summary},
        {"method": "openai_single_analysis_call", **openai_summary},
    ])

    hf_mean = hf_summary["mean_seconds"]
    openai_mean = openai_summary["mean_seconds"]
    summary["mean_speedup_vs_hf"] = [1.0, hf_mean / openai_mean if openai_mean else float("nan")]

    summary.to_csv(LATENCY_DIR / "pipeline_latency_summary.csv", index=False)

    print("\nPipeline latency comparison")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved to: {LATENCY_DIR}")


if __name__ == "__main__":
    main()
