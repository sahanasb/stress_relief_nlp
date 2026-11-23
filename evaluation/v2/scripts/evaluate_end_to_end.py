import json
import time

import pandas as pd
from dotenv import load_dotenv

from _common import DATA_DIR, RESULTS_DIR, ensure_dir, parse_history, save_json

load_dotenv()
OUT = ensure_dir(RESULTS_DIR / "end_to_end")


def _text_or_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def _bool_value(value, default=False):
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"true", "1", "yes"}


def _rag_match(expected, actual_sources):
    if expected is None:
        return True
    acceptable = [x.strip() for x in expected.split("|") if x.strip()]
    return any(source in actual_sources for source in acceptable)


def main():
    # Import after dotenv so API-backed components see OPENAI_API_KEY.
    from main import process_message

    df = pd.read_csv(DATA_DIR / "end_to_end_eval.csv")
    required = {
        "id", "category", "difficulty", "text", "history",
        "expected_safety", "expected_state", "expected_rag",
        "expected_should_close"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    for i, row in df.iterrows():
        history = parse_history(row.history, row.id)
        working_history = [dict(m) for m in history]
        before = len(working_history)
        expected_safety = _text_or_none(row.expected_safety)
        expected_state = _text_or_none(row.expected_state)
        expected_rag = _text_or_none(row.expected_rag)
        expected_close = _bool_value(row.expected_should_close)

        start = time.perf_counter()
        try:
            result = process_message(
                str(row.text),
                user_name="Jade",
                history=working_history,
            )
            elapsed = time.perf_counter() - start
            actual_sources = [x.get("source", "") for x in (result.get("rag_sources", []) or [])]

            checks = {
                "safety": result.get("safety_state") == expected_safety,
                "conversation_state": result.get("conversation_state") == expected_state,
                "should_close": bool(result.get("should_close")) == expected_close,
                "response_nonempty": bool(str(result.get("response", "")).strip()),
                "history_updated": len(working_history) == before + 2,
                "rag": _rag_match(expected_rag, actual_sources),
            }
            passed = all(checks.values())
            record = {
                **row.to_dict(),
                "actual_safety": result.get("safety_state"),
                "actual_state": result.get("conversation_state"),
                "actual_should_close": bool(result.get("should_close")),
                "actual_rag_sources": json.dumps(actual_sources),
                "response": result.get("response", ""),
                **{f"check_{k}": v for k, v in checks.items()},
                "passed": passed,
                "latency_seconds": elapsed,
                "error": "",
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start
            record = {
                **row.to_dict(),
                "actual_safety": "ERROR",
                "actual_state": "ERROR",
                "actual_should_close": False,
                "actual_rag_sources": "[]",
                "response": "",
                "check_safety": False,
                "check_conversation_state": False,
                "check_should_close": False,
                "check_response_nonempty": False,
                "check_history_updated": False,
                "check_rag": False,
                "passed": False,
                "latency_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(record)
        print(f"End-to-end {i+1}/{len(df)}: {row.id} {'PASS' if record['passed'] else 'FAIL'}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "end_to_end_results.csv", index=False)
    results[~results.passed].to_csv(OUT / "end_to_end_failures.csv", index=False)

    check_cols = [
        "check_safety", "check_conversation_state", "check_should_close",
        "check_response_nonempty", "check_history_updated", "check_rag"
    ]
    metrics = {
        "n_scenarios": int(len(results)),
        "passed": int(results.passed.sum()),
        "failed": int((~results.passed).sum()),
        "pass_rate": float(results.passed.mean()),
        "mean_latency_seconds": float(results.latency_seconds.mean()),
        "median_latency_seconds": float(results.latency_seconds.median()),
        "p95_latency_seconds": float(results.latency_seconds.quantile(0.95)),
        "n_errors": int((results.error.fillna("") != "").sum()),
    }
    for col in check_cols:
        metrics[f"{col}_pass_rate"] = float(results[col].mean())
    save_json(OUT / "end_to_end_metrics.json", metrics)

    for group_col in ("category", "difficulty"):
        summary = []
        for name, g in results.groupby(group_col, sort=False):
            item = {
                group_col: name,
                "n_scenarios": int(len(g)),
                "passed": int(g.passed.sum()),
                "pass_rate": float(g.passed.mean()),
                "mean_latency_seconds": float(g.latency_seconds.mean()),
                "n_errors": int((g.error.fillna("") != "").sum()),
            }
            for col in check_cols:
                item[f"{col}_pass_rate"] = float(g[col].mean())
            summary.append(item)
        pd.DataFrame(summary).to_csv(OUT / f"end_to_end_by_{group_col}.csv", index=False)

    print("\nExpanded end-to-end results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
