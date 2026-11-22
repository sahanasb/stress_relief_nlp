import time

import pandas as pd

from _common import DATA_DIR, RESULTS_DIR, ensure_dir, parse_history, save_json

OUT = ensure_dir(RESULTS_DIR / "context_retention")
from agent.stress_agent import generate_response


def normalize(text):
    return " ".join(str(text).lower().strip().split())


def call_agent(current_message, history):
    # Keep auxiliary signals fixed exactly as in V1 so the variable under test is history usage.
    return generate_response(
        user_message=current_message,
        emotion="neutral", emotion_score=1.0,
        stress_label="non_stress", stress_score=1.0,
        topic="general", topic_score=1.0,
        conversation_history=history,
        conversation_state="continue",
        knowledge=None,
    )


def breakdown(df, col):
    return pd.DataFrame([{col:n,"n_examples":int(len(g)),"strict_accuracy":float(g.strict_correct.mean())} for n,g in df.groupby(col,sort=False)])


def main():
    df=pd.read_csv(DATA_DIR/"context_retention_eval.csv")
    required={"id","history","current_message","expected_term","expected_answer","difficulty","subcategory"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")

    responses,passed,latencies,errors=[],[],[],[]
    for i,row in df.iterrows():
        start=time.perf_counter()
        try:
            history=parse_history(row.history,row.id)
            response=str(call_agent(row.current_message,history)); err=""
            ok=normalize(row.expected_term) in normalize(response)
        except Exception as exc:
            response=""; ok=False; err=f"{type(exc).__name__}: {exc}"
        responses.append(response); passed.append(bool(ok)); latencies.append(time.perf_counter()-start); errors.append(err)
        n=i+1
        if n==1 or n%5==0 or n==len(df): print(f"Context retention: {n}/{len(df)}")

    df["agent_response"]=responses; df["strict_correct"]=passed; df["latency_seconds"]=latencies; df["error"]=errors
    valid=df[df.error==""].copy(); failed=df[df.error!=""].copy(); failures=valid[~valid.strict_correct].copy()
    metrics={
        "n_total_examples":int(len(df)),"n_valid_predictions":int(len(valid)),"n_failed_predictions":int(len(failed)),
        "strict_context_recall_accuracy":float(valid.strict_correct.mean()) if len(valid) else 0.0,
        "strict_pass_count":int(valid.strict_correct.sum()),"strict_fail_count":int((~valid.strict_correct).sum()),
        "mean_latency_seconds":float(valid.latency_seconds.mean()) if len(valid) else None,
        "median_latency_seconds":float(valid.latency_seconds.median()) if len(valid) else None,
        "p95_latency_seconds":float(valid.latency_seconds.quantile(.95)) if len(valid) else None,
    }
    df.to_csv(OUT/"context_retention_predictions.csv",index=False); save_json(OUT/"context_retention_metrics.json",metrics)
    breakdown(valid,"difficulty").to_csv(OUT/"context_retention_by_difficulty.csv",index=False)
    breakdown(valid,"subcategory").to_csv(OUT/"context_retention_by_subcategory.csv",index=False)
    failures.to_csv(OUT/"context_retention_strict_failures.csv",index=False)
    review=failures[["id","history","current_message","expected_answer","expected_term","agent_response","difficulty","subcategory"]].copy()
    review["human_review"]=""; review["review_notes"]=""; review.to_csv(OUT/"context_retention_manual_review.csv",index=False)
    if not failed.empty: failed.to_csv(OUT/"failed_predictions.csv",index=False)
    print(f"\nStrict context recall={metrics['strict_context_recall_accuracy']:.4f}; manual-review cases={metrics['strict_fail_count']}")
    print(f"Saved to {OUT}")


if __name__=="__main__": main()
