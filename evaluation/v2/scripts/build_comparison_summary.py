import json
from pathlib import Path

import pandas as pd

from _common import EVAL_DIR, RESULTS_DIR

V1=EVAL_DIR.parent/"v1"/"results"
V2=RESULTS_DIR


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def add(rows,metric,v1,v2,note=""):
    if v2 is None: return
    abs_change=(v2-v1) if v1 is not None else None
    rel=(abs_change/v1*100) if v1 not in (None,0) else None
    rows.append({"metric":metric,"v1":v1,"v2":v2,"absolute_change":abs_change,"relative_change_pct":rel,"note":note})


def main():
    rows=[]
    v1_hf=V1/"hf_first_version"/"hf_summary.csv"; v2_hf=V2/"hf"/"hf_summary.csv"
    if v2_hf.exists():
        b=pd.read_csv(v1_hf).set_index("task") if v1_hf.exists() else None; n=pd.read_csv(v2_hf).set_index("task")
        for task in ["emotion","stress","topic"]:
            add(rows,f"{task}_accuracy",float(b.loc[task,"accuracy"]) if b is not None else None,float(n.loc[task,"accuracy"]),"same fixed V1 dataset")
            add(rows,f"{task}_macro_f1",float(b.loc[task,"macro_f1"]) if b is not None else None,float(n.loc[task,"macro_f1"]),"same fixed V1 dataset")

    b=read_json(V1/"safety_first_version"/"safety_metrics.json"); n=read_json(V2/"safety"/"safety_metrics.json")
    if n:
        for k in ["accuracy","macro_f1","concern_recall","immediate_risk_recall","safety_false_negative_rate","normal_false_alarm_rate"]: add(rows,f"safety_{k}",b.get(k) if b else None,n.get(k),"same fixed V1 dataset")

    b=read_json(V1/"conversation_state_first_version"/"conversation_state_metrics.json"); n=read_json(V2/"conversation_state"/"conversation_state_metrics.json")
    if n:
        for k in ["accuracy","macro_f1","continue_recall","offer_to_close_recall","close_recall"]: add(rows,f"conversation_{k}",b.get(k) if b else None,n.get(k),"same fixed V1 dataset")

    b=read_json(V1/"context_retention_first_version"/"context_retention_metrics.json"); n=read_json(V2/"context_retention"/"context_retention_metrics.json")
    if n: add(rows,"context_retention_accuracy",b.get("strict_context_recall_accuracy") if b else None,n.get("strict_context_recall_accuracy"),"same 30 V1 scenarios; strict keyword metric")

    v1lat=V1/"latency_first_version"/"pipeline_latency_summary_before.csv"; v2lat=V2/"latency"/"pipeline_latency_summary.csv"
    if v2lat.exists():
        nb=pd.read_csv(v2lat); nr=nb[nb.method=="hugging_face_3_model_pipeline"].iloc[0]
        br=None
        if v1lat.exists():
            bt=pd.read_csv(v1lat); q=bt[bt.method=="hugging_face_3_model_pipeline"]; br=q.iloc[0] if not q.empty else None
        for col,label in [("mean_seconds","mean"),("median_seconds","median"),("p95_seconds","p95")]: add(rows,f"hf_pipeline_{label}_seconds",float(br[col]) if br is not None else None,float(nr[col]),"same 580-message latency corpus")

    rag=read_json(V2/"rag"/"rag_metrics.json")
    if rag:
        for k in ["hit_at_1","hit_at_3","mrr","irrelevant_rejection_accuracy"]: add(rows,f"rag_{k}",None,rag.get(k),"new in V2")
    db=read_json(V2/"database"/"database_metrics.json")
    if db: add(rows,"database_implemented_checks_pass_rate",None,db.get("implemented_checks_pass_rate"),f"new in V2; not_implemented={db.get('not_implemented')}")
    smoke=read_json(V2/"smoke"/"smoke_metrics.json")
    if smoke: add(rows,"integrated_smoke_pass_rate",None,smoke.get("pass_rate"),"new in V2")

    out=pd.DataFrame(rows); out.to_csv(V2/"v1_vs_v2_summary.csv",index=False)
    print(out.to_string(index=False)); print(f"\nSaved to {V2/'v1_vs_v2_summary.csv'}")


if __name__=="__main__": main()
