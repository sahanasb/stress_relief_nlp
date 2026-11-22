import time

import pandas as pd

from _common import DATA_DIR, EVAL_DIR, RESULTS_DIR, ensure_dir

OUT = ensure_dir(RESULTS_DIR / "latency")
from nlp.emotion_detector import detect_emotion
from nlp.stress_classifier import detect_stress
from nlp.topic_classifier import detect_topic


def load_texts():
    frames=[]
    for filename,task in [("emotion_eval.csv","emotion"),("stress_eval.csv","stress"),("topic_eval.csv","topic")]:
        d=pd.read_csv(DATA_DIR/filename)[["id","text"]].copy(); d["source_task"]=task; frames.append(d)
    return pd.concat(frames,ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)


def summarize(s):
    return {"n":int(s.notna().sum()),"mean_seconds":float(s.mean()),"median_seconds":float(s.median()),"p95_seconds":float(s.quantile(.95))}


def main():
    df=load_texts(); emo=[]; stress=[]; topic=[]; total=[]; errors=[]
    for i,row in df.iterrows():
        text=row.text; err=""
        try:
            total_start=time.perf_counter()
            t0=time.perf_counter(); detect_emotion(text); e=time.perf_counter()-t0
            t0=time.perf_counter(); detect_stress(text); s=time.perf_counter()-t0
            t0=time.perf_counter(); detect_topic(text); t=time.perf_counter()-t0
            tot=time.perf_counter()-total_start
        except Exception as exc:
            e=s=t=tot=float("nan"); err=f"{type(exc).__name__}: {exc}"
        emo.append(e); stress.append(s); topic.append(t); total.append(tot); errors.append(err)
        n=i+1
        if n==1 or n%10==0 or n==len(df): print(f"Latency: {n}/{len(df)}")
    df["emotion_seconds"]=emo; df["stress_seconds"]=stress; df["topic_seconds"]=topic; df["hf_pipeline_seconds"]=total; df["error"]=errors
    df.to_csv(OUT/"pipeline_latency_raw.csv",index=False)
    rows=[]
    for method,col in [("emotion_model","emotion_seconds"),("stress_model","stress_seconds"),("topic_model","topic_seconds"),("hugging_face_3_model_pipeline","hf_pipeline_seconds")]:
        rows.append({"method":method,**summarize(df[col])})
    summary=pd.DataFrame(rows)

    # Add direct V1 deltas when the archived summary is available.
    v1_path=EVAL_DIR.parent/"v1"/"results"/"latency_first_version"/"pipeline_latency_summary_before.csv"
    if v1_path.exists():
        v1=pd.read_csv(v1_path)
        base=v1[v1.method=="hugging_face_3_model_pipeline"]
        if not base.empty:
            b=base.iloc[0]
            mask=summary.method=="hugging_face_3_model_pipeline"
            summary.loc[mask,"v1_mean_seconds"]=float(b.mean_seconds)
            summary.loc[mask,"v1_median_seconds"]=float(b.median_seconds)
            summary.loc[mask,"v1_p95_seconds"]=float(b.p95_seconds)
            summary.loc[mask,"mean_change_pct"]=(summary.loc[mask,"mean_seconds"]-float(b.mean_seconds))/float(b.mean_seconds)*100
            summary.loc[mask,"median_change_pct"]=(summary.loc[mask,"median_seconds"]-float(b.median_seconds))/float(b.median_seconds)*100
            summary.loc[mask,"p95_change_pct"]=(summary.loc[mask,"p95_seconds"]-float(b.p95_seconds))/float(b.p95_seconds)*100
    summary.to_csv(OUT/"pipeline_latency_summary.csv",index=False)
    print("\n"+summary.to_string(index=False)); print(f"\nSaved to {OUT}")


if __name__=="__main__": main()
