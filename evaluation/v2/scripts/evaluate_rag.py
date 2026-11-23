import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from _common import DATA_DIR, EVAL_DIR, RESULTS_DIR, ensure_dir, parse_history, save_json


def main():
    parser=argparse.ArgumentParser(description="Evaluate SerenAI RAG retrieval")
    parser.add_argument("--offline", action="store_true", help="Use deterministic fake embeddings instead of OpenAI embeddings")
    args=parser.parse_args()
    if args.offline:
        os.environ["RAG_FAKE_EMBED"]="1"

    from rag import store

    if args.offline:
        # Keep the dry-run index isolated from the production RAG store.
        store.STORE_DIR=EVAL_DIR/".rag_store_eval_fake"
        store._local=None; store._bm25=None; store._qcache.clear()
        store.build_index()

    out=ensure_dir(RESULTS_DIR/"rag")
    df=pd.read_csv(DATA_DIR/"rag_eval.csv")
    required={"id","query","history","topic","expected_source","should_retrieve","difficulty","subcategory"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")

    records=[]
    for i,row in df.iterrows():
        history=parse_history(row.history,row.id)
        should_retrieve=str(row.should_retrieve).strip().lower() in {"true","1","yes"}
        expected="" if pd.isna(row.expected_source) else str(row.expected_source).strip()
        start=time.perf_counter()
        try:
            hits=store.retrieve_knowledge(str(row.query),history,str(row.topic))
            err=""
        except Exception as exc:
            hits=[]; err=f"{type(exc).__name__}: {exc}"
        elapsed=time.perf_counter()-start
        sources=[h.get("source","") for h in hits]
        rank=(sources.index(expected)+1) if expected and expected in sources else None
        hit1=bool(should_retrieve and rank==1)
        hit3=bool(should_retrieve and rank is not None and rank<=3)
        rr=(1.0/rank) if rank else 0.0
        rejected=bool((not should_retrieve) and len(hits)==0)
        records.append({
            **row.to_dict(),"n_results":len(hits),"top1_source":sources[0] if sources else "",
            "retrieved_sources":json.dumps(sources),"expected_rank":rank,"hit_at_1":hit1,"hit_at_3":hit3,
            "reciprocal_rank":rr,"irrelevant_rejected":rejected,"latency_seconds":elapsed,"error":err,
        })
        n=i+1
        if n==1 or n%5==0 or n==len(df): print(f"RAG: {n}/{len(df)}")

    pred=pd.DataFrame(records)
    positives=pred[pred.should_retrieve.astype(str).str.lower().isin(["true","1","yes"])].copy()
    negatives=pred[~pred.should_retrieve.astype(str).str.lower().isin(["true","1","yes"])].copy()
    metrics={
        "mode":"offline_fake_embeddings" if args.offline else "production_embeddings",
        "n_examples":int(len(pred)),"n_positive_queries":int(len(positives)),"n_irrelevant_queries":int(len(negatives)),
        "hit_at_1":float(positives.hit_at_1.mean()) if len(positives) else None,
        "hit_at_3":float(positives.hit_at_3.mean()) if len(positives) else None,
        "mrr":float(positives.reciprocal_rank.mean()) if len(positives) else None,
        "irrelevant_rejection_accuracy":float(negatives.irrelevant_rejected.mean()) if len(negatives) else None,
        "mean_latency_seconds":float(pred.latency_seconds.mean()),"median_latency_seconds":float(pred.latency_seconds.median()),"p95_latency_seconds":float(pred.latency_seconds.quantile(.95)),
        "n_errors":int((pred.error!="").sum()),
    }
    pred.to_csv(out/"rag_predictions.csv",index=False); save_json(out/"rag_metrics.json",metrics)
    for col in ["difficulty","subcategory"]:
        rows=[]
        for name,g in positives.groupby(col,sort=False):
            rows.append({col:name,"n_examples":int(len(g)),"hit_at_1":float(g.hit_at_1.mean()),"hit_at_3":float(g.hit_at_3.mean()),"mrr":float(g.reciprocal_rank.mean())})
        pd.DataFrame(rows).to_csv(out/f"rag_by_{col}.csv",index=False)
    positives[~positives.hit_at_3].to_csv(out/"rag_misses.csv",index=False)
    negatives[~negatives.irrelevant_rejected].to_csv(out/"rag_irrelevant_false_positives.csv",index=False)
    print("\nRAG retrieval results")
    for k,v in metrics.items(): print(f"{k}: {v}")
    print(f"Saved to {out}")


if __name__=="__main__": main()
