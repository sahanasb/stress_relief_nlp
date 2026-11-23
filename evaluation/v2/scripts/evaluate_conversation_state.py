import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, recall_score

from _common import DATA_DIR, RESULTS_DIR, ensure_dir, parse_history, save_json

OUT = ensure_dir(RESULTS_DIR / "conversation_state")
from agent.conversation_state import detect_conversation_state

LABELS = ["continue", "offer_to_close", "close"]


def save_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6)); image = ax.imshow(cm)
    ax.set_title("V2 Conversation State Confusion Matrix"); ax.set_xlabel("Predicted state"); ax.set_ylabel("True state")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=30, ha="right"); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(OUT / "conversation_state_confusion_matrix.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def breakdown(df, col):
    return pd.DataFrame([{col: n, "n_examples": int(len(g)), "accuracy": float(accuracy_score(g.state_true, g.state_pred))} for n,g in df.groupby(col, sort=False)])


def main():
    df = pd.read_csv(DATA_DIR / "conversation_state_eval.csv")
    required = {"id", "history", "current_message", "state_true", "difficulty", "subcategory"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")

    preds, latencies, errors = [], [], []
    for i,row in df.iterrows():
        start=time.perf_counter()
        try:
            history=parse_history(row.history,row.id)
            pred=str(detect_conversation_state(user_message=row.current_message, conversation_history=history)).strip().lower(); err=""
        except Exception as exc:
            pred="ERROR"; err=f"{type(exc).__name__}: {exc}"
        preds.append(pred); latencies.append(time.perf_counter()-start); errors.append(err)
        n=i+1
        if n==1 or n%10==0 or n==len(df): print(f"Conversation state: {n}/{len(df)}")

    df["state_pred"]=preds; df["latency_seconds"]=latencies; df["error"]=errors; df["correct"]=df.state_true==df.state_pred
    valid=df[df.state_pred.isin(LABELS)].copy(); failed=df[~df.state_pred.isin(LABELS)].copy()
    if valid.empty: raise RuntimeError("No valid conversation-state predictions")
    y_true,y_pred=valid.state_true,valid.state_pred
    mp,mr,mf1,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average="macro",zero_division=0)
    wp,wr,wf1,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average="weighted",zero_division=0)
    recalls=dict(zip(LABELS, recall_score(y_true,y_pred,labels=LABELS,average=None,zero_division=0)))
    c2c=valid[(valid.state_true=="continue")&(valid.state_pred=="close")]
    o2c=valid[(valid.state_true=="offer_to_close")&(valid.state_pred=="close")]
    c2cont=valid[(valid.state_true=="close")&(valid.state_pred=="continue")]
    c2offer=valid[(valid.state_true=="close")&(valid.state_pred=="offer_to_close")]
    ncont=(valid.state_true=="continue").sum(); noffer=(valid.state_true=="offer_to_close").sum(); nclose=(valid.state_true=="close").sum()
    metrics={
        "n_total_examples":int(len(df)),"n_valid_predictions":int(len(valid)),"n_failed_predictions":int(len(failed)),
        "accuracy":float(accuracy_score(y_true,y_pred)),"macro_precision":float(mp),"macro_recall":float(mr),"macro_f1":float(mf1),
        "weighted_precision":float(wp),"weighted_recall":float(wr),"weighted_f1":float(wf1),
        "continue_recall":float(recalls["continue"]),"offer_to_close_recall":float(recalls["offer_to_close"]),"close_recall":float(recalls["close"]),
        "continue_to_close_count":int(len(c2c)),"continue_to_close_rate":float(len(c2c)/max(1,ncont)),
        "offer_to_close_to_close_count":int(len(o2c)),"offer_to_close_to_close_rate":float(len(o2c)/max(1,noffer)),
        "close_to_continue_count":int(len(c2cont)),"close_to_continue_rate":float(len(c2cont)/max(1,nclose)),
        "close_to_offer_count":int(len(c2offer)),"close_to_offer_rate":float(len(c2offer)/max(1,nclose)),
        "mean_latency_seconds":float(valid.latency_seconds.mean()),"median_latency_seconds":float(valid.latency_seconds.median()),"p95_latency_seconds":float(valid.latency_seconds.quantile(.95)),
        "classification_report":classification_report(y_true,y_pred,labels=LABELS,output_dict=True,zero_division=0),
    }
    df.to_csv(OUT/"conversation_state_predictions.csv",index=False); save_json(OUT/"conversation_state_metrics.json",metrics)
    breakdown(valid,"difficulty").to_csv(OUT/"conversation_state_by_difficulty.csv",index=False)
    breakdown(valid,"subcategory").to_csv(OUT/"conversation_state_by_subcategory.csv",index=False)
    breakdown(valid,"state_true").to_csv(OUT/"conversation_state_by_class.csv",index=False)
    valid[~valid.correct].to_csv(OUT/"conversation_state_misclassified.csv",index=False)
    c2c.to_csv(OUT/"critical_continue_to_close.csv",index=False); c2cont.to_csv(OUT/"critical_close_to_continue.csv",index=False)
    if not failed.empty: failed.to_csv(OUT/"failed_predictions.csv",index=False)
    save_confusion(y_true,y_pred)
    print(f"\nAccuracy={metrics['accuracy']:.4f} Macro F1={metrics['macro_f1']:.4f} Close recall={metrics['close_recall']:.4f}")
    print(f"Saved to {OUT}")


if __name__=="__main__": main()
