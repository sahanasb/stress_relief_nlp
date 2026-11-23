import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
from werkzeug.security import generate_password_hash

from _common import PROJECT_ROOT, RESULTS_DIR, ensure_dir, save_json

OUT=ensure_dir(RESULTS_DIR/"database")
import database.db_manager as db


def main():
    checks=[]
    def record(name,status,details=""):
        checks.append({"check":name,"status":status,"details":details})

    original_path=db.DATABASE_PATH
    with tempfile.TemporaryDirectory() as td:
        test_db=str(Path(td)/"eval.sqlite3")
        db.DATABASE_PATH=test_db
        try:
            db.create_database(test_db); record("create_database","PASS")
            password="EvalPass!123"
            uid=db.create_user("eval_user",generate_password_hash(password)); record("create_user","PASS",f"user_id={uid}")
            user=db.retrieve_user("eval_user",password); record("login_correct_password","PASS" if user and user.get("user_id")==uid else "FAIL")
            wrong=db.retrieve_user("eval_user","wrong-password"); record("reject_wrong_password","PASS" if wrong is None else "FAIL")
            history=[{"role":"user","content":"I am stressed about an exam."},{"role":"assistant","content":"We can break it down together."}]
            db.save_conversation({"start_time":"2026-11-22T10:00:00","end_time":"2026-11-22T10:05:00","user_id":uid},history)
            record("save_conversation_function","PASS")
            con=sqlite3.connect(test_db); cur=con.cursor()
            conv=cur.execute("SELECT conversation_id,user_id FROM conversations").fetchone()
            msgs=cur.execute("SELECT role,content FROM messages ORDER BY message_id").fetchall(); con.close()
            record("conversation_persisted","PASS" if conv else "FAIL")
            record("conversation_owner_link","PASS" if conv and conv[1]==uid else "FAIL")
            record("messages_persisted","PASS" if msgs==[(m["role"],m["content"]) for m in history] else "FAIL",f"message_count={len(msgs)}")
        finally:
            db.DATABASE_PATH=original_path

    # Final integration checks: distinguish implemented DB primitives from app wiring.
    app_text=(PROJECT_ROOT/"app.py").read_text(encoding="utf-8")
    register_wired='@app.post("/api/register")' in app_text and "db.create_user(" in app_text
    login_wired='@app.post("/api/login")' in app_text and "db.retrieve_user(" in app_text
    record("app_register_route_wired","PASS" if register_wired else "FAIL")
    record("app_login_route_wired","PASS" if login_wired else "FAIL")
    record("app_calls_save_conversation","PASS" if "db.save_conversation(" in app_text else "NOT_IMPLEMENTED",
           "app.py does not currently call db.save_conversation" if "db.save_conversation(" not in app_text else "")
    retrieval_fn=any(hasattr(db,n) for n in ["retrieve_conversations","get_conversations","load_conversations","read_conversations"])
    record("conversation_retrieval_function","PASS" if retrieval_fn else "NOT_IMPLEMENTED","No conversation retrieval function found in database/db_manager.py" if not retrieval_fn else "")

    result=pd.DataFrame(checks); result.to_csv(OUT/"database_functional_checks.csv",index=False)
    metrics={
        "n_checks":int(len(result)),"passed":int((result.status=="PASS").sum()),"failed":int((result.status=="FAIL").sum()),
        "not_implemented":int((result.status=="NOT_IMPLEMENTED").sum()),
        "implemented_checks_pass_rate":float((result.status=="PASS").sum()/max(1,(result.status.isin(["PASS","FAIL"])).sum())),
        "app_conversation_persistence_integrated":bool((result.check.eq("app_calls_save_conversation") & result.status.eq("PASS")).any()),
        "conversation_retrieval_available":bool((result.check.eq("conversation_retrieval_function") & result.status.eq("PASS")).any()),
    }
    save_json(OUT/"database_metrics.json",metrics)
    print(result.to_string(index=False)); print("\n",metrics); print(f"Saved to {OUT}")


if __name__=="__main__": main()
