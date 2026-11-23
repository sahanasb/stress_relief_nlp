import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from _common import DATA_DIR, RESULTS_DIR, ensure_dir, parse_history, save_json

load_dotenv()

OUT = ensure_dir(RESULTS_DIR / "rag_groundedness")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5.6-luna")


def _json_from_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _judge(client, user_message, history, response, rag_sources):
    evidence_blocks = []
    for i, src in enumerate(rag_sources, start=1):
        evidence_blocks.append(
            f"SOURCE {i}\n"
            f"file: {src.get('source', '')}\n"
            f"section: {src.get('section', '')}\n"
            f"text: {src.get('text', '')}"
        )
    evidence = "\n\n".join(evidence_blocks) or "NO RETRIEVED KNOWLEDGE"
    history_text = json.dumps(history, ensure_ascii=False)

    prompt = f"""
You are evaluating a stress-support assistant that uses retrieval-augmented generation (RAG).

Judge ONLY the assistant response shown below. The retrieved knowledge passages are the grounding evidence.
The user's own message and prior conversation may also support personalized statements about the user's situation.

Do NOT penalize ordinary empathy, acknowledgement, questions, or conversational wording for not appearing in the knowledge base.
Do evaluate practical advice, factual claims, exercise instructions, timing/duration claims, escalation guidance, and other substantive guidance for support in the retrieved evidence.
Paraphrasing is expected; exact quotation is NOT required.

Definitions:
- groundedness_score (1-5): 5 means all substantive guidance is supported by the retrieved evidence or directly by the user's stated context; 1 means major unsupported or contradictory guidance.
- relevance_score (1-5): how well the response addresses the user's current request and situation.
- source_use_score (1-5): how meaningfully the response uses the retrieved knowledge when practical guidance was requested. A response that ignores clearly useful retrieved evidence should score low.
- hallucination_detected: true only if the response introduces a substantive claim/advice that is unsupported by or contradicts the evidence/user context.
- unsupported_claim_count: number of distinct substantive unsupported claims.

Return ONLY valid JSON with exactly these keys:
{{
  "groundedness_score": 1,
  "relevance_score": 1,
  "source_use_score": 1,
  "hallucination_detected": false,
  "unsupported_claim_count": 0,
  "reason": "one concise sentence"
}}

PRIOR CONVERSATION:
{history_text}

CURRENT USER MESSAGE:
{user_message}

RETRIEVED KNOWLEDGE:
{evidence}

ASSISTANT RESPONSE:
{response}
"""

    start = time.perf_counter()
    result = client.responses.create(model=JUDGE_MODEL, input=prompt)
    elapsed = time.perf_counter() - start
    parsed = _json_from_text(result.output_text)

    for key in ("groundedness_score", "relevance_score", "source_use_score"):
        parsed[key] = int(parsed[key])
        if parsed[key] < 1 or parsed[key] > 5:
            raise ValueError(f"Judge returned invalid {key}: {parsed[key]}")
    parsed["hallucination_detected"] = bool(parsed["hallucination_detected"])
    parsed["unsupported_claim_count"] = int(parsed["unsupported_claim_count"])
    parsed["reason"] = str(parsed.get("reason", "")).strip()
    return parsed, elapsed


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running this evaluation.")

    # Import after dotenv is loaded so every project component sees the key.
    from main import process_message

    client = OpenAI(api_key=api_key)
    df = pd.read_csv(DATA_DIR / "rag_groundedness_eval.csv")
    required = {
        "id", "query", "history", "expected_source", "difficulty",
        "subcategory", "guidance_focus"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    for i, row in df.iterrows():
        history = parse_history(row.history, row.id)
        working_history = [dict(m) for m in history]
        start = time.perf_counter()
        try:
            result = process_message(
                str(row.query),
                user_name="Jade",
                history=working_history,
            )
            pipeline_latency = time.perf_counter() - start
            rag_sources = result.get("rag_sources", []) or []
            retrieved = [x.get("source", "") for x in rag_sources]
            expected_source = str(row.expected_source).strip()
            source_retrieved = expected_source in retrieved

            judge_error = ""
            if not rag_sources:
                judged = {
                    "groundedness_score": 1,
                    "relevance_score": 1,
                    "source_use_score": 1,
                    "hallucination_detected": True,
                    "unsupported_claim_count": 0,
                    "reason": "No RAG evidence was retrieved for a case that requires grounded guidance.",
                }
                judge_latency = 0.0
            else:
                try:
                    judged, judge_latency = _judge(
                        client,
                        str(row.query),
                        history,
                        result.get("response", ""),
                        rag_sources,
                    )
                except Exception as exc:
                    judge_error = f"{type(exc).__name__}: {exc}"
                    judge_latency = 0.0
                    judged = {
                        "groundedness_score": 0,
                        "relevance_score": 0,
                        "source_use_score": 0,
                        "hallucination_detected": True,
                        "unsupported_claim_count": 0,
                        "reason": "Judge error",
                    }

            faithfulness_pass = bool(
                source_retrieved
                and not judge_error
                and judged["groundedness_score"] >= 4
                and judged["relevance_score"] >= 4
                and judged["source_use_score"] >= 3
                and not judged["hallucination_detected"]
            )

            rows.append({
                **row.to_dict(),
                "actual_safety": result.get("safety_state"),
                "actual_state": result.get("conversation_state"),
                "response": result.get("response", ""),
                "retrieved_sources": json.dumps(retrieved),
                "expected_source_retrieved": source_retrieved,
                **judged,
                "faithfulness_pass": faithfulness_pass,
                "pipeline_latency_seconds": pipeline_latency,
                "judge_latency_seconds": judge_latency,
                "error": judge_error,
            })
        except Exception as exc:
            rows.append({
                **row.to_dict(),
                "actual_safety": "ERROR",
                "actual_state": "ERROR",
                "response": "",
                "retrieved_sources": "[]",
                "expected_source_retrieved": False,
                "groundedness_score": 0,
                "relevance_score": 0,
                "source_use_score": 0,
                "hallucination_detected": True,
                "unsupported_claim_count": 0,
                "reason": "Pipeline error",
                "faithfulness_pass": False,
                "pipeline_latency_seconds": time.perf_counter() - start,
                "judge_latency_seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })

        n = i + 1
        status = "PASS" if rows[-1]["faithfulness_pass"] else "FAIL"
        print(f"Groundedness {n}/{len(df)}: {row.id} {status}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "rag_groundedness_results.csv", index=False)
    out_df[~out_df.faithfulness_pass].to_csv(OUT / "rag_groundedness_failures.csv", index=False)

    valid_judged = out_df[out_df.error.fillna("") == ""]
    metrics = {
        "judge_model": JUDGE_MODEL,
        "n_examples": int(len(out_df)),
        "expected_source_retrieval_rate": float(out_df.expected_source_retrieved.mean()),
        "mean_groundedness_score": float(valid_judged.groundedness_score.mean()) if len(valid_judged) else None,
        "mean_relevance_score": float(valid_judged.relevance_score.mean()) if len(valid_judged) else None,
        "mean_source_use_score": float(valid_judged.source_use_score.mean()) if len(valid_judged) else None,
        "hallucination_rate": float(valid_judged.hallucination_detected.mean()) if len(valid_judged) else None,
        "mean_unsupported_claim_count": float(valid_judged.unsupported_claim_count.mean()) if len(valid_judged) else None,
        "faithfulness_pass_rate": float(out_df.faithfulness_pass.mean()),
        "mean_pipeline_latency_seconds": float(out_df.pipeline_latency_seconds.mean()),
        "mean_judge_latency_seconds": float(out_df.judge_latency_seconds.mean()),
        "n_errors": int((out_df.error.fillna("") != "").sum()),
    }
    save_json(OUT / "rag_groundedness_metrics.json", metrics)

    for col in ("difficulty", "subcategory", "expected_source"):
        summary = []
        for name, g in out_df.groupby(col, sort=False):
            ok = g[g.error.fillna("") == ""]
            summary.append({
                col: name,
                "n_examples": int(len(g)),
                "source_retrieval_rate": float(g.expected_source_retrieved.mean()),
                "mean_groundedness_score": float(ok.groundedness_score.mean()) if len(ok) else None,
                "faithfulness_pass_rate": float(g.faithfulness_pass.mean()),
            })
        pd.DataFrame(summary).to_csv(OUT / f"rag_groundedness_by_{col}.csv", index=False)

    print("\nRAG groundedness / faithfulness results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nNote: this test uses an LLM-as-judge. Treat scores as evaluation evidence, not absolute ground truth.")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
