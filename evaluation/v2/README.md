# Evaluation V2

This folder evaluates the final integrated SerenAI code while preserving V1 results under `evaluation/v1/`.

## What is repeated for apples-to-apples comparison

- Hugging Face emotion / stress / topic classification on the exact V1 datasets
- Safety classification on the exact V1 dataset
- Conversation-state classification on the exact V1 dataset
- Context-retention evaluation on the exact V1 scenarios
- Hugging Face 3-model NLP latency on the same combined 580-message corpus

Human evaluation is **not repeated**. The prior V1 human evaluation should be reported as prior evidence with the limitation that it predates the final RAG/prompt/UI refinements.

The stale `origin/openai-nlp` branch is **not rerun** as a V2 classifier baseline.

## New V2-only tests

- RAG retrieval: 30 fixed queries; Hit@1, Hit@3, MRR, irrelevant-query rejection
- Database functional checks: user creation, login, wrong-password rejection, conversation persistence, ownership, and explicit reporting of missing app/retrieval integration
- Integrated core smoke test: 10 representative end-to-end `process_message()` scenarios

## Install evaluation-only dependencies

From the repository root:

```bash
pip install -r evaluation/v2/requirements.txt
```

The project dependencies in the root `requirements.txt` must also already be installed.

## Recommended run order

```bash
python evaluation/v2/scripts/evaluate_hf.py
python evaluation/v2/scripts/evaluate_safety.py
python evaluation/v2/scripts/evaluate_conversation_state.py
python evaluation/v2/scripts/evaluate_context_retention.py
python evaluation/v2/scripts/evaluate_pipeline_latency.py
python evaluation/v2/scripts/evaluate_rag.py
python evaluation/v2/scripts/evaluate_database.py
python evaluation/v2/scripts/evaluate_smoke.py
python evaluation/v2/scripts/build_comparison_summary.py
```

Safety, conversation-state, context-retention, production RAG embeddings, and smoke tests require the project's OpenAI configuration/API key.

## RAG dry run

To validate the RAG evaluator without OpenAI embeddings:

```bash
python evaluation/v2/scripts/evaluate_rag.py --offline
```

Do **not** use the offline fake-embedding score as the final presentation metric. Run `evaluate_rag.py` without `--offline` for the production retrieval result.

## Important implementation observations from the final branch

- `main.py` still executes the three Hugging Face classifiers sequentially: emotion -> stress -> topic. Therefore V2 latency is reported as a sequential 3-model pipeline and should not be described as parallel unless the tested code changes.
- RAG is integrated into `process_message()` before response generation.
- The database module implements create user, login lookup, and save conversation primitives.
- In the inspected final branch, `app.py` does not call `db.save_conversation()`, and `database/db_manager.py` does not expose a conversation-retrieval function. The database evaluator reports these as `NOT_IMPLEMENTED` instead of pretending the integration is complete.
- Twilio is on a separate branch and is outside these core V2 scripts.
