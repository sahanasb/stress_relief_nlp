from pathlib import Path
import sys
import json
import time
import inspect

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVALUATION_DIR.parent
DATA_DIR = EVALUATION_DIR / 'data'
RESULTS_DIR = EVALUATION_DIR / 'results' / 'context_retention'

sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the ACTUAL response generator from the project.
from agent.stress_agent import generate_response


def call_project_agent(current_message, history):
    """Call the project's real generate_response() while keeping NLP signals fixed.

    Fixed auxiliary NLP signals isolate the variable we care about here:
    whether the response generator correctly uses conversation history.
    """
    params = inspect.signature(generate_response).parameters

    # Current branch signature.
    if 'emotion' in params:
        return generate_response(
            user_message=current_message,
            emotion='neutral',
            emotion_score=1.0,
            stress_label='non_stress',
            stress_score=1.0,
            topic='general',
            topic_score=1.0,
            conversation_history=history,
            conversation_state='continue',
        )

    # Forward-compatible fallback if stress_agent.py is later updated to
    # accept the combined OpenAI analysis object used by main.py.
    if 'analysis' in params:
        return generate_response(
            user_message=current_message,
            analysis={
                'emotion': 'neutral',
                'stress': 'no_stress',
                'topic': 'general',
            },
            conversation_history=history,
            conversation_state='continue',
        )

    raise RuntimeError(
        'Unsupported generate_response() signature. '
        'Please inspect agent/stress_agent.py and update this adapter.'
    )


def normalize(text):
    return ' '.join(str(text).lower().strip().split())


def breakdown(df, column):
    rows = []
    for name, group in df.groupby(column, sort=False):
        rows.append({
            column: name,
            'n_examples': len(group),
            'strict_accuracy': group['strict_correct'].mean(),
        })
    return pd.DataFrame(rows)


def main():
    csv_path = DATA_DIR / 'context_retention_eval.csv'
    df = pd.read_csv(csv_path)

    required = {
        'id', 'history', 'current_message', 'expected_term',
        'expected_answer', 'difficulty', 'subcategory'
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    responses = []
    strict_correct = []
    latencies = []
    errors = []

    total = len(df)

    for index, row in df.iterrows():
        history = json.loads(row['history'])
        current_message = row['current_message']
        expected_term = normalize(row['expected_term'])

        try:
            start = time.perf_counter()
            response = call_project_agent(current_message, history)
            elapsed = time.perf_counter() - start

            response_text = str(response)
            passed = expected_term in normalize(response_text)

            responses.append(response_text)
            strict_correct.append(bool(passed))
            latencies.append(elapsed)
            errors.append('')

        except Exception as exc:
            elapsed = time.perf_counter() - start
            responses.append('')
            strict_correct.append(False)
            latencies.append(elapsed)
            errors.append(str(exc))

        current = index + 1
        if current == 1 or current % 5 == 0 or current == total:
            print(f'Context retention: {current}/{total}')

    df['agent_response'] = responses
    df['strict_correct'] = strict_correct
    df['latency_seconds'] = latencies
    df['error'] = errors

    valid_df = df[df['error'] == ''].copy()
    failed_df = df[df['error'] != ''].copy()

    strict_accuracy = float(valid_df['strict_correct'].mean()) if len(valid_df) else 0.0

    metrics = {
        'n_total_examples': int(len(df)),
        'n_valid_predictions': int(len(valid_df)),
        'n_failed_predictions': int(len(failed_df)),
        'strict_context_recall_accuracy': strict_accuracy,
        'strict_pass_count': int(valid_df['strict_correct'].sum()),
        'strict_fail_count': int((~valid_df['strict_correct']).sum()),
        'mean_latency_seconds': float(valid_df['latency_seconds'].mean()) if len(valid_df) else None,
        'median_latency_seconds': float(valid_df['latency_seconds'].median()) if len(valid_df) else None,
        'p95_latency_seconds': float(valid_df['latency_seconds'].quantile(0.95)) if len(valid_df) else None,
    }

    df.to_csv(RESULTS_DIR / 'context_retention_predictions.csv', index=False)
    breakdown(valid_df, 'difficulty').to_csv(
        RESULTS_DIR / 'context_retention_by_difficulty.csv', index=False
    )
    breakdown(valid_df, 'subcategory').to_csv(
        RESULTS_DIR / 'context_retention_by_subcategory.csv', index=False
    )

    strict_failures = valid_df[~valid_df['strict_correct']].copy()
    strict_failures.to_csv(
        RESULTS_DIR / 'context_retention_strict_failures.csv', index=False
    )

    # Keyword matching is intentionally conservative. A failure may be a
    # legitimate paraphrase, so this file is the small set that should receive
    # human review rather than being blindly treated as a semantic failure.
    review_df = strict_failures[
        ['id','history','current_message','expected_answer','expected_term',
         'agent_response','difficulty','subcategory']
    ].copy()
    review_df['human_review'] = ''
    review_df['review_notes'] = ''
    review_df.to_csv(
        RESULTS_DIR / 'context_retention_manual_review.csv', index=False
    )

    if not failed_df.empty:
        failed_df.to_csv(RESULTS_DIR / 'failed_predictions.csv', index=False)

    with open(RESULTS_DIR / 'context_retention_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('\nContext retention results')
    print('-' * 70)
    print(f"Examples:                 {len(df)}")
    print(f"Valid responses:          {len(valid_df)}")
    print(f"Failed API/code calls:    {len(failed_df)}")
    print(f"Strict recall accuracy:   {strict_accuracy:.4f}")
    print(f"Strict passes:            {metrics['strict_pass_count']}")
    print(f"Manual-review cases:      {metrics['strict_fail_count']}")
    if len(valid_df):
        print(f"Mean latency:             {metrics['mean_latency_seconds']:.4f} s")
        print(f"Median latency:           {metrics['median_latency_seconds']:.4f} s")
        print(f"P95 latency:              {metrics['p95_latency_seconds']:.4f} s")
    print(f"\nResults saved to:\n{RESULTS_DIR}")
    print('\nImportant: strict failures are not automatically semantic failures.')
    print('Review context_retention_manual_review.csv for valid paraphrases.')


if __name__ == '__main__':
    main()
