from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EVALUATION_DIR / "results"
HF_DIR = RESULTS_DIR / "hf"
OPENAI_DIR = RESULTS_DIR / "openai"

hf_path = HF_DIR / "hf_summary.csv"
openai_path = OPENAI_DIR / "openai_summary.csv"

if not hf_path.exists():
    raise FileNotFoundError(f"Missing Hugging Face summary: {hf_path}")
if not openai_path.exists():
    raise FileNotFoundError(f"Missing OpenAI summary: {openai_path}")

hf = pd.read_csv(hf_path)
openai = pd.read_csv(openai_path)

quality = hf[["task", "n_examples", "accuracy", "macro_f1"]].merge(
    openai[["task", "n_examples", "accuracy", "macro_f1", "out_of_schema_rate", "api_errors"]],
    on="task",
    suffixes=("_hf", "_openai"),
)

quality = quality.rename(columns={
    "n_examples_hf": "hf_n",
    "accuracy_hf": "hf_accuracy",
    "macro_f1_hf": "hf_macro_f1",
    "n_examples_openai": "openai_n",
    "accuracy_openai": "openai_accuracy",
    "macro_f1_openai": "openai_macro_f1",
})

quality["accuracy_delta_openai_minus_hf"] = (
    quality["openai_accuracy"] - quality["hf_accuracy"]
)
quality["macro_f1_delta_openai_minus_hf"] = (
    quality["openai_macro_f1"] - quality["hf_macro_f1"]
)

out_path = RESULTS_DIR / "hf_vs_openai_quality.csv"
quality.to_csv(out_path, index=False)

print("\nHF vs OpenAI classification quality")
print("=" * 80)
print(quality.to_string(index=False))
print(f"\nSaved to: {out_path}")
print("\nNote: latency is intentionally not compared here. Use the dedicated")
print("pipeline-latency script because the OpenAI analyzer predicts all three")
print("signals in one API call, while the HF application uses three models.")
