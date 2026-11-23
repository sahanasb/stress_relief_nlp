from pathlib import Path
import json
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent.parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_history(raw, row_id="unknown"):
    if raw is None or (isinstance(raw, float) and str(raw) == "nan") or str(raw).strip() == "":
        return []
    try:
        history = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid history JSON for {row_id}: {exc}") from exc
    if not isinstance(history, list):
        raise ValueError(f"History for {row_id} must be a JSON list")
    for i, message in enumerate(history):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError(f"History item {i} for {row_id} must contain role/content")
    return history
