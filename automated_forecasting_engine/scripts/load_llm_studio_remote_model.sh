#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
LMS="${LMS:-/Users/ruddigarcia/.lmstudio/bin/lms}"
UNLOAD_EXISTING="${UNLOAD_EXISTING:-1}"

MODEL_ID="$(
  PYTHONPATH="$PROJECT_DIR/automated_forecasting_engine/src" "$PYTHON" - <<'PY'
from market_forecasting_engine.llm_model_catalog import DEFAULT_LOCAL_TRADER_MODEL

print(DEFAULT_LOCAL_TRADER_MODEL)
PY
)"
REMOTE_MODEL_PATH="$(
  PYTHONPATH="$PROJECT_DIR/automated_forecasting_engine/src" "$PYTHON" - <<'PY'
from market_forecasting_engine.llm_model_catalog import DEFAULT_LLM_STUDIO_REMOTE_MODEL_PATH

print(DEFAULT_LLM_STUDIO_REMOTE_MODEL_PATH)
PY
)"

if [[ "$UNLOAD_EXISTING" == "1" ]]; then
  "$LMS" unload -a >/dev/null 2>&1 || true
fi

"$LMS" load "$MODEL_ID" --identifier "$MODEL_ID" -y
"$LMS" ps
