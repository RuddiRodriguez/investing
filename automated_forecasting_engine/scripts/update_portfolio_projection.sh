#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ENGINE_ROOT}/.." && pwd)"

VENV="${VENV:-${PROJECT_ROOT}/venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
PORTFOLIO_PDF="${PORTFOLIO_PDF:-/Users/ruddigarcia/Downloads/Patrimonio neto 3.pdf}"
OUTPUT_DIR="${OUTPUT_DIR:-${ENGINE_ROOT}/runs/portfolio_projection_latest}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-}"
HORIZONS="${HORIZONS:-1,5,30}"
PROJECTION_HORIZON="${PROJECTION_HORIZON:-5}"
SEARCH_LEVEL="${SEARCH_LEVEL:-fast}"
REFRESH_DATA_CACHE="${REFRESH_DATA_CACHE:-1}"
ENABLE_LIGHTGBM="${ENABLE_LIGHTGBM:-0}"
ENABLE_STAT_MODELS="${ENABLE_STAT_MODELS:-0}"
INCLUDE_LSTM="${INCLUDE_LSTM:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mpl}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  echo "Set VENV=/path/to/venv or PYTHON=/path/to/python and retry." >&2
  exit 1
fi

if [[ ! -f "${PORTFOLIO_PDF}" ]]; then
  echo "Portfolio PDF not found: ${PORTFOLIO_PDF}" >&2
  echo "Set PORTFOLIO_PDF=/path/to/statement.pdf and retry." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

extra_args=()
if [[ -n "${END_DATE}" ]]; then
  extra_args+=(--end "${END_DATE}")
fi
if [[ "${REFRESH_DATA_CACHE}" == "1" ]]; then
  extra_args+=(--refresh-data-cache)
fi
if [[ "${ENABLE_LIGHTGBM}" != "1" ]]; then
  extra_args+=(--no-lightgbm)
fi
if [[ "${ENABLE_STAT_MODELS}" != "1" ]]; then
  extra_args+=(--no-statistical-models)
fi
if [[ "${INCLUDE_LSTM}" == "1" ]]; then
  extra_args+=(--include-lstm)
fi

cd "${PROJECT_ROOT}"

echo "Updating portfolio projection"
echo "PDF: ${PORTFOLIO_PDF}"
echo "Output: ${OUTPUT_DIR}"
echo "Projection horizon: ${PROJECTION_HORIZON} trading days"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${ENGINE_ROOT}/src" \
MPLCONFIGDIR="${MPLCONFIGDIR}" \
"${PYTHON}" -m market_forecasting_engine.portfolio_cli \
  --pdf "${PORTFOLIO_PDF}" \
  --start "${START_DATE}" \
  --horizons "${HORIZONS}" \
  --projection-horizon "${PROJECTION_HORIZON}" \
  --search-level "${SEARCH_LEVEL}" \
  --output-dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"

echo
echo "Updated files:"
echo "- ${OUTPUT_DIR}/portfolio_projection.csv"
echo "- ${OUTPUT_DIR}/portfolio_projection.html"
echo "- ${OUTPUT_DIR}/portfolio_projection_plotly.html"
echo "- ${OUTPUT_DIR}/portfolio_projection_summary.json"
