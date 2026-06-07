#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/configs/cpu_baseline_fallback.toml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUT="${REPO_ROOT}/logs/cpu_baseline_${TIMESTAMP}.json"

if [[ ! -d "${REPO_ROOT}/logs" ]]; then
  mkdir -p "${REPO_ROOT}/logs"
fi

run_python="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${run_python}" ]]; then
  run_python="/usr/bin/env python3"
fi

HAS_CONFIG=0
HAS_OUT=0
SELECTED_CONFIG="${DEFAULT_CONFIG}"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      HAS_CONFIG=1
      SELECTED_CONFIG="${2:-}"
      if [[ -z "${SELECTED_CONFIG}" ]]; then
        echo "error: --config requires a value" >&2
        exit 2
      fi
      ARGS+=("--config" "${SELECTED_CONFIG}")
      shift 2
      ;;
    --config=*)
      HAS_CONFIG=1
      SELECTED_CONFIG="${1#--config=}"
      ARGS+=("${1}")
      shift
      ;;
    --out)
      HAS_OUT=1
      ARGS+=("--out" "${2:-}")
      if [[ -z "${2:-}" ]]; then
        echo "error: --out requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --out=*)
      HAS_OUT=1
      ARGS+=("${1}")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
 done

if [[ "${HAS_CONFIG}" == "0" ]]; then
  ARGS+=(--config "${DEFAULT_CONFIG}")
fi
if [[ "${HAS_OUT}" == "0" ]]; then
  ARGS+=(--out "${DEFAULT_OUT}")
fi

echo "Running CPU baseline benchmark..." >&2
echo "  config: ${SELECTED_CONFIG}" >&2
if [[ "${HAS_OUT}" == "0" ]]; then
  echo "  output: ${DEFAULT_OUT}" >&2
fi

if ! "${run_python}" "${REPO_ROOT}/scripts/bench_cpu.py" "${ARGS[@]}"; then
  echo "benchmark failed" >&2
  exit 1
fi
