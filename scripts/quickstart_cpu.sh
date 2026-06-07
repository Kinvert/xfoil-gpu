#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it and rerun, or create .venv manually." >&2
    exit 1
  fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv venv .venv --python 3.11
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv pip install -e .
fi

alpha="${1:-5}"
reynolds="${2:-1000000}"

"${REPO_ROOT}/.venv/bin/python" scripts/env_probe.py
PYTHONPATH="${REPO_ROOT}/src" "${REPO_ROOT}/.venv/bin/python" -m xfoil_port.cli \
  "${REPO_ROOT}/data/naca0012.dat" --alpha "${alpha}" --re "${reynolds}" --use-mock
