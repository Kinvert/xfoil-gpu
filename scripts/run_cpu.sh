#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "Expected ${REPO_ROOT}/.venv/bin/python. Create it first with: uv venv .venv" >&2
  exit 1
fi
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ $# -eq 0 ]]; then
  echo "usage: run_cpu.sh <geometry> [xfoil flags]" >&2
  echo "example: run_cpu.sh data/naca0012.dat --alpha 5.0 --re 1000000" >&2
  exit 2
fi

declare -a args=("$@")

if [[ -z "${1:-}" || "${1:0:1}" == "-" ]]; then
  echo "run_cpu.sh expects a geometry path as the first positional argument." >&2
  exit 2
fi

geometry_candidate="${args[0]}"
if [[ "${geometry_candidate:0:1}" != "/" && ! -f "${geometry_candidate}" ]]; then
  if [[ -f "${REPO_ROOT}/${geometry_candidate}" ]]; then
    geometry_candidate="${REPO_ROOT}/${geometry_candidate}"
    args[0]="${geometry_candidate}"
  fi
fi
if [[ "${geometry_candidate:0:1}" != "/" && "${geometry_candidate:0:1}" != "." ]]; then
  if [[ -f "${REPO_ROOT}/${geometry_candidate}" ]]; then
    geometry_candidate="${REPO_ROOT}/${geometry_candidate}"
    args[0]="${geometry_candidate}"
  fi
fi
if [[ ! -f "${args[0]}" ]]; then
  echo "run_cpu.sh error: geometry file not found: ${args[0]}" >&2
  exit 2
fi

declare -a forwarded_args=()
require_oracle=0
use_mock=0
xfoil_candidate="${XFOIL_EXE:-xfoil}"
xfoil_path=""
explicit_xfoil=0
config_path=""
backend_mode="cpu"
explicit_backend=0

for ((i = 0; i < ${#args[@]}; i++)); do
  arg="${args[i]}"
  case "${arg}" in
    --require-oracle)
      require_oracle=1
      forwarded_args+=("${arg}")
      ;;
    --use-mock)
      use_mock=1
      forwarded_args+=("${arg}")
      ;;
    --xfoil=*)
      xfoil_candidate="${arg#*=}"
      explicit_xfoil=1
      forwarded_args+=("${arg}")
      ;;
    --xfoil)
      forwarded_args+=("${arg}")
      if [[ $((i + 1)) -lt ${#args[@]} ]]; then
        i=$((i + 1))
        xfoil_candidate="${args[i]}"
        forwarded_args+=("${xfoil_candidate}")
        explicit_xfoil=1
      fi
      ;;
    --config=*)
      config_path="${arg#*=}"
      forwarded_args+=("${arg}")
      ;;
    --config)
      forwarded_args+=("${arg}")
      if [[ $((i + 1)) -lt ${#args[@]} ]]; then
        i=$((i + 1))
        config_path="${args[i]}"
        forwarded_args+=("${config_path}")
      fi
      ;;
    --backend=*)
      backend_mode="${arg#*=}"
      explicit_backend=1
      forwarded_args+=("${arg}")
      ;;
    --backend)
      forwarded_args+=("${arg}")
      if [[ $((i + 1)) -lt ${#args[@]} ]]; then
        i=$((i + 1))
        backend_mode="${args[i]}"
        forwarded_args+=("${backend_mode}")
      fi
      explicit_backend=1
      ;;
    *)
      forwarded_args+=("${arg}")
      ;;
  esac
done

is_native_backend=0
case "${backend_mode}" in
  native|native-cpu|cpu-native)
    is_native_backend=1
    ;;
esac

if [[ -n "${config_path}" ]]; then
  if [[ "${config_path:0:1}" != "/" && ! -f "${config_path}" ]]; then
    if [[ -f "${REPO_ROOT}/${config_path}" ]]; then
      config_path="${REPO_ROOT}/${config_path}"
    fi
  fi
fi

  if [[ "${is_native_backend}" == "0" && "${explicit_backend}" == "0" && -f "${config_path}" ]]; then
  detected_backend="$(${PYTHON_BIN} - "${config_path}" <<'PY'
import sys
import tomllib

path = sys.argv[1]
with open(path, 'rb') as handle:
    data = tomllib.load(handle)

backend = (data.get('backend', {}) or {})
candidate = backend.get('backend') or backend.get('backend_name')
if isinstance(candidate, str):
    print(candidate.strip())
PY
)"
  detected_backend="$(echo "${detected_backend}" | tr '[:upper:]' '[:lower:]')"
  detected_backend="${detected_backend//_/-}"
  case "${detected_backend}" in
    native|native-cpu|cpu-native)
      backend_mode="${detected_backend}"
      ;;
  esac
fi
case "${backend_mode}" in
  native|native-cpu|cpu-native)
    is_native_backend=1
    ;;
  *)
    is_native_backend=0
    ;;
esac

if [[ "${is_native_backend}" == "1" && ("${require_oracle}" == "1" || "${use_mock}" == "1") ]]; then
  echo "run_cpu.sh error: --backend native is incompatible with --require-oracle/--use-mock." >&2
  exit 2
fi

if [[ "${xfoil_candidate}" == */* ]] && [[ "${xfoil_candidate:0:1}" != "/" ]]; then
  if [[ -x "${REPO_ROOT}/${xfoil_candidate}" ]]; then
    xfoil_candidate="${REPO_ROOT}/${xfoil_candidate}"
  fi
fi
if [[ "${explicit_xfoil}" == "0" && "${xfoil_candidate}" != "xfoil" && "${is_native_backend}" == "0" ]]; then
  forwarded_args+=(--xfoil "${xfoil_candidate}")
fi

declare -a normalized_args=()
for ((i = 0; i < ${#forwarded_args[@]}; i++)); do
  arg="${forwarded_args[i]}"
  if [[ "${arg}" == --xfoil=* ]]; then
    normalized_args+=("--xfoil=${xfoil_candidate}")
    continue
  fi
  if [[ "${arg}" == --xfoil ]]; then
    normalized_args+=("--xfoil")
    if [[ $((i + 1)) -lt ${#forwarded_args[@]} ]]; then
      i=$((i + 1))
      normalized_args+=("${xfoil_candidate}")
    fi
    continue
  fi
  normalized_args+=("${arg}")
done
forwarded_args=("${normalized_args[@]}")

if [[ "${require_oracle}" == "1" && "${use_mock}" == "1" ]]; then
  echo "run_cpu.sh error: --require-oracle and --use-mock are incompatible." >&2
  exit 2
fi

if [[ "${is_native_backend}" == "1" ]]; then
  has_xfoil=0
else
  has_xfoil=0
  if [[ -x "${xfoil_candidate}" ]]; then
    has_xfoil=1
    xfoil_path="${xfoil_candidate}"
  fi
  if [[ "${has_xfoil}" == "0" && "${xfoil_candidate}" != "xfoil" ]]; then
    if command -v "${xfoil_candidate}" >/dev/null 2>&1; then
      has_xfoil=1
      xfoil_path="$(command -v "${xfoil_candidate}")"
    fi
  fi
  if [[ "${has_xfoil}" == "0" && "${xfoil_candidate}" == "xfoil" ]]; then
    if command -v xfoil >/dev/null 2>&1; then
      has_xfoil=1
      xfoil_path="$(command -v xfoil)"
    fi
  fi
fi

mode="fallback"
if [[ "${is_native_backend}" == "1" ]]; then
  mode="native"
elif [[ "${require_oracle}" == "1" ]]; then
  mode="oracle"
elif [[ "${use_mock}" == "1" ]]; then
  mode="fallback"
elif [[ "${has_xfoil}" == "1" ]]; then
  mode="oracle"
fi

if [[ "${is_native_backend}" == "0" && "${require_oracle}" == "0" && "${use_mock}" == "0" && "${has_xfoil}" == "0" ]]; then
  forwarded_args+=(--use-mock)
  mode="fallback"
fi

${PYTHON_BIN} - "$mode" "$xfoil_candidate" "$xfoil_path" "$require_oracle" "$use_mock" "$has_xfoil" <<'PY'
import json
import sys

mode, requested, resolved, require_oracle, use_mock, available = sys.argv[1:]
print(
    json.dumps(
        {
            "xfoil_launch": {
                "mode": mode,
                "xfoil": {
                    "requested": requested,
                    "resolved": resolved,
                    "available": available == "1",
                },
                "mode_forced": {
                    "require_oracle": require_oracle == "1",
                    "use_mock": use_mock == "1",
                },
            }
        },
        sort_keys=True,
    ),
    file=sys.stderr,
)
PY

PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -m xfoil_port.cli "${forwarded_args[@]}"
exit $?
