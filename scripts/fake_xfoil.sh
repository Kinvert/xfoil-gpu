#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r line; do
  if [[ "$line" == "PACC" ]]; then
    read -r polar_file
    if [[ -n "${polar_file}" ]]; then
      printf "  5.0  0.500  0.0100  0.0002  -0.0200\\n" > "${polar_file}"
    fi
  fi
done
