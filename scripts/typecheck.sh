#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x "./venv/bin/python" ]]; then
  PY="./venv/bin/python"
elif [[ -x "./.venv/bin/python" ]]; then
  PY="./.venv/bin/python"
else
  PY="python"
fi

"${PY}" -m mypy cortex --show-error-codes --no-color-output
