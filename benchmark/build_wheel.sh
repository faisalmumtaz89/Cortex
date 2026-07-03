#!/usr/bin/env bash
# Build a lean Cortex wheel for benchmark containers (no macOS sidecar binary).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY=python3

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'node_modules' \
  --exclude 'cortex/ui_runtime/bin' --exclude 'frontend' \
  --exclude 'tests' --exclude 'docs' --exclude 'jobs' --exclude 'benchmark' \
  "$ROOT/" "$STAGE/src/"

rm -rf "$ROOT/benchmark/dist"
"$PY" -m pip wheel --quiet --no-deps -w "$ROOT/benchmark/dist" "$STAGE/src"
ls -lh "$ROOT/benchmark/dist"
