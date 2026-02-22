#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-latest}"
if [[ -n "${BASH_SOURCE[0]-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
else
  # When executed via `curl ... | bash`, BASH_SOURCE can be unset under `set -u`.
  SCRIPT_DIR="$(pwd -P)"
fi

if [[ -n "${1:-}" ]] && [[ ! "$1" =~ ^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$ ]]; then
  echo "Usage: $0 [stable|latest|VERSION]" >&2
  exit 1
fi

if [[ -t 1 ]]; then
  RED=$'\033[31m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  CYAN=$'\033[36m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  RED=""
  GREEN=""
  YELLOW=""
  CYAN=""
  BOLD=""
  RESET=""
fi

log() {
  echo "${CYAN}→${RESET} $*"
}

ok() {
  echo "${GREEN}✓${RESET} $*"
}

warn() {
  echo "${YELLOW}⚠${RESET} $*" >&2
}

die() {
  echo "${RED}✗${RESET} $*" >&2
  exit 1
}

pick_python() {
  local candidates=()
  if [[ -n "${CORTEX_PYTHON:-}" ]]; then
    candidates+=("${CORTEX_PYTHON}")
  fi
  candidates+=("python3.13" "python3.12" "python3.11" "python3")

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

is_source_checkout() {
  if [[ ! -f "${SCRIPT_DIR}/pyproject.toml" ]] || [[ ! -d "${SCRIPT_DIR}/cortex" ]]; then
    return 1
  fi
  grep -q '^name = "cortex-llm"' "${SCRIPT_DIR}/pyproject.toml"
}

pick_downloader() {
  if command -v curl >/dev/null 2>&1; then
    echo "curl"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    echo "wget"
    return 0
  fi
  return 1
}

purge_stale_site_package_cortex() {
  local python_bin="$1"
  local source_root="$2"

  "${python_bin}" - "$source_root" <<'PY'
import shutil
import site
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
removed = []

site_dirs = []
try:
    site_dirs.extend(Path(p).resolve() for p in site.getsitepackages())
except Exception:
    pass
user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site:
    site_dirs.append(Path(user_site).resolve())

for base in site_dirs:
    candidate = base / "cortex"
    if not candidate.exists() or not candidate.is_dir():
        continue
    # Keep source tree packages untouched; only purge stale copies in site-packages.
    if source_root in candidate.parents:
        continue
    shutil.rmtree(candidate)
    removed.append(str(candidate))

for path in removed:
    print(path)
PY
}

clear_hidden_site_flags() {
  local python_bin="$1"
  command -v chflags >/dev/null 2>&1 || return 0

  local site_dirs=""
  site_dirs="$("${python_bin}" - <<'PY'
import site
from pathlib import Path

dirs = []
try:
    dirs.extend(site.getsitepackages())
except Exception:
    pass
user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site:
    dirs.append(user_site)

for value in dirs:
    path = Path(value).expanduser().resolve()
    print(path)
PY
)"

  while IFS= read -r dir; do
    [[ -z "${dir}" ]] && continue
    [[ -d "${dir}" ]] || continue
    chflags nohidden "${dir}" >/dev/null 2>&1 || true
    find "${dir}" -maxdepth 1 -type f -name "*.pth" -exec chflags nohidden {} + >/dev/null 2>&1 || true
  done <<< "${site_dirs}"
}

ensure_source_package_link() {
  local python_bin="$1"
  local source_root="$2"

  "${python_bin}" - "$source_root" <<'PY'
import shutil
import site
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
source_package = source_root / "cortex"
if not source_package.is_dir():
    raise SystemExit("missing source package directory")

site_dirs = []
try:
    site_dirs.extend(Path(p).resolve() for p in site.getsitepackages())
except Exception:
    pass
user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site:
    site_dirs.append(Path(user_site).resolve())

created_link = None
for base in site_dirs:
    if not base.exists():
        continue
    candidate = base / "cortex"
    if candidate.exists() or candidate.is_symlink():
        if candidate.is_symlink() and candidate.resolve() == source_package:
            created_link = candidate
            break
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
    candidate.symlink_to(source_package, target_is_directory=True)
    created_link = candidate
    break

if created_link is None:
    raise SystemExit("unable to create site-packages source link for cortex")

print(created_link)
PY
}

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return 0
  fi

  local bun_local="${HOME}/.bun/bin/bun"
  if [[ -x "${bun_local}" ]]; then
    echo "${bun_local}"
    return 0
  fi

  local downloader
  downloader="$(pick_downloader || true)"
  if [[ -z "${downloader}" ]]; then
    die "Bun is required to build the OpenTUI sidecar, but neither curl nor wget is available."
  fi

  log "Bun not found. Installing Bun..." >&2
  if [[ "${downloader}" == "curl" ]]; then
    curl -fsSL https://bun.sh/install | bash -s -- >/dev/null
  else
    wget -q -O - https://bun.sh/install | bash -s -- >/dev/null
  fi

  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return 0
  fi
  if [[ -x "${bun_local}" ]]; then
    echo "${bun_local}"
    return 0
  fi

  die "Bun installation failed. Install manually from https://bun.sh and rerun installer."
}

ensure_source_sidecar() {
  local source_root="$1"
  local sidecar="${source_root}/cortex/ui_runtime/bin/cortex-tui"
  local frontend_dir="${source_root}/frontend/cortex-tui"
  if [[ ! -d "${frontend_dir}" ]]; then
    die "OpenTUI frontend sources not found at ${frontend_dir}."
  fi

  local needs_build="0"
  local reason="missing"
  if [[ ! -x "${sidecar}" ]]; then
    needs_build="1"
    reason="missing"
  else
    # Rebuild when frontend sources/config are newer than the bundled sidecar.
    local stale_marker=""
    stale_marker="$(find "${frontend_dir}" -type f \
      \( -name "*.ts" -o -name "*.tsx" -o -name "package.json" -o -name "bun.lock*" -o -name "tsconfig*.json" \) \
      -newer "${sidecar}" -print -quit 2>/dev/null || true)"
    if [[ -n "${stale_marker}" ]]; then
      needs_build="1"
      reason="stale"
    fi
  fi

  if [[ "${needs_build}" != "1" ]]; then
    return 0
  fi

  local bun_bin
  bun_bin="$(ensure_bun)"

  if [[ "${reason}" == "stale" ]]; then
    log "Rebuilding OpenTUI sidecar (frontend sources changed)..."
  else
    log "Building OpenTUI sidecar..."
  fi
  (
    cd "${frontend_dir}"
    "${bun_bin}" install >/dev/null
    "${bun_bin}" run build >/dev/null
  )

  if [[ ! -x "${sidecar}" ]]; then
    die "OpenTUI sidecar build failed: ${sidecar} was not produced."
  fi

  ok "OpenTUI sidecar built."
}

verify_sidecar_runtime() {
  local sidecar_path
  local sidecar_ok
  sidecar_path="$("${RUNTIME_PYTHON}" - <<'PY'
from cortex.ui_runtime.launcher import _bundled_binary
print(_bundled_binary())
PY
)"
  sidecar_ok="$("${RUNTIME_PYTHON}" - <<'PY'
import os
from cortex.ui_runtime.launcher import _bundled_binary
path = _bundled_binary()
print("1" if path.exists() and os.access(path, os.X_OK) else "0")
PY
)"

  if [[ "${sidecar_ok}" != "1" ]]; then
    die "OpenTUI sidecar is missing or not executable at ${sidecar_path}. Installation incomplete."
  fi
  ok "OpenTUI sidecar ready: ${sidecar_path}"
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  die "Cortex is currently supported on macOS only."
fi

shell_arch="$(uname -m)"
if [[ "${shell_arch}" != "arm64" ]]; then
  if [[ "${shell_arch}" == "x86_64" ]] && [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    warn "Terminal is running under Rosetta. Native arm64 terminal is recommended."
  else
    die "Unsupported architecture: ${shell_arch}. Cortex requires Apple Silicon."
  fi
fi

if ! command -v xcode-select >/dev/null 2>&1 || ! xcode-select -p >/dev/null 2>&1; then
  die "Xcode Command Line Tools are required. Run: xcode-select --install"
fi

PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  die "Python 3.11+ was not found. Install it with: brew install python@3.11"
fi

PY_ARCH="$("${PYTHON_BIN}" -c 'import platform; print(platform.machine())')"
if [[ "${PY_ARCH}" != "arm64" ]]; then
  die "Detected ${PYTHON_BIN} (${PY_ARCH}). Install/use an arm64 Python 3.11+."
fi

INSTALL_ROOT="${CORTEX_INSTALL_ROOT:-$HOME/.cortex/install}"
VENV_DIR="${INSTALL_ROOT}/venv"
BIN_DIR="${CORTEX_BIN_DIR:-$HOME/.local/bin}"
LOCAL_LINK="${BIN_DIR}/cortex"

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Creating isolated runtime at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PIP="${VENV_DIR}/bin/pip"
RUNTIME_PYTHON="${VENV_DIR}/bin/python"

log "Upgrading installer tooling"
"${PIP}" install --upgrade pip setuptools wheel >/dev/null
clear_hidden_site_flags "${RUNTIME_PYTHON}"

INSTALL_MODE="pypi"
if [[ "${TARGET}" =~ ^(stable|latest)$ ]] && is_source_checkout && [[ "${CORTEX_INSTALL_SOURCE:-1}" == "1" ]]; then
  INSTALL_MODE="source"
fi

if [[ "${INSTALL_MODE}" == "source" ]]; then
  log "Installing Cortex from source checkout: ${SCRIPT_DIR}"
  stale_removed="$(purge_stale_site_package_cortex "${RUNTIME_PYTHON}" "${SCRIPT_DIR}" || true)"
  if [[ -n "${stale_removed}" ]]; then
    warn "Removed stale site-packages Cortex module(s):"
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "  - ${line}" >&2
    done <<< "${stale_removed}"
  fi
  "${PIP}" install --upgrade -e "${SCRIPT_DIR}"
  clear_hidden_site_flags "${RUNTIME_PYTHON}"
  SOURCE_LINK_PATH="$(ensure_source_package_link "${RUNTIME_PYTHON}" "${SCRIPT_DIR}")" \
    || die "Unable to create stable source link in site-packages."
  ok "Source package link: ${SOURCE_LINK_PATH}"

  SOURCE_IMPORT_PATH="$("${RUNTIME_PYTHON}" - "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
source_root = Path(sys.argv[1]).resolve()
import cortex
module_path = Path(cortex.__file__).resolve()
print(module_path)
if source_root not in module_path.parents:
    raise SystemExit(1)
PY
)" || die "Editable install did not resolve cortex imports to source checkout. Re-run installer."

  ok "Python import path: ${SOURCE_IMPORT_PATH}"
  ensure_source_sidecar "${SCRIPT_DIR}"
else
  PACKAGE_SPEC="cortex-llm"
  if [[ "${TARGET}" != "stable" ]] && [[ "${TARGET}" != "latest" ]]; then
    PACKAGE_SPEC="cortex-llm==${TARGET}"
  fi
  log "Installing ${PACKAGE_SPEC}"
  "${PIP}" install --upgrade "${PACKAGE_SPEC}"
  clear_hidden_site_flags "${RUNTIME_PYTHON}"
fi

if [[ ! -x "${VENV_DIR}/bin/cortex" ]]; then
  die "Installation completed but cortex entrypoint was not created."
fi

verify_sidecar_runtime

ln -sfn "${VENV_DIR}/bin/cortex" "${LOCAL_LINK}"
chmod +x "${VENV_DIR}/bin/cortex"

ACTIVE_BIN="${LOCAL_LINK}"
for system_dir in "/opt/homebrew/bin" "/usr/local/bin"; do
  if [[ -d "${system_dir}" ]] && [[ -w "${system_dir}" ]]; then
    ln -sfn "${LOCAL_LINK}" "${system_dir}/cortex"
    ACTIVE_BIN="${system_dir}/cortex"
    break
  fi
done

INSTALLED_VERSION="$("${RUNTIME_PYTHON}" - <<'PY'
from importlib.metadata import version
print(version("cortex-llm"))
PY
)"

ok "Installed Cortex ${INSTALLED_VERSION}"
ok "Launcher: ${ACTIVE_BIN}"

if ! command -v cortex >/dev/null 2>&1; then
  if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    echo
    warn "cortex is not currently on PATH."
    echo "Add this to your shell profile:"
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
    echo
  fi
fi

echo
echo "${BOLD}Installation complete.${RESET}"
echo "Run: cortex"
echo
