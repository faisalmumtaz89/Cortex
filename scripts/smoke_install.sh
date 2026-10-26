#!/usr/bin/env bash
# Smoke-test a built wheel the way a new user gets it, on the machine it was
# built for: install.sh installs it from a local stub release (laid out like a
# GitHub release), then the CLI, a headless agent turn and the TUI must work.
#
#   scripts/smoke_install.sh DIST_DIR        # DIST_DIR holds one wheel + its .sha256
#
# CORTEX_MANAGED_PYTHON=1 makes install.sh use its private Python instead of
# the system one.
set -euo pipefail

dist="$(cd "$1" && pwd)"
repo="$(cd "$(dirname "$0")/.." && pwd)"
wheel="$(ls "${dist}"/*.whl | head -n1)"
name="$(basename "${wheel}")"
version="$(echo "${name}" | cut -d- -f2)"
python="$(command -v python3)"

work="$(mktemp -d "${TMPDIR:-/tmp}/cortex-smoke.XXXXXX")"
release="${work}/srv/faisalmumtaz89/Cortex/releases/download/v${version}"
mkdir -p "${release}" "${work}/home" "${work}/proj"
cp "${wheel}" "${wheel}.sha256" "${release}/"

port="$("${python}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
"${python}" -m http.server "${port}" --bind 127.0.0.1 --directory "${work}/srv" >"${work}/http.log" 2>&1 &
server=$!
trap 'kill "${server}" 2>/dev/null || true; rm -rf "${work}"' EXIT
sleep 1

step() { echo "==> $*"; }

step "install.sh ${version} (${name})"
(
  cd "${work}"
  HOME="${work}/home" SHELL=/bin/bash \
    CORTEX_RELEASE_BASE="http://127.0.0.1:${port}/faisalmumtaz89/Cortex" \
    CORTEX_ALLOW_INSECURE_BASE=1 CORTEX_SYSTEM_SYMLINK=0 \
    bash "${repo}/install.sh" "${version}"
)
cortex="${work}/home/.local/bin/cortex"
grep -qsF ".local/bin" "${work}/home/.bashrc" || { echo "install.sh did not add ~/.local/bin to PATH"; exit 1; }

step "cortex --version"
HOME="${work}/home" "${cortex}" --version | tee "${work}/version.txt"
grep -q "${version}" "${work}/version.txt"

step "headless agent turn (scripted model: read a file, then answer)"
printf 'def add(a, b):\n    return a + b\n' >"${work}/proj/calc.py"
cat >"${work}/script.json" <<'EOF'
{"responses": [[{"tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}]},
                {"text": "SMOKE_OK: calc.py defines add()."}]]}
EOF
(
  cd "${work}/proj"
  HOME="${work}/home" OPENAI_API_KEY=sk-smoke CORTEX_SCRIPTED_MODEL="${work}/script.json" \
    "${cortex}" -p "what does calc.py do" --model openai:scripted
) >"${work}/turn.txt" 2>"${work}/turn.err"
cat "${work}/turn.txt" "${work}/turn.err"
grep -q "SMOKE_OK" "${work}/turn.txt"
grep -q "read_file ok" "${work}/turn.err"

step "TUI renders in a terminal"
HOME="${work}/home" "${python}" - "${cortex}" "${work}/proj" <<'PY'
import os, pty, select, signal, struct, sys, time, fcntl, termios

cortex, project = sys.argv[1], sys.argv[2]
pid, fd = pty.fork()
if pid == 0:
    os.chdir(project)
    os.environ["TERM"] = "xterm-256color"
    os.execv(cortex, [cortex])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
seen, deadline = b"", time.time() + 60
while time.time() < deadline and b"Ask anything" not in seen:
    if select.select([fd], [], [], 1)[0]:
        try:
            seen += os.read(fd, 65536)
        except OSError:
            break
os.kill(pid, signal.SIGTERM)
os.waitpid(pid, 0)
if b"Ask anything" not in seen:
    sys.exit("the TUI did not render its prompt; last output: %r" % seen[-400:])
print("TUI rendered its prompt")
PY

echo "smoke test passed for ${name}"
