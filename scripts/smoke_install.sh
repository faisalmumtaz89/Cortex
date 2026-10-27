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
wheels=("${dist}"/*.whl)
wheel="${wheels[0]}"
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
    CORTEX_ALLOW_INSECURE_BASE=1 \
    bash "${repo}/install.sh" "${version}"
)
cortex="${work}/home/.local/bin/cortex"
mode="$("${python}" -c 'import os, sys; print(oct(os.stat(sys.argv[1]).st_mode & 0o777))' "${work}/home/.cortex")"
[[ "${mode}" == "0o700" ]] || { echo "install.sh left ~/.cortex readable by others (${mode})"; exit 1; }
grep -qsF ".local/bin" "${work}/home/.bashrc" || { echo "install.sh did not add ~/.local/bin to PATH"; exit 1; }
# A new terminal finds cortex, whether it starts a login shell or not.
for flags in -lc -ic; do
  if ! env -i HOME="${work}/home" PATH=/usr/bin:/bin bash "${flags}" 'command -v cortex' >/dev/null 2>&1; then
    echo "a new bash ${flags} shell does not find cortex"
    exit 1
  fi
done
for system_dir in /usr/local/bin /opt/homebrew/bin; do
  if [[ "$(readlink "${system_dir}/cortex" 2>/dev/null)" == "${work}"* ]]; then
    echo "install.sh linked ${system_dir}/cortex: it must write only under HOME"
    exit 1
  fi
done
# Outside ~/.cortex, install.sh adds the cortex link and the PATH lines: no
# other command in ~/.local/bin (a private Python's python3.12 shadowed the
# system one) and no cache in ~/.cache.
extra="$(ls -A "${work}/home/.local/bin" | grep -vx cortex || true)"
[[ -z "${extra}" ]] || { echo "install.sh put more than cortex in ~/.local/bin:"; echo "${extra}"; exit 1; }
[[ ! -e "${work}/home/.cache" ]] || { echo "install.sh left a cache in ~/.cache:"; ls -A "${work}/home/.cache"; exit 1; }

base_prefix() { "$1" -c 'import os, sys; print(os.path.realpath(sys.base_prefix))'; }
if [[ "${CORTEX_MANAGED_PYTHON:-0}" == "1" ]]; then
  step "install.sh again, on the system Python: the runtime moves to it"
  (
    cd "${work}"
    HOME="${work}/home" SHELL=/bin/bash CORTEX_MANAGED_PYTHON=0 CORTEX_PYTHON="${python}" \
      CORTEX_RELEASE_BASE="http://127.0.0.1:${port}/faisalmumtaz89/Cortex" \
      CORTEX_ALLOW_INSECURE_BASE=1 \
      bash "${repo}/install.sh" "${version}"
  ) | tee "${work}/reinstall.txt"
  runtime="${work}/home/.cortex/install/venv/bin/python"
  [[ "$(base_prefix "${runtime}")" == "$(base_prefix "${python}")" ]] \
    || { echo "the reinstall kept the runtime on the private Python"; exit 1; }
  [[ ! -e "${work}/home/.cortex/install/python" ]] \
    || { echo "the reinstall left the private Python, which nothing runs on now"; exit 1; }
  grep -q "Reinstalled Cortex ${version}" "${work}/reinstall.txt"
fi

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

step "TUI renders in a terminal and quits on Ctrl+C"
mkdir -p "${work}/tmp"
HOME="${work}/home" TMPDIR="${work}/tmp" "${python}" - "${cortex}" "${work}/proj" <<'PY'
import os, pty, select, signal, struct, sys, time, fcntl, termios

cortex, project = sys.argv[1], sys.argv[2]
pid, fd = pty.fork()
if pid == 0:
    os.chdir(project)
    os.environ["TERM"] = "xterm-256color"
    os.execv(cortex, [cortex])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
seen, status = b"", None


def read_for(seconds, done):
    """Read the TUI's output until done() or the time is up. Reading never
    stops while the TUI runs: a full pty would block its writes for good."""
    global seen
    deadline = time.time() + seconds
    while time.time() < deadline and not done():
        if select.select([fd], [], [], 0.2)[0]:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # Linux: EIO once nothing holds the pty open
                chunk = b""
            if not chunk:
                time.sleep(0.1)
            seen += chunk


def exited():
    global status
    if status is None:
        reaped, code = os.waitpid(pid, os.WNOHANG)
        if reaped:
            status = code
    return status is not None


def group_alive():
    try:
        os.killpg(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):  # macOS: EPERM once only zombies are left
        return False


def fail(message):
    if group_alive():
        os.killpg(pid, signal.SIGKILL)
    if not exited():
        os.waitpid(pid, 0)
    sys.exit("%s; last output: %r" % (message, seen[-600:]))


read_for(60, lambda: b"Ask anything" in seen or exited())
if b"Ask anything" not in seen:
    fail("the TUI did not render its prompt")
print("TUI rendered its prompt")

os.write(fd, b"\x03")  # Ctrl+C, the way a user quits
read_for(15, exited)
if not exited():
    fail("the TUI was still running 15s after Ctrl+C")
read_for(10, lambda: not group_alive())
if group_alive():
    fail("Ctrl+C left Cortex processes running")
print("TUI quit on Ctrl+C and left no processes behind")
PY
# The sidecar unpacks a native library into the temp folder; Cortex removes it.
leftovers="$(find "${work}/tmp" -type f | head -n 5)"
if [[ -n "${leftovers}" ]]; then
  echo "the TUI left files in the temp folder:"
  echo "${leftovers}"
  exit 1
fi

echo "smoke test passed for ${name}"
