# Install Cortex on Windows. Cortex runs inside WSL (Windows Subsystem for
# Linux); this script installs it there and adds a `cortex` command to Windows
# that runs it in the current folder.
#
#   irm https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.ps1 | iex
#
# Optional: $env:CORTEX_VERSION = "X.Y.Z" installs that release instead of the latest.

& {
    $ErrorActionPreference = 'Stop'
    $installer = 'https://raw.githubusercontent.com/faisalmumtaz89/Cortex/main/install.sh'

    function Say([string]$text) { Write-Host "-> $text" -ForegroundColor Cyan }
    function Stop-With([string]$text) { Write-Host "x $text" -ForegroundColor Red }

    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Stop-With 'Cortex runs in WSL, which is not installed.'
        Write-Host '  1. Open PowerShell as Administrator and run:  wsl --install'
        Write-Host '  2. Restart Windows, open "Ubuntu" from the Start menu once to create your Linux user.'
        Write-Host '  3. Run this installer again.'
        return
    }

    # A Linux distribution must exist and start; `wsl --install` alone may still need a restart.
    & wsl.exe --exec sh -c 'exit 0' 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stop-With 'WSL has no working Linux distribution yet.'
        Write-Host '  Run:  wsl --install -d Ubuntu'
        Write-Host '  then open "Ubuntu" from the Start menu once to create your Linux user, and run this installer again.'
        return
    }

    $version = if ($env:CORTEX_VERSION) { $env:CORTEX_VERSION } else { 'latest' }
    if ($version -notmatch '^(latest|stable|v?\d+\.\d+\.\d+)$') {
        Stop-With "CORTEX_VERSION must be X.Y.Z (got: $version)."
        return
    }

    Say 'Installing Cortex inside WSL'
    & wsl.exe --exec bash -c "curl -fsSL '$installer' | bash -s -- $version"
    if ($LASTEXITCODE -ne 0) {
        Stop-With 'Installing Cortex inside WSL failed; see the messages above.'
        return
    }

    # The Linux launcher by absolute path: a non-login, non-interactive shell does not read ~/.bashrc.
    $linuxHome = (& wsl.exe --exec sh -c 'printf %s "$HOME"').Trim()
    $launcher = "$linuxHome/.local/bin/cortex"
    & wsl.exe --exec test -x $launcher
    if ($LASTEXITCODE -ne 0) {
        Stop-With "Cortex was installed, but $launcher was not found inside WSL."
        return
    }

    $dir = Join-Path $env:LOCALAPPDATA 'Programs\Cortex'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $cmd = Join-Path $dir 'cortex.cmd'
    # --cd maps the current Windows folder (C:\...) to its WSL path (/mnt/c/...).
    Set-Content -Path $cmd -Encoding Ascii -Value "@echo off`r`nwsl.exe --cd `"%CD%`" --exec $launcher %*"

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($userPath -split ';') -contains $dir)) {
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $dir } else { "$userPath;$dir" }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Say "Added $dir to your PATH"
    }
    if (-not (($env:Path -split ';') -contains $dir)) { $env:Path = "$env:Path;$dir" }

    Write-Host ''
    Write-Host 'Cortex is installed. Run: cortex' -ForegroundColor Green
    Write-Host 'It runs inside WSL; keep projects in your Linux home (\\wsl$) for the best speed.'
}
