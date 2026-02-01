#!/bin/bash

# Cortex Installer - Clean setup for macOS Apple Silicon
# Usage: ./install.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Banner
echo -e "${CYAN}"
echo "╭─────────────────────────────────────────────────────╮"
echo "│               CORTEX INSTALLER                      │"
echo "│    GPU-Accelerated LLM for Apple Silicon           │"
echo "╰─────────────────────────────────────────────────────╯"
echo -e "${RESET}\n"

# Check system
echo -e "${BOLD}System Check${RESET}\n"

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}✗ Cortex requires macOS${RESET}"
    exit 1
fi
echo -e "${GREEN}✓ macOS detected${RESET}"

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}⚠ Cortex is optimized for Apple Silicon (detected: $ARCH)${RESET}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Apple Silicon detected${RESET}"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${RESET}"
    echo -e "${BLUE}→ Install with: brew install python@3.11${RESET}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 - <<'PY'
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY
then
    echo -e "${RED}✗ Python 3.11+ required (found ${PYTHON_VERSION})${RESET}"
    echo -e "${BLUE}→ Install with: brew install python@3.11${RESET}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${RESET}"

# Clean and setup venv
echo -e "\n${BOLD}Setting Up Environment${RESET}\n"

if [ -d "venv" ]; then
    echo -e "${BLUE}→ Removing existing virtual environment...${RESET}"
    rm -rf venv
fi

echo -e "${BLUE}→ Creating virtual environment...${RESET}"
python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${RESET}"

# Install dependencies
echo -e "\n${BOLD}Installing Dependencies${RESET}\n"

echo -e "${BLUE}→ Upgrading pip...${RESET}"
./venv/bin/pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${RESET}"

echo -e "${BLUE}→ Installing Cortex with dependencies...${RESET}"
if ./venv/bin/pip install -e .; then
    echo -e "${GREEN}✓ Cortex installed${RESET}"
else
    echo -e "${RED}✗ Installation failed${RESET}"
    exit 1
fi

# Create launcher script
echo -e "\n${BOLD}Creating Launcher${RESET}\n"

CORTEX_HOME=$(pwd)
LAUNCHER_SCRIPT="/tmp/cortex_launcher"

cat > "$LAUNCHER_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Cortex launcher"""
import os, sys, subprocess
from pathlib import Path

CORTEX_HOME = Path("CORTEX_HOME_PLACEHOLDER")
VENV_PATH = CORTEX_HOME / "venv"

if not VENV_PATH.exists():
    print(f"Error: Run install.sh in {CORTEX_HOME}")
    sys.exit(1)

env = os.environ.copy()
env["VIRTUAL_ENV"] = str(VENV_PATH)
python_path = VENV_PATH / "bin" / "python"

try:
    subprocess.run([str(python_path), "-m", "cortex"] + sys.argv[1:], env=env)
except KeyboardInterrupt:
    sys.exit(130)
EOF

# Replace placeholder with actual path
sed -i '' "s|CORTEX_HOME_PLACEHOLDER|$CORTEX_HOME|g" "$LAUNCHER_SCRIPT"

# Install to user directory first
mkdir -p ~/.local/bin
cp "$LAUNCHER_SCRIPT" ~/.local/bin/cortex
chmod +x ~/.local/bin/cortex
echo -e "${GREEN}✓ Launcher created at ~/.local/bin/cortex${RESET}"

# Try to create system-wide link
echo -e "${BLUE}→ Attempting system-wide install...${RESET}"

# Check if we can write to /usr/local/bin
if [ -w "/usr/local/bin" ]; then
    ln -sf ~/.local/bin/cortex /usr/local/bin/cortex
    echo -e "${GREEN}✓ System-wide 'cortex' command installed${RESET}"
    INSTALL_LOCATION="/usr/local/bin"
else
    # Try with sudo (will fail in non-interactive, but provide instructions)
    read -p "Install system-wide 'cortex' command with sudo? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if sudo ln -sf ~/.local/bin/cortex /usr/local/bin/cortex 2>/dev/null; then
            echo -e "${GREEN}✓ System-wide 'cortex' command installed${RESET}"
            INSTALL_LOCATION="/usr/local/bin"
        else
            echo -e "${YELLOW}⚠ Could not install system-wide${RESET}"
            echo -e "${BLUE}→ To install system-wide, run:${RESET}"
            echo -e "   ${CYAN}sudo ln -sf ~/.local/bin/cortex /usr/local/bin/cortex${RESET}"
            INSTALL_LOCATION="~/.local/bin"
        fi
    else
        INSTALL_LOCATION="~/.local/bin"
    fi
    
    # Update PATH if needed
    if [[ "$INSTALL_LOCATION" = "~/.local/bin" ]] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        read -p "Add ~/.local/bin to PATH in .zshrc? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo >> ~/.zshrc
            echo "# Cortex" >> ~/.zshrc
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            echo -e "${GREEN}✓ Added ~/.local/bin to PATH in .zshrc${RESET}"
        else
            echo -e "${YELLOW}⚠ ~/.local/bin not added to PATH${RESET}"
        fi
    fi
fi

rm -f "$LAUNCHER_SCRIPT"

# Create directories
echo -e "\n${BOLD}Creating Directories${RESET}\n"
mkdir -p ~/models ~/.cortex
echo -e "${GREEN}✓ Created ~/models and ~/.cortex${RESET}"

# Verify installation
echo -e "\n${BOLD}Verifying Installation${RESET}\n"

if ./venv/bin/python -c "import cortex" 2>/dev/null; then
    echo -e "${GREEN}✓ Cortex imports correctly${RESET}"
else
    echo -e "${RED}✗ Import test failed${RESET}"
fi

# Success message
echo -e "\n${GREEN}${BOLD}"
echo "╭─────────────────────────────────────────────────────╮"
echo "│                INSTALLATION COMPLETE!               │"
echo "╰─────────────────────────────────────────────────────╯"
echo -e "${RESET}\n"

if [ "$INSTALL_LOCATION" = "/usr/local/bin" ]; then
    echo -e "${BOLD}Ready to use!${RESET}\n"
    echo -e "Start Cortex:"
    echo -e "   ${CYAN}cortex${RESET}\n"
else
    echo -e "${BOLD}Next Steps:${RESET}\n"
    echo -e "1. Add to PATH:"
    echo -e "   ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
    echo -e "2. Start Cortex:"
    echo -e "   ${CYAN}cortex${RESET}\n"
fi

echo -e "Download models (inside Cortex):"
echo -e "   ${CYAN}/download${RESET}\n"

echo -e "${BOLD}Alternative (works immediately):${RESET}"
echo -e "   ${CYAN}./venv/bin/python -m cortex${RESET}\n"
