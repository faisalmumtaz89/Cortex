#!/bin/bash

# Apple Silicon Optimization Tool for Cortex
# Optional tool to verify and optimize Apple Silicon settings

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}Apple Silicon Optimization Check${NC}"
echo -e "${BLUE}=================================${NC}"

# Quick verification
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "This tool is for Apple Silicon Macs only"
    exit 1
fi

echo -e "${GREEN}✅ Apple Silicon detected${NC}"

# Run validation test
echo -e "\n${BLUE}Running GPU validation...${NC}"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    python tests/test_apple_silicon.py 2>&1 | grep -E "(✅|❌|VALIDATION|GFLOPS)" | tail -10
else
    echo "Please run ./install.sh first"
    exit 1
fi

echo -e "\n${BLUE}Optimization Tips:${NC}"
echo "1. Ensure Xcode Command Line Tools are installed"
echo "2. Keep macOS updated for latest Metal improvements"
echo "3. Prefer MLX models (mlx-community) or convert to MLX with 4/8-bit quantization"
echo "4. Monitor GPU usage with Activity Monitor > Window > GPU History"

echo -e "\n${GREEN}Done! Run 'cortex' to start.${NC}"
