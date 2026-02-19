#!/usr/bin/env bash
set -e

echo ""
echo " ======================================================"
echo "  PitchBook Observer -- SA Setup"
echo " ======================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " [ERROR] Python 3 not found. Install Python 3.12+ from https://python.org"
    exit 1
fi
PYVER=$(python3 --version 2>&1 | awk '{print $2}')
echo " [OK] Python $PYVER"

# Install dependencies
echo ""
echo " [1/3] Installing dependencies..."
pip install -r requirements.txt --quiet
echo " [OK] Dependencies installed"

# API key setup
echo ""
echo " [2/3] API Key Setup"
if [ -f .env ] && grep -q "ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null; then
    echo " [OK] API key already configured in .env"
else
    echo ""
    echo " Get your key at: https://console.anthropic.com/"
    echo ""
    read -p "  Enter your Anthropic API key (sk-ant-...): " SA_KEY
    if [ -z "$SA_KEY" ]; then
        echo " [WARN] No key entered. Running in template mode (no LLM synthesis)."
    else
        cat > .env <<EOF
# PitchBook Observer - Local Environment
# This file is gitignored and will NOT be committed.

ANTHROPIC_API_KEY=$SA_KEY
CLAUDE_MODEL=claude-opus-4-6
EOF
        echo " [OK] API key saved to .env"
    fi
fi

# Verify
echo ""
echo " [3/3] Verifying setup..."
python3 -c "from pipeline.agents.runtime_agent import RuntimeAgent; print('  [OK] Agent imports OK')" 2>/dev/null || \
    echo "  [WARN] Import check failed - re-run setup or check Python version"

python3 -c "
import pathlib
idx = pathlib.Path('indexes/chroma')
files = [f for f in idx.glob('*') if f.is_file()] if idx.exists() else []
print(f'  [OK] Index ready: {len(files)} files')
" 2>/dev/null

echo ""
echo " ======================================================"
echo "  Setup complete! How to use:"
echo ""
echo "  Web UI (recommended):  bash runit.sh"
echo "  Claude Code:           /pb what funding rounds were announced?"
echo ""
echo "  To refresh data (admin only):  /pipeline"
echo " ======================================================"
echo ""
