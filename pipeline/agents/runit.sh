#!/usr/bin/env bash
# =============================================================
#  PitchBook Observer Agent - Launch from Agent Hub
# =============================================================

echo ""
echo "  ======================================================"
echo "  PitchBook Observer Agent - Starting..."
echo "  ======================================================"
echo ""

# Navigate to project root (two levels up)
cd "$(dirname "$0")/../.."

# Delegate to the main launcher
if [ -f runit.sh ]; then
    bash runit.sh
else
    echo "  Agent could not locate its modules."
    echo "  Make sure the full project structure is intact."
    exit 1
fi
