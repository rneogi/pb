#!/usr/bin/env bash
# =============================================================
#  PitchBook Observer Agent - One-Click Launcher
# =============================================================
set -e

echo ""
echo "  ======================================================"
echo "  PitchBook Observer Agent"
echo "  Powered by Claude Opus"
echo "  ======================================================"
echo ""

# ---- Quietly check Python ----
if ! command -v python3 &>/dev/null; then
    echo "  The agent needs a runtime component to start."
    echo "  Please install Python 3.12+ from https://www.python.org/downloads/"
    echo "  Then run this again."
    exit 1
fi

cd "$(dirname "$0")"

# ---- Agent boot sequence (hide the plumbing) ----
echo "  Agent is initializing..."
echo ""

echo "    [1/3] Loading agent modules..."
pip3 install -q pyyaml "httpx[http2]" beautifulsoup4 lxml trafilatura \
    fastapi uvicorn pydantic numpy scikit-learn python-dotenv anthropic 2>/dev/null
pip3 install -q sentence-transformers 2>/dev/null || true
pip3 install -q streamlit plotly pandas 2>/dev/null

echo "    [2/3] Preparing knowledge base..."
mkdir -p data/{raw,clean,meta,events,memory,responses}
mkdir -p indexes/chroma db products batch_results runs
python3 scripts/inject_demo_data.py >/dev/null 2>&1

echo "    [3/3] Launching agent..."
echo ""
echo "  ======================================================"
echo "  PitchBook Observer Agent is starting!"
echo "  Your browser will open automatically."
echo ""
echo "  If it doesn't, go to: http://localhost:8501"
echo "  ======================================================"
echo ""

python3 -m streamlit run app/streamlit_chat.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
