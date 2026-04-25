#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Run install_all.sh first."
    exit 1
fi

if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: Node.js not found. Install Node.js 20+ before launching the new desktop UI."
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "ERROR: Frontend dependencies are missing."
    echo "Run: npm install"
    echo "You can still launch the previous UI with ./launch_legacy.sh"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)/SuperGluePretrainedNetwork"
source .venv/bin/activate
python3 desktop_backend.py &
npm run desktop:dev
