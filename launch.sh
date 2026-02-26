#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Run install_all.sh first."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)/SuperGluePretrainedNetwork"
source .venv/bin/activate
python3 ReMap-GUI.py
