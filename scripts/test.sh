#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?You must provide RUN_DIR as first argument}"
shift

CONFIG="${1:-configs/app.yaml}"
shift || true

cd ..
source venv/bin/activate

mkdir -p logs

SAFE_RUN_DIR=$(echo "$RUN_DIR" | tr ' /' '__')
LOG="logs/test_${SAFE_RUN_DIR}_$(date +%Y%m%d_%H%M%S).log"

echo "Run dir: $RUN_DIR"
echo "Config: $CONFIG"
echo "Logging to: $LOG"

python -m das_classification.cli test \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    "$@" \
    2>&1 | tee "$LOG"
