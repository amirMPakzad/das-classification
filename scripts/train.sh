#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/app.yaml}"
RUN_NAME="${2:-EXP}"

cd "../" 
git pull
source venv/bin/activate

mkdir -p logs
LOG="logs/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG"

python -m das_classification.cli train \
    --config "$CONFIG"\
    --run-name "$RUN_NAME"\
    2>&1 | tee "$LOG"
