#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/app.yaml}

cd "../" 
git pull
source venv/bin/activate

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG"

python -m das_classification.cli train --config "$CONFIG" 2>&1 | tee "$LOG"
