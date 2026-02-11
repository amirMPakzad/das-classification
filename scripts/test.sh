#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/app.yaml}
shift || true   

cd "../"  
source venv/bin/activate

mkdir -p logs
LOG="logs/test_$(date +%Y%m%d_%H%M%S).log"

echo "Config: $CONFIG"
echo "Logging to: $LOG"

python -m das_classification.cli test --config "$CONFIG" "$@" 2>&1 | tee "$LOG"
