#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./scripts/test.sh                # uses default config
#   ./scripts/test.sh configs/app.yaml

CONFIG=${1:-configs/app.yaml}

cd "../"  

source venv/bin/activate


mkdir -p logs
LOG="logs/test_$(date +%Y%m%d_%H%M%S).log"

echo "Config: $CONFIG"
echo "Logging to: $LOG"

python -m das_classification.cli test --config "$CONFIG" 2>&1 | tee "$LOG"
