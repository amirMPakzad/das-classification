#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/app.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd ..
source venv/bin/activate

echo "Config: $CONFIG"

python -m das_classification.cli train --config "$CONFIG"
