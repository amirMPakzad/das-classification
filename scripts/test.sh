#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=""
CONFIG="configs/app.yaml"

usage() {
    echo "Usage: $0 --run_dir RUN_DIR [--config CONFIG]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Error: --run_dir is required"
    usage
fi

cd ..
source venv/bin/activate

RUN_PATH="runs/$RUN_DIR"
mkdir -p "$RUN_PATH"

LOG="$RUN_PATH/test.log"

echo "Run dir: $RUN_PATH"
echo "Config: $CONFIG"
echo "Logging to: $LOG"

python -m das_classification.cli test \
    --config "$CONFIG" \
    --run-dir "$RUN_PATH" \
    2>&1 | tee "$LOG"
