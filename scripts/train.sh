#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/app.yaml}

cd "../" 
git pull
source venv/bin/activate


python -m das_classification.cli train --config "$CONFIG" 2>&1 | tee "$LOG"
