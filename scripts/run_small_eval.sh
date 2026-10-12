#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${1:-results/small_eval}"

QUERY_GATE="${QUERY_GATE:-off}"
CANDIDATE_GATE="${CANDIDATE_GATE:-on}"
PRODUCT_GATE="${PRODUCT_GATE:-off}"
MAX_STEPS="${MAX_STEPS:-20}"

mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_BIN" -m evaluation.run_experiment \
  --dataset data/eval/webshop_415.json \
  --output-dir "$OUTPUT_DIR" \
  --num-products 1000 \
  --max-steps "$MAX_STEPS" \
  --query-gate "$QUERY_GATE" \
  --candidate-gate "$CANDIDATE_GATE" \
  --product-gate "$PRODUCT_GATE"
