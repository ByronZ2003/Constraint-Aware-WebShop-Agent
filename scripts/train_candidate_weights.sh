#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_DIR="${1:-results/candidate_weight_train}"

MAX_STEPS="${MAX_STEPS:-20}"
QUERY_GATE="${QUERY_GATE:-off}"
PRODUCT_GATE="${PRODUCT_GATE:-off}"

export CANDIDATE_ONLINE_WEIGHTS=1
export CANDIDATE_INIT_TYPE="${CANDIDATE_INIT_TYPE:-0.5}"
export CANDIDATE_INIT_REQ="${CANDIDATE_INIT_REQ:-0.5}"
export CANDIDATE_ONLINE_ETA="${CANDIDATE_ONLINE_ETA:-0.1}"

mkdir -p "$OUTPUT_DIR"

export CANDIDATE_ONLINE_LOG="$OUTPUT_DIR/candidate_weights.jsonl"

exec "$PYTHON_BIN" -m evaluation.run_experiment \
  --dataset data/candidate_train/webshop_candidate_train_415.json \
  --output-dir "$OUTPUT_DIR" \
  --num-products 1000 \
  --max-steps "$MAX_STEPS" \
  --query-gate "$QUERY_GATE" \
  --candidate-gate on \
  --product-gate "$PRODUCT_GATE" \
  --webshop-file-path \
    data/candidate_train/items_shuffle_candidate_train_1000.json \
  --webshop-attr-path \
    data/candidate_train/items_ins_v2_candidate_train_1000.json \
  --webshop-index-path \
    data/candidate_train/search_index
