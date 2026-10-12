#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

RESOURCE_DIR="data/candidate_train/search_resources"
INDEX_DIR="data/candidate_train/search_index"

echo "Generating Candidate search documents..."

"$PYTHON_BIN" \
    scripts/build_candidate_train_search_docs.py

echo
echo "Rebuilding Candidate Lucene index..."

rm -rf "$INDEX_DIR"

"$PYTHON_BIN" -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$RESOURCE_DIR" \
    --index "$INDEX_DIR" \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1 \
    --storePositions \
    --storeDocvectors \
    --storeRaw

rm -rf "$RESOURCE_DIR"

echo
echo "Candidate index rebuilt:"
du -sh "$INDEX_DIR"
