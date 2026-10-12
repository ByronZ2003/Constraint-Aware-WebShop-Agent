#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "============================================================"
echo "Constraint-Aware WebShop environment setup"
echo "============================================================"

echo
echo "Python:"
"$PYTHON_BIN" --version

echo
echo "Java:"
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: Java is required by Pyserini/Lucene."
    echo "Install Java 11, then rerun this script."
    exit 1
fi

java -version

echo
echo "Installing Python requirements..."

"$PYTHON_BIN" -m pip install \
    -r requirements.txt

echo
echo "Checking spaCy model..."

if "$PYTHON_BIN" - <<'PY'
import spacy
spacy.load("en_core_web_sm")
print("en_core_web_sm already installed")
PY
then
    :
else
    echo "Installing en_core_web_sm..."
    "$PYTHON_BIN" -m spacy download \
        en_core_web_sm
fi

echo
echo "Checking required Python modules..."

"$PYTHON_BIN" - <<'PY'
import ijson
import openai
import pyserini
import spacy

spacy.load("en_core_web_sm")

print("openai:", openai.__version__)
print("Environment dependency check: PASS")
PY

echo
echo "============================================================"
echo "ENVIRONMENT SETUP COMPLETE"
echo "============================================================"
