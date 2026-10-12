#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw_webshop"

mkdir -p "$OUT"

echo "============================================================"
echo "FULL WEBSHOP DOWNLOAD"
echo "============================================================"
echo "Output: $OUT"
echo

download_file () {
    NAME="$1"
    URL="https://hf-mirror.com/datasets/YWZBrandon/webshop-data/resolve/main/${NAME}?download=true"

    echo
    echo "------------------------------------------------------------"
    echo "Downloading: $NAME"
    echo "------------------------------------------------------------"

    if command -v wget >/dev/null 2>&1; then

        wget \
          -c \
          --progress=dot:giga \
          -O "$OUT/$NAME" \
          "$URL"

    elif command -v curl >/dev/null 2>&1; then

        curl \
          -L \
          --retry 10 \
          --retry-delay 5 \
          -C - \
          -o "$OUT/$NAME" \
          "$URL"

    else

        echo "ERROR: wget or curl is required."
        exit 1
    fi
}

download_file "items_shuffle.json"
download_file "items_ins_v2.json"
download_file "items_human_ins.json"

echo
echo "============================================================"
echo "FILE SIZES"
echo "============================================================"

ls -lh \
  "$OUT/items_shuffle.json" \
  "$OUT/items_ins_v2.json" \
  "$OUT/items_human_ins.json"

echo
echo "============================================================"
echo "SHA256"
echo "============================================================"

sha256sum \
  "$OUT/items_shuffle.json" \
  "$OUT/items_ins_v2.json" \
  "$OUT/items_human_ins.json"

echo
echo "============================================================"
echo "VERIFY KNOWN LARGE-FILE HASHES"
echo "============================================================"

EXPECTED_PRODUCTS="2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"
EXPECTED_ATTRS="1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e"

ACTUAL_PRODUCTS="$(sha256sum "$OUT/items_shuffle.json" | awk '{print $1}')"
ACTUAL_ATTRS="$(sha256sum "$OUT/items_ins_v2.json" | awk '{print $1}')"

if [ "$ACTUAL_PRODUCTS" != "$EXPECTED_PRODUCTS" ]; then
    echo "ERROR: items_shuffle.json SHA256 mismatch"
    exit 1
fi

if [ "$ACTUAL_ATTRS" != "$EXPECTED_ATTRS" ]; then
    echo "ERROR: items_ins_v2.json SHA256 mismatch"
    exit 1
fi

echo "HASH CHECK: PASS"

touch "$OUT/.download_complete"

echo
echo "============================================================"
echo "FULL WEBSHOP DOWNLOAD COMPLETE"
echo "============================================================"
