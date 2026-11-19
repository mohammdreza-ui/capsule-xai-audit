#!/usr/bin/env bash
# CPU notebook. Needs the Kvasir cache from 01 (CACHE_INPUT). Builds the Capsule Vision cache,
# removes its Kvasir-Capsule frames by pHash + NCC and writes the external index.
source "$(dirname "$0")/env.sh"
require CV_ROOT "Attach the Capsule Vision 2024 train/val dataset, or set CV_ROOT."
require_prepared
cxa index-cv --root "$CV_ROOT" --out "$WORK/index/cv2024.csv"
cxa cache --index "$WORK/index/cv2024.csv" --out "$WORK/cache/cv2024_224" --size 224 --workers "$(nproc)"
cxa dedup --kvasir-index "$INDEX_DIR/kvasir.csv" --kvasir-cache "$CACHE_INPUT/cache/kvasir_224" \
  --cv-index "$WORK/index/cv2024.csv" --cv-cache "$WORK/cache/cv2024_224" \
  --out "$WORK/index/cv2024_external.csv" --manifests "$MANIFESTS"
cat "$MANIFESTS/cv2024_dedup_summary.json"
