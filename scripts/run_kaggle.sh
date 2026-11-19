#!/usr/bin/env bash
# Run the whole pipeline in a Kaggle notebook and pack the outputs for download.
#
# Notebook settings: Accelerator "GPU T4 x2", Internet on, and as inputs the Kvasir-Capsule
# labelled images and Capsule Vision 2024 train/val (any layout scripts/run_all.py finds
# below /kaggle/input). notebooks/kaggle_run.ipynb gets the code and calls this script.
#
#   !bash /kaggle/working/capsule-xai-audit/scripts/run_kaggle.sh
#   !PROFILE=minimum bash /kaggle/working/capsule-xai-audit/scripts/run_kaggle.sh --seeds 0 1 2
#
# Everything is written below /kaggle/working, which Kaggle keeps as the notebook output;
# image caches go to /tmp. No new job starts after 11 h. To continue a stopped run, attach
# its output as an input of a new notebook and run this again. Afterwards
# /kaggle/working/cxa_results.tar.gz holds results/, manifests/, logs/, index/ and runs/
# without model weights and cached features; it is written after a failure too.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_ROOT="${CXA_KAGGLE_ROOT:-/kaggle}"  # CXA_KAGGLE_ROOT: simulated /kaggle for testing
WORK="$KAGGLE_ROOT/working"
PROFILE="${PROFILE:-full}"

if [ ! -d "$KAGGLE_ROOT/input" ]; then
    echo "$KAGGLE_ROOT/input not found: this is not a Kaggle notebook; use scripts/run_local.sh" >&2
    exit 1
fi

python "$REPO/scripts/run_all.py" --profile "$PROFILE" "$@"
status=$?

cd "$WORK" || exit 1
parts=()
for d in results manifests logs index runs; do
    [ -e "$d" ] && parts+=("$d")
done
if [ ${#parts[@]} -gt 0 ]; then
    tar --exclude='*.pt' --exclude='runs/features' -czf cxa_results.tar.gz "${parts[@]}"
    echo "packed ${parts[*]} into $WORK/cxa_results.tar.gz ($(du -h cxa_results.tar.gz | cut -f1))"
fi
exit "$status"
