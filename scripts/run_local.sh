#!/usr/bin/env bash
# Run the whole pipeline on this machine: install, preflight (with unit tests), prepare,
# smoke, every experiment of the profile, and the report.
#
#   bash scripts/run_local.sh                     # full profile, datasets found below data/
#   PROFILE=minimum bash scripts/run_local.sh     # minimum | core | full
#   bash scripts/run_local.sh --seeds 0 1 2       # further flags go to scripts/run_all.py
#   bash scripts/run_local.sh --dry-run           # print the job plan and stop
#   INPUT_ROOT=/mnt/datasets bash scripts/run_local.sh
#
# Activate the Python environment to install into first, or set PYTHON. Put the downloads
# below data/ (or INPUT_ROOT): the Kvasir-Capsule labelled images (the Simula zip, its
# per-class .tar.gz archives or the extracted class folders) and the Capsule Vision 2024
# train/val zip or folder. metadata.csv and the official split are downloaded when missing.
# Outputs: results/RESULTS.md, manifests/, runs/ (git-ignored), data/logs/.
# Re-running continues where it stopped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
PROFILE="${PROFILE:-full}"
INPUT_ROOT="${INPUT_ROOT:-$REPO/data}"

cd "$REPO"
"$PYTHON" -m pip install -q -e ".[dev]"
"$PYTHON" scripts/run_all.py --profile "$PROFILE" --input-root "$INPUT_ROOT" "$@"
echo "report: $REPO/results/RESULTS.md"
