# Shared settings for the Kaggle scripts. Source it; override any variable beforehand.
# Unset input paths are looked up below $INPUT_ROOT (the datasets attached to the notebook).
#
#   KVASIR_IMAGES    Kvasir-Capsule labelled images (the folder holding the class folders)
#   KVASIR_METADATA  metadata.csv from the same release (bounding boxes)
#   CV_ROOT          unzipped Capsule Vision 2024 train/val ("Dataset" folder or a parent)
#   CACHE_INPUT      folder holding index/ and cache/ built by 01/02
#   CV_INPUT         folder holding the Capsule Vision cache and external index, when 02 ran in a
#                    separate notebook from 01 (default: same as CACHE_INPUT)
#   WORK             writable output folder (Kaggle keeps /kaggle/working between commits)
set -euo pipefail

: "${REPO_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${WORK:=/kaggle/working}"
: "${INPUT_ROOT:=/kaggle/input}"

# First path below $INPUT_ROOT matching the find expression, or nothing.
find_input() { find -L "$INPUT_ROOT" "$@" -print -quit 2>/dev/null || true; }

if [[ -z "${KVASIR_IMAGES:-}" ]]; then
  # The Simula release unpacks to labelled_images/ (one archive or folder per class).
  KVASIR_IMAGES=$(find_input -type d -iname labelled_images)
fi
if [[ -z "${KVASIR_IMAGES:-}" ]]; then
  # Otherwise: the common parent of two class folders. Matching is case-insensitive; Capsule
  # Vision spells the first class "Angioectasia", so it cannot match here.
  angiectasia=$(find_input -type d -iname angiectasia)
  pylorus=$(find_input -type d -iname pylorus -not -path '*/training/*' -not -path '*/validation/*')
  if [[ -n $angiectasia && -n $pylorus ]]; then
    KVASIR_IMAGES=$(python3 -c 'import os, sys; print(os.path.commonpath(sys.argv[1:]))' "$angiectasia" "$pylorus")
  fi
fi
: "${KVASIR_METADATA:=$(find_input -type f -name metadata.csv)}"
if [[ -z "${CV_ROOT:-}" ]]; then
  cv_class=$(find_input -type d -name Angioectasia -path '*/training/*')
  [[ -n $cv_class ]] && CV_ROOT=$(dirname "$(dirname "$cv_class")")
fi
if [[ -z "${CACHE_INPUT:-}" ]]; then
  if [[ -f $WORK/cache/kvasir_224.json ]]; then
    CACHE_INPUT=$WORK
  else
    meta=$(find_input -type f -name kvasir_224.json -path '*/cache/*')
    CACHE_INPUT=${meta:+$(dirname "$(dirname "$meta")")}
    : "${CACHE_INPUT:=$WORK}"
  fi
fi
if [[ -z "${CV_INPUT:-}" ]]; then
  if [[ -f $CACHE_INPUT/index/cv2024_external.csv ]]; then
    CV_INPUT=$CACHE_INPUT
  else
    ext=$(find_input -type f -name cv2024_external.csv -path '*/index/*')
    CV_INPUT=${ext:+$(dirname "$(dirname "$ext")")}
    : "${CV_INPUT:=$CACHE_INPUT}"
  fi
fi
: "${KVASIR_IMAGES:=}" "${CV_ROOT:=}"
: "${RUNS:=$WORK/runs}"
: "${FOLD:=1}"
: "${SEEDS:=0 1 2 3 4}"

INDEX_DIR="$CACHE_INPUT/index"
MANIFESTS="$WORK/manifests"
LOGS="$WORK/logs"
mkdir -p "$LOGS" "$MANIFESTS"

# --set overrides that point every experiment at the Kaggle paths.
PATH_ARGS=(
  "paths.kvasir_index=$INDEX_DIR/kvasir.csv"
  "paths.kvasir_cache=$CACHE_INPUT/cache/kvasir_224"
  "paths.external_index=$CV_INPUT/index/cv2024_external.csv"
  "paths.external_cache=$CV_INPUT/cache/cv2024_224"
  "paths.class_mapping=$REPO_DIR/configs/class_mapping.yaml"
  "paths.runs_root=$RUNS"
)

cd "$REPO_DIR"

# Stop with a clear message when an input path could not be found.
require() {
  local name=$1 hint=$2
  if [[ -z "${!name:-}" || ! -e "${!name}" ]]; then
    echo "ERROR: $name not found (${!name:-unset}). $hint" >&2
    exit 1
  fi
  echo "$name=${!name}"
}

# GPU steps need the output of 01/02 (index/ and cache/) attached or present in $WORK.
require_prepared() {
  if [[ ! -f $CACHE_INPUT/cache/kvasir_224.json ]]; then
    echo "ERROR: no prepared data found (looked for cache/kvasir_224.json in $CACHE_INPUT and below $INPUT_ROOT)." >&2
    echo "       Attach the output of the preparation notebook (01 + 02) to this notebook." >&2
    exit 1
  fi
  echo "CACHE_INPUT=$CACHE_INPUT"
  echo "CV_INPUT=$CV_INPUT"
  # Continue from an earlier notebook: copy its runs/ (attached as input) into $WORK once.
  if [[ ! -d $RUNS/train ]]; then
    local previous
    previous=$(find_input -type d -path '*/runs/train' -prune)
    if [[ -n $previous ]]; then
      echo "copying earlier runs from $(dirname "$previous")"
      mkdir -p "$RUNS"
      cp -r "$(dirname "$previous")/." "$RUNS/"
    fi
  fi
}

# Run the commands in file $1 on up to two GPUs: line i goes to GPU (i mod n_gpus), one job
# per GPU at a time. Each line is a full shell command; output goes to $LOGS/job_<i>.log.
run_on_two_gpus() {
  local jobs_file=$1 n_gpus worker
  n_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
  (( n_gpus < 1 )) && n_gpus=1
  (( n_gpus > 2 )) && n_gpus=2
  for worker in $(seq 0 $((n_gpus - 1))); do
    (
      i=0
      while IFS= read -r cmd; do
        if (( i % n_gpus == worker )); then
          echo "[gpu$worker] job $i: $cmd"
          CUDA_VISIBLE_DEVICES=$worker bash -c "$cmd" > "$LOGS/job_$i.log" 2>&1 \
            || echo "[gpu$worker] job $i FAILED, see $LOGS/job_$i.log"
        fi
        i=$((i + 1))
      done < "$jobs_file"
    ) &
  done
  wait
}
