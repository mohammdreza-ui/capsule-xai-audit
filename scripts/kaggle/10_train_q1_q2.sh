#!/usr/bin/env bash
# GPU notebook (T4 x2). Q1 + Q2: three split protocols (official, random, video) x seeds for one
# backbone, two jobs at a time (one per GPU). Rerunning skips finished runs and resumes
# interrupted ones. PROTOCOLS="official random" trains only the pre-registered pair.
#   BACKBONE=resnet50 SEEDS="0 1 2" bash scripts/kaggle/10_train_q1_q2.sh
source "$(dirname "$0")/env.sh"
require_prepared
: "${BACKBONE:=resnet50}"
: "${PROTOCOLS:=official random video}"
jobs=$(mktemp)
for seed in $SEEDS; do
  for protocol in $PROTOCOLS; do
    echo "cxa train --set ${PATH_ARGS[*]} train.backbone=$BACKBONE train.split_protocol=$protocol train.test_fold=$FOLD train.seed=$seed" >> "$jobs"
  done
done
run_on_two_gpus "$jobs"
if [[ $(wc -w <<< "$PROTOCOLS") -ge 2 ]]; then
  cxa leakage --runs-root "$RUNS" --backbone "$BACKBONE" --test-fold "$FOLD"
else
  echo "Q1 analysis skipped: it compares at least two split protocols."
fi
cxa calibration --runs-root "$RUNS" --backbone "$BACKBONE" --test-fold "$FOLD" \
  --class-mapping "$REPO_DIR/configs/class_mapping.yaml"
