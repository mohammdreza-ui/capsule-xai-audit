#!/usr/bin/env bash
# GPU (T4 x2). Trains Lab-augmented models (tag lab-aug) for the same seeds as the base
# official-split runs, then evaluates both tags under controlled CIELAB casts.
source "$(dirname "$0")/env.sh"
require_prepared
: "${BACKBONE:=resnet50}"
: "${MAX_DELTA_E:=20}"
jobs=$(mktemp)
for seed in $SEEDS; do
  echo "cxa train --set ${PATH_ARGS[*]} train.backbone=$BACKBONE train.test_fold=$FOLD train.seed=$seed train.tag=lab-aug train.color_aug_max_delta_e=$MAX_DELTA_E" >> "$jobs"
done
run_on_two_gpus "$jobs"
cxa color --set "${PATH_ARGS[@]}" \
  "color.runs=[train/official/$BACKBONE/base/fold$FOLD/seed*, train/official/$BACKBONE/lab-aug/fold$FOLD/seed*]"
