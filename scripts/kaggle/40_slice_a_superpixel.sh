#!/usr/bin/env bash
# GPU (light). Reuses the DINOv2 features cached by 20_fewshot_q3.sh when present.
source "$(dirname "$0")/env.sh"
require_prepared
cxa superpixel --set "${PATH_ARGS[@]}" superpixel.test_fold="$FOLD"
