#!/usr/bin/env bash
# GPU (single T4 is enough): feature extraction once per backbone, then CPU probes.
source "$(dirname "$0")/env.sh"
require_prepared
cxa fewshot --set "${PATH_ARGS[@]}" fewshot.test_fold="$FOLD"
