#!/usr/bin/env bash
# GPU. Needs a finished official-split run from 10_train_q1_q2.sh.
source "$(dirname "$0")/env.sh"
require_prepared
: "${BACKBONE:=resnet50}"
: "${XAI_SEED:=0}"
cxa xai --set "${PATH_ARGS[@]}" xai.run="train/official/$BACKBONE/base/fold$FOLD/seed$XAI_SEED"
