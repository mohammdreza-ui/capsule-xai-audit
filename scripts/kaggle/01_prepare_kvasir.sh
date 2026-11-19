#!/usr/bin/env bash
# CPU notebook. Builds the Kvasir-Capsule index and the 224x224 uint8 cache (~7 GB) in $WORK.
# Save the notebook output as a Kaggle dataset and attach it as CACHE_INPUT afterwards.
source "$(dirname "$0")/env.sh"
require KVASIR_IMAGES "Attach the Kvasir-Capsule labelled-images dataset, or set KVASIR_IMAGES."

# The Simula zip holds one .tar.gz per class; Kaggle unpacks the zip but not the archives.
# They are extracted outside $WORK so the images do not end up in the notebook output.
if compgen -G "$KVASIR_IMAGES/*.tar.gz" > /dev/null; then
  : "${KVASIR_UNPACKED:=/tmp/kvasir_labelled_images}"
  cxa unpack-kvasir --src "$KVASIR_IMAGES" --out "$KVASIR_UNPACKED"
  KVASIR_IMAGES=$KVASIR_UNPACKED
fi

# metadata.csv (bounding boxes) is not in the Simula zip; fetch it from OSF when not attached.
if [[ -z ${KVASIR_METADATA:-} || ! -f $KVASIR_METADATA ]]; then
  KVASIR_METADATA=$WORK/metadata.csv
  [[ -f $KVASIR_METADATA ]] || cxa fetch-metadata --out "$KVASIR_METADATA"
fi
require KVASIR_METADATA "Download it from https://osf.io/download/kzc8w/ and attach it, or set KVASIR_METADATA."

CACHE_INPUT=$WORK; INDEX_DIR=$WORK/index
if [[ -n ${SPLITS_DIR:-} ]]; then  # local copy of official_splits/, e.g. when GitHub is unreachable
  mkdir -p "$WORK/official_splits" && cp "$SPLITS_DIR"/split_*.csv "$WORK/official_splits/"
else
  cxa fetch-splits --out "$WORK/official_splits"
fi
cxa index-kvasir --images "$KVASIR_IMAGES" --splits "$WORK/official_splits" \
  --metadata "$KVASIR_METADATA" --out "$INDEX_DIR/kvasir.csv" --manifests "$MANIFESTS"
cxa cache --index "$INDEX_DIR/kvasir.csv" --out "$WORK/cache/kvasir_224" --size 224 --workers "$(nproc)"
