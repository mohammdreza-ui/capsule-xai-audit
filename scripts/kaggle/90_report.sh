#!/usr/bin/env bash
# CPU. Collects runs/analysis into results/ (copy results/ and manifests/ back into the repo).
source "$(dirname "$0")/env.sh"
cxa report --runs-root "$RUNS" --out "$WORK/results"
