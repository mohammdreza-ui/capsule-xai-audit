#!/usr/bin/env bash
# Install the package into the Kaggle image (internet must be enabled for the notebook).
set -euo pipefail
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
pip install -q -e "$REPO_DIR"
python -c "import capsule_xai_audit, torch; print('cxa', capsule_xai_audit.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L || true
